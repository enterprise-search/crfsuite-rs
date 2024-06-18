use serde::{Deserialize, Serialize};

use super::crf1d::{
    context::{Crf1dContext, CtxOpt},
    model::FeatRefs,
};
use crate::{crf::crf1d::context::ResetOpt, Dataset, Sequence};
use std::{collections::HashSet, convert::TryFrom, hash::Hash, path::PathBuf, time::Instant};

#[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Clone, Copy)]
pub(crate) enum FeatType {
    FT_STATE = 0,
    FT_TRANS = 1,
}

impl TryFrom<u32> for FeatType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            x if x == Self::FT_STATE as u32 => Ok(Self::FT_STATE),
            x if x == Self::FT_TRANS as u32 => Ok(Self::FT_TRANS),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
struct Feat {
    ftype: FeatType,
    src: usize,
    dst: usize,
    freq: f64,
}

impl Hash for Feat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ftype.hash(state);
        self.src.hash(state);
        self.dst.hash(state);
    }
}

impl PartialEq for Feat {
    fn eq(&self, other: &Self) -> bool {
        self.ftype == other.ftype && self.src == other.src && self.dst == other.dst
    }
}

impl Eq for Feat {}

#[derive(Debug, Default)]
struct Crf1deOpt {
    feature_possible_states: bool,
    feature_possible_transitions: bool,
    feature_minfreq: f64,
}

#[derive(Debug, Default)]
pub struct Crf1dEncoder {
    opt: Crf1deOpt,
    features: Vec<Feat>,
    attrs: Vec<FeatRefs>,
    forward_trans: Vec<FeatRefs>,
    ctx: Crf1dContext,
}

impl Crf1dEncoder {
    // #[inline]
    pub fn num_labels(&self) -> usize {
        self.forward_trans.len()
    }

    // #[inline]
    pub(crate) fn num_features(&self) -> usize {
        self.features.len()
    }

    fn state_score(&mut self, seq: &crate::Sequence, w: &[f64]) {
        let T = seq.len();

        /* Loop over the items in the sequence. */
        for t in 0..T {
            let item = &seq.items[t];

            /* Loop over the contents (attributes) attached to the item. */
            for attr in item {
                /* Access the list of state features associated with the attribute. */
                let attr_feat_refs = &self.attrs[attr.id as usize];

                /* Loop over the state features associated with the attribute. */
                for &fid in attr_feat_refs {
                    /* State feature associates the attribute #a with the label #(f->dst). */
                    let f = &self.features[fid];
                    self.ctx.state[self.ctx.num_labels * (t) + f.dst as usize] +=
                        w[fid] * attr.value;
                }
            }
        }
    }

    fn transition_score(&mut self, w: &[f64]) {
        for i in 0..self.num_labels() {
            let edge = &self.forward_trans[i];
            for &fid in edge {
                self.ctx.trans[self.ctx.num_labels * i + self.features[fid].dst as usize] = w[fid];
            }
        }
    }

    fn model_expectation(&self, seq: &Sequence, w: &mut [f64], weight: f64) {
        let T = seq.len();
        let L = self.num_labels();

        for t in 0..T {
            /* Compute expectations for state features at position #t. */
            /* Access the attribute. */
            for attr in &seq.items[t] {
                /* Loop over state features for the attribute. */
                for &fid in &self.attrs[attr.id as usize] {
                    let f = &self.features[fid];
                    w[fid] += self.ctx.mexp_state[self.ctx.num_labels * (t) + (f.dst as usize)]
                        * attr.value
                        * weight;
                }
            }
        }

        /* Loop over the labels (t, i) */
        for i in 0..L {
            let edge = &self.forward_trans[i];
            for &fid in edge {
                let f = &self.features[fid];
                w[fid] +=
                    self.ctx.mexp_trans[self.ctx.num_labels * (i) + (f.dst as usize)] * weight;
            }
        }
    }
}

#[derive(Debug, Default)]
struct FeatSet {
    m: HashSet<Feat>,
}

impl FeatSet {
    fn add(&mut self, f: Feat) {
        let mut f = f;
        let mut found = false;
        if let Some(p) = self.m.get(&f) {
            f.freq += p.freq;
            found = true;
        }
        if found {
            self.m.remove(&f);
        }
        self.m.insert(f);
    }

    fn to_vec(self, min_freq: f64) -> Vec<Feat> {
        self.m.into_iter().filter(|x| x.freq >= min_freq).collect()
    }
}

fn crf1df_init_references(
    attrs: &mut [FeatRefs],
    forward_trans: &mut [FeatRefs],
    features: &[Feat],
) {
    /*
        The purpose of this routine is to collect references (indices) of:
        - state features fired by each attribute (attributes)
        - transition features pointing from each label (trans)
    */
    // for k in 0..K {
    // let f = &features[k];
    features
        .iter()
        .enumerate()
        .for_each(|(k, f)| match &f.ftype {
            FeatType::FT_STATE => attrs[f.src as usize].push(k),
            FeatType::FT_TRANS => forward_trans[f.src as usize].push(k),
        });
}

fn crf1df_generate(
    ds: &Dataset,
    connect_all_attrs: bool,
    connect_all_edges: bool,
    feature_min_freq: f64,
) -> Vec<Feat> {
    let L = ds.num_labels();
    let mut set = FeatSet::default();
    for seq in &ds.seqs {
        let mut prev = L;

        assert!(seq.len() > 0, "unexpected empty sequence");

        /* Loop over the items in the sequence. */
        for t in 0..seq.len() {
            let item = &seq.items[t];
            let curr = seq.labels[t];

            /* Transition feature: label #prev -> label #(item->yid).
            Features with previous label #L are transition BOS. */
            if prev != L {
                set.add(Feat {
                    ftype: FeatType::FT_TRANS,
                    src: prev,
                    dst: curr,
                    freq: seq.weight,
                })
            }

            for attr in item {
                /* State feature: attribute #a -> state #(item->yid). */
                set.add(Feat {
                    ftype: FeatType::FT_STATE,
                    src: attr.id,
                    dst: curr,
                    freq: seq.weight * attr.value,
                });

                /* Generate state features connecting attributes with all
                output labels. These features are not unobserved in the
                training data (zero expexcations). */
                if connect_all_attrs {
                    for i in 0..L {
                        set.add(Feat {
                            ftype: FeatType::FT_STATE,
                            src: attr.id,
                            dst: i,
                            freq: 0.0,
                        });
                    }
                }
            }

            prev = curr;
        }
        // log::info!("progress: {s}/{N}");
    }
    /* Generate edge features representing all pairs of labels.
    These features are not unobserved in the training data
    (zero expexcations). */
    if connect_all_edges {
        for i in 0..L {
            for j in 0..L {
                set.add(Feat {
                    ftype: FeatType::FT_TRANS,
                    src: i,
                    dst: j,
                    freq: 0.0,
                });
            }
        }
    }
    set.to_vec(feature_min_freq)
}

/**
 * Interface for a graphical model.
 */
pub(crate) trait Encoder {
    /// initializes the encoder with a training data set
    fn set_data(&mut self, ds: &Dataset);
    /// compute the objective value and gradients for the whole data set.
    fn objective_and_gradients_batch(&mut self, ds: &Dataset, w: &[f64], g: &mut [f64]) -> f64;
}

impl Encoder for Crf1dEncoder {
    fn set_data(&mut self, ds: &Dataset) {
        let L = ds.num_labels();
        let A = ds.num_attrs();
        let N = ds.len();
        let T = ds.max_seq_length();
        log::info!("set data (L: {L}, A: {A}, N: {N}, T: {T})");
        self.ctx = Crf1dContext::new(CtxOpt::CTXF_VITERBI | CtxOpt::CTXF_MARGINALS, L, T);
        log::info!("TODO: opts info");
        log::info!("feature generation (type: crf1d, min_freq: {}, possible_states: {}, possible_transitions: {}", self.opt.feature_minfreq, self.opt.feature_possible_states, self.opt.feature_possible_transitions);
        let begin = Instant::now();
        self.opt.feature_possible_transitions = true;
        self.features = crf1df_generate(
            ds,
            self.opt.feature_possible_states,
            self.opt.feature_possible_transitions,
            self.opt.feature_minfreq,
        );
        log::info!(
            "number of features: {}, time cost: {:?}",
            self.features.len(),
            begin.elapsed()
        );
        self.attrs.resize(A, FeatRefs::default());
        self.forward_trans.resize(L, FeatRefs::default());
        crf1df_init_references(&mut self.attrs, &mut self.forward_trans, &self.features);
    }

    /* LEVEL_NONE -> LEVEL_NONE. */
    fn objective_and_gradients_batch(&mut self, ds: &Dataset, w: &[f64], g: &mut [f64]) -> f64 {
        // Initialize the gradients with observation expectations.
        for i in 0..self.num_features() {
            g[i] = -self.features[i].freq;
        }

        /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
        */
        self.ctx.reset(ResetOpt::RF_TRANS);
        self.transition_score(w);
        self.ctx.exp_transition();

        // Compute model expectations.
        let mut logl = 0.0;
        for seq in &ds.seqs {
            /* Set label sequences and state scores. */
            self.ctx.resize(seq.len());
            self.ctx.reset(ResetOpt::RF_STATE);
            self.state_score(seq, w);
            self.ctx.exp_state();

            /* Compute forward/backward scores. */
            self.ctx.alpha_score();
            self.ctx.beta_score();
            self.ctx.marginals();

            /* Compute the probability of the input sequence on the model. */
            let logp = self.ctx.score(&seq.labels) - self.ctx.lognorm();
            /* Update the log-likelihood. */
            logl += logp * seq.weight;

            /* Update the model expectations of features. */
            self.model_expectation(seq, g, seq.weight);
        }
        -logl
    }
}

pub trait Crf1dTrainer {
    fn train(&mut self, data: &Dataset, fpath: PathBuf, holdout: usize);
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use test::Bencher;

    use super::*;

    #[test]
    fn set_data() {
        let mut o = Crf1dEncoder::default();
        assert_eq!(o.num_features(), 0);
        assert_eq!(o.num_labels(), 0);
        let s = "P\thello\tworld
        Q\thi\tthere\n\n";
        let ds = Dataset::from(s.lines());
        o.set_data(&ds);
        assert_eq!(o.num_labels(), 2);
        assert_eq!(o.num_features(), 8);
        assert_eq!(o.attrs.len(), 4);
        assert_eq!(o.forward_trans.len(), 2);
    }

    #[test]
    fn feat() {
        assert_eq!(1, std::mem::size_of::<FeatType>());
        assert_eq!(24, std::mem::size_of::<Feat>());
    }

    #[test]
    fn objective_and_gradients_batch() {
        // let xseq = vec![
        //     vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        //     vec![Attribute::new("walk", 1.0)],
        //     vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        //     vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        //     vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        //     vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
        //     vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        //     vec![],
        //     vec![Attribute::new("clean", 1.0)],
        // ];
        let yseq = [
            "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
        ];
        let o = Crf1dEncoder::default();
    }

    #[test]
    fn feature_generate() {
        let f = File::open("test.data").expect("failed to open file");
        let ds = Dataset::try_from(f).expect("failed to read file");
        let feats = crf1df_generate(&ds, false, true, 0.0);
        assert_eq!(feats.len(), 137415);
    }

    #[bench]
    fn bench_feature(b: &mut Bencher) {
        // 112,398,766.60 ns/iter (+/- 25,375,252.00)
        //  86,646,579.20 ns/iter (+/- 2,964,393.23)
        let f = File::open("test.data").expect("failed to open file");
        let ds = Dataset::try_from(f).expect("failed to read file");
        b.iter(|| {
            let feats = crf1df_generate(&ds, false, true, 0.0);
            assert_eq!(feats.len(), 137415);
        });
    }

    #[bench]
    fn bench_feature_init_refs(b: &mut Bencher) {
        // 1,029,148.44 ns/iter (+/- 511,912.35)
        let f = File::open("test.data").expect("failed to open file");
        let ds = Dataset::try_from(f).expect("failed to read file");
        let feats = crf1df_generate(&ds, false, true, 0.0);
        let A = ds.num_attrs();
        let L = ds.num_labels();
        let mut attrs = vec![FeatRefs::default(); A];
        let mut trans = vec![FeatRefs::default(); L];
        b.iter(|| {
            crf1df_init_references(&mut attrs, &mut trans, &feats);
        });
    }
}
