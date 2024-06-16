
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
    src: u32,
    dst: u32,
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
struct Crf1de {
    opt: Crf1deOpt,
    features: Vec<Feat>,
    attrs: Vec<FeatRefs>,
    forward_trans: Vec<FeatRefs>,
    ctx: Crf1dContext,
}

impl Crf1de {
    pub fn num_labels(&self) -> usize {
        self.forward_trans.len()
    }

    fn num_features(&self) -> usize {
        self.features.len()
    }

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
                        * weight
                }
            }
        }

        /* Loop over the labels (t, i) */
        for i in 0..L {
            let edge = &self.forward_trans[i];
            for &fid in edge {
                let f = &self.features[fid];
                w[fid] += self.ctx.mexp_trans[self.ctx.num_labels * (i) + (f.dst as usize)] * weight
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
    attrs: &mut Vec<FeatRefs>,
    forward_trans: &mut Vec<FeatRefs>,
    features: &Vec<Feat>,
) {
    let K = features.len();
    let A = attrs.len();
    let L = forward_trans.len();
    log::info!("generate: A: {A}, K: {K}, L: {L}");

    /*
        The purpose of this routine is to collect references (indices) of:
        - state features fired by each attribute (attributes)
        - transition features pointing from each label (trans)
    */
    for k in 0..K {
        let f = &features[k];
        match &f.ftype {
            FeatType::FT_STATE => attrs[f.src as usize].push(k),
            FeatType::FT_TRANS => forward_trans[f.src as usize].push(k),            
        }
    }
}

fn crf1df_generate(
    ds: &Dataset,
    connect_all_attrs: bool,
    connect_all_edges: bool,
    feature_min_freq: f64,
) -> Vec<Feat> {
    let N = ds.len();
    let L = ds.num_labels();
    log::info!("crf1df_generate: (N: {N}, L: {L})");
    let mut set = FeatSet::default();
    for s in 0..N {
        let mut prev = L;
        let mut curr = 0;
        let seq = ds.get(s);
        let T = seq.len();

        assert!(T > 0, "unexpected empty sequence");

        /* Loop over the items in the sequence. */
        for t in 0..T {
            let item = &seq.items[t];
            curr = seq.labels[t];

            /* Transition feature: label #prev -> label #(item->yid).
            Features with previous label #L are transition BOS. */
            if prev != L {
                set.add(Feat {
                    ftype: FeatType::FT_TRANS,
                    src: prev as u32,
                    dst: curr as u32,
                    freq: seq.weight,
                })
            }

            for attr in item {
                /* State feature: attribute #a -> state #(item->yid). */
                set.add(Feat {
                    ftype: FeatType::FT_STATE,
                    src: attr.id as u32,
                    dst: curr as u32,
                    freq: seq.weight * attr.value,
                });

                /* Generate state features connecting attributes with all
                output labels. These features are not unobserved in the
                training data (zero expexcations). */
                if connect_all_attrs {
                    for i in 0..L {
                        set.add(Feat {
                            ftype: FeatType::FT_STATE,
                            src: attr.id as u32,
                            dst: i as u32,
                            freq: 0.0,
                        });
                    }
                }
            }

            prev = curr;
        }
        // log::info!("progress: {s}/{N}");
    }
    log::info!("finished {}", set.m.len());
    /* Generate edge features representing all pairs of labels.
    These features are not unobserved in the training data
    (zero expexcations). */
    if connect_all_edges {
        for i in 0..L {
            for j in 0..L {
                set.add(Feat {
                    ftype: FeatType::FT_TRANS,
                    src: i as u32,
                    dst: j as u32,
                    freq: 0.0,
                });
            }
        }
    }
    set.to_vec(feature_min_freq)
}

#[derive(Debug)]
pub struct TagEncoder {
    internal: Crf1de,
}

impl TagEncoder {
    pub fn new() -> Self {
        Self {
            internal: Crf1de::default(),
        }
    }

    /* LEVEL_NONE -> LEVEL_NONE. */
    pub(crate) fn objective_and_gradients_batch(
        &mut self,
        ds: &Dataset,
        w: &[f64],
        g: &mut [f64],
    ) -> f64 {
        let N = ds.len();
        let K = self.internal.num_features();

        // Initialize the gradients with observation expectations.
        for i in 0..K {
            g[i] = -self.internal.features[i].freq;
        }

        /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
        */
        self.internal.ctx.reset(ResetOpt::RF_TRANS);
        self.internal.transition_score(w);
        self.internal.ctx.crf1dc_exp_transition();

        // Compute model expectations.
        let mut logp = 0.0;
        let mut logl = 0.0;
        for seq in &ds.seqs {
            /* Set label sequences and state scores. */
            self.internal.ctx.crf1dc_set_num_items(seq.len());
            self.internal.ctx.reset(ResetOpt::RF_STATE);
            self.internal.state_score(seq, w);
            self.internal.ctx.crf1dc_exp_state();

            /* Compute forward/backward scores. */
            self.internal.ctx.crf1dc_alpha_score();
            self.internal.ctx.crf1dc_beta_score();
            self.internal.ctx.crf1dc_marginals();

            /* Compute the probability of the input sequence on the model. */
            logp = self.internal.ctx.crf1dc_score(&seq.labels) - self.internal.ctx.crf1dc_lognorm();
            /* Update the log-likelihood. */
            logl += logp * seq.weight;

            /* Update the model expectations of features. */
            self.internal.model_expectation(seq, g, seq.weight);
        }
        -logl
    }

    pub fn num_features(&self) -> usize {
        self.internal.num_features()
    }

    pub(crate) fn set_data(&mut self, ds: &Dataset) {
        self.internal.set_data(ds);
    }
}

pub trait Crf1dTrainer {
    fn train(&mut self, data: &Dataset, fpath: PathBuf, holdout: usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_data() {
        let mut o = Crf1de::default();
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
        let o = TagEncoder::new();
    }
}
