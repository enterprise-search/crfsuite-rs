use std::{collections::HashSet, ffi::CStr, hash::Hash, mem::MaybeUninit, path::PathBuf, time::Instant};

use clap::Parser;
use libc::c_void;
use liblbfgs_sys::{
    lbfgs, lbfgs_free, lbfgs_malloc, lbfgs_parameter_init, lbfgs_parameter_t, lbfgs_strerror,
    LBFGS_LINESEARCH_MORETHUENTE,
};

use crate::{crf::crf1d::context::ResetOpt, Algorithm, Dataset};

use super::crf1d::{context::{Crf1dContext, Opt}, model::FeatRefs};

#[derive(Debug, Default)]
struct Feat {
    ftype: u32,
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

impl Eq for Feat {
    
}

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
    ctx: Option<Crf1dContext>,
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
        self.ctx = Some(Crf1dContext::new(&[Opt::CTXF_VITERBI, Opt::CTXF_MARGINALS], L, T));
        log::info!("TODO: opts info");
        let begin = Instant::now();
        self.features = crf1df_generate(ds, self.opt.feature_possible_states, self.opt.feature_possible_transitions, self.opt.feature_minfreq);
        log::info!("number of features: {}", self.features.len());
        log::info!("took: {:?}", begin.elapsed());
        self.attrs.resize(A, FeatRefs::default());
        self.forward_trans.resize(L, FeatRefs::default());
        crf1df_init_references(&mut self.attrs, &mut self.forward_trans, &self.features);
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

fn crf1df_init_references(attrs: &mut Vec<Vec<usize>>, forward_trans: &mut Vec<Vec<usize>>, features: &Vec<Feat>) {
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
        match f.ftype {
            FT_STATE => {
                attrs[f.src as usize].push(k);
            },
            FT_TRANS => {
                forward_trans[f.src as usize].push(k);
            },
            _ => panic!("unexpected feature type")
        }
    }
}

const FT_STATE: u32 = 0;
const FT_TRANS: u32 = 1;

fn crf1df_generate(ds: &Dataset, connect_all_attrs: bool, connect_all_edges: bool, feature_min_freq: f64) -> Vec<Feat> {
    let N = ds.len();
    let L = ds.num_labels();
    log::info!("start generate: N: {N}, L: {L}");
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
                set.add(Feat { ftype: FT_TRANS, src: prev as u32, dst: curr as u32, freq: seq.weight })
               }

               for attr in item {
                /* State feature: attribute #a -> state #(item->yid). */
                set.add(Feat { ftype: FT_STATE, src: attr.id as u32, dst: curr as u32, freq: seq.weight * attr.value });

                /* Generate state features connecting attributes with all
                   output labels. These features are not unobserved in the
                   training data (zero expexcations). */
                   if connect_all_attrs {
                    for i in 0..L {
                        set.add(Feat { ftype: FT_STATE, src: attr.id as u32, dst: i as u32, freq: 0.0 });
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
                set.add(Feat { ftype: FT_TRANS, src: i as u32, dst: j as u32, freq: 0.0 });
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
        Self { internal: Crf1de::default() }
    }
    
    /* LEVEL_NONE -> LEVEL_NONE. */
    pub(crate) fn objective_and_gradients_batch(&mut self, ds: &Dataset, w: *const f64, g: *mut f64) -> f64 {        
        let N = ds.len();
        let K = self.internal.num_features();

        /// Initialize the gradients with observation expectations.
        for i in 0..K {
            unsafe { *g.offset(i as isize) = -self.internal.features[i].freq };
        }

        /*
        Set the scores (weights) of transition features here because
        these are independent of input label sequences.
        */
        let ctx = self.internal.ctx.as_mut().expect("no ctx");
        ctx.reset(&[ResetOpt::RF_TRANS]);
        self.internal.transition_score(w);
        ctx.crf1dc_exp_transition();

        /// Compute model expectations.
        let mut logp = 0.0;
        let mut logl = 0.0;
        for seq in &ds.v {
            /* Set label sequences and state scores. */
            ctx.crf1dc_set_num_items(seq.len());
            ctx.reset(&[ResetOpt::RF_STATE]);
            self.internal.state_score(seq, w);
            ctx.crf1dc_exp_state();

            /* Compute forward/backward scores. */
            ctx.crf1dc_alpha_score();
            ctx.crf1dc_beta_score();
            ctx.crf1dc_marginals();

            /* Compute the probability of the input sequence on the model. */
            logp = ctx.crf1dc_score(seq.labels) - ctx.crf1dc_lognorm();
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
