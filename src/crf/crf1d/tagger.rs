use crate::crf::tagger::Tagger;
use crate::Sequence;

use super::context::Crf1dContext;
use super::{
    context::{self, ResetOpt},
    model::Crf1dModel,
};

#[derive(Debug)]
enum Level {
    LEVEL_NONE = 0,
    LEVEL_SET,
    LEVEL_ALPHABETA,
}
pub struct Crf1dTagger<'a> {
    model: &'a Crf1dModel,
    ctx: Crf1dContext,
    num_labels: usize,
    num_attrs: usize,
    level: Level,
}

impl<'a> Crf1dTagger<'a> {
    pub fn new(model: &'a Crf1dModel) -> Self {
        let L = model.num_labels();
        let mut ctx = Crf1dContext::new(
            &[context::Opt::CTXF_VITERBI, context::Opt::CTXF_MARGINALS],
            L,
            0,
        );
        ctx.reset(&[context::ResetOpt::RF_TRANS]);
        {
            /* Compute transition scores between two labels. */
            for i in 0..L {
                let edge = model.crf1dm_get_labelref(i);
                for r in 0..edge.len() {
                    /* Transition feature from #i to #(f->dst). */
                    let fid = model.crf1dm_get_featureid(edge, r);
                    let f = model.crf1dm_get_feature(fid);
                    ctx.trans[ctx.num_labels * i + f.dst as usize] = f.weight;
                }
            }
        }
        ctx.crf1dc_exp_transition();
        let mut this = Self {
            model: model,
            ctx: ctx,
            num_labels: L,
            num_attrs: model.num_attrs(),
            level: Level::LEVEL_NONE,
        };
        this.level = Level::LEVEL_NONE;
        this
    }

    pub fn crf1dt_transition_score(&mut self) {
        todo!()
    }
}

impl<'a> Tagger for Crf1dTagger<'a> {
    fn set_seq(&mut self, instance: &Sequence) {
        let T = instance.len();
        self.ctx.crf1dc_set_num_items(T);
        self.ctx.reset(&[ResetOpt::RF_STATE]);

        /* Loop over the items in the sequence. */
        for (i, item) in instance.items.iter().enumerate() {
            /* Loop over the contents (attributes) attached to the item. */
            for attr in item {
                /* A scale usually represents the atrribute frequency in the item. */
                let value = attr.value;
                let attr_ref = self.model.crf1dm_get_attrref(attr.id as usize);
                /* Loop over the state features associated with the attribute. */
                for k in 0..attr_ref.len() {
                    /* The state feature #(attr->fids[r]), which is represented by
                    the attribute #a, outputs the label #(f->dst). */
                    let fid = self.model.crf1dm_get_featureid(attr_ref, k);
                    let f = self.model.crf1dm_get_feature(fid);
                    self.ctx.state[self.ctx.num_labels * i + f.dst as usize] += f.weight * value;
                }
            }
        }
        self.level = Level::LEVEL_SET;
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn viterbi(&mut self, labels: &mut Vec<usize>) -> f64 {
        self.ctx.viterbi(labels)
    }

    fn lognorm(&self) -> f64 {
        todo!()
    }

    fn marginal_point(&self, l: usize, t: usize) -> f64 {
        todo!()
    }

    fn marginal_path(&self, path: Vec<usize>, begin: usize, end: usize) -> f64 {
        todo!()
    }

    fn score(&self, path: Vec<usize>) -> f64 {
        todo!()
    }
}
