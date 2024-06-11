use crate::crf::tagger::Tagger;

use super::{
    context::{self},
    model::Crf1dModel,
};
use super::context::Crf1dContext;

#[derive(Debug)]
enum Level {
    LEVEL_NONE = 0,
    LEVEL_SET,
    LEVEL_ALPHABETA,
}
struct Crf1dTagger<'a> {
    model: &'a Crf1dModel,
    ctx: Crf1dContext,
    num_labels: usize,
    num_attrs: usize,
    level: Level,
}

impl<'a> Crf1dTagger<'a> {
    pub fn new(model: &'a Crf1dModel) -> Self {
        let L = model.num_labels();
        let mut ctx = Crf1dContext::new(&[context::Opt::CTXF_VITERBI, context::Opt::CTXF_MARGINALS], L, 0);
        ctx.reset(context::ResetOpt::RF_TRANS);
        {        
            /* Compute transition scores between two labels. */
            for i in 0..L {
                let edge = model.crf1dm_get_labelref(i);
                for r in 0..edge.num_features {
                    /* Transition feature from #i to #(f->dst). */
                    let fid = model.crf1dm_get_featureid(edge, r);
                    let f = model.crf1dm_get_feature(fid);
                    ctx.trans[ctx.num_labels * i + f.dst] = f.weight;
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
    fn set_instance(&self, instance: crate::crf::data::Instance) {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn viterbi(&self, labels: Vec<usize>) -> crate::crf::data::Float {
        todo!()
    }

    fn lognorm(&self) -> crate::crf::data::Float {
        todo!()
    }

    fn marginal_point(&self, l: usize, t: usize) -> crate::crf::data::Float {
        todo!()
    }

    fn marginal_path(&self, path: Vec<usize>, begin: usize, end: usize) -> crate::crf::data::Float {
        todo!()
    }

    fn score(&self, path: Vec<usize>) -> crate::crf::data::Float {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_tagger() {
        let buffer = [0; 10];
        let model = Crf1dModel::from_memory(buffer.to_vec());
        let tagger = Crf1dTagger::new(&model);
    }
}