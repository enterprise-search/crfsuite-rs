use crate::Sequence;

pub trait Tagger {
    fn set_seq(&mut self, instance: &Sequence);
    fn len(&self) -> usize;
    fn viterbi(&mut self, labels: &mut Vec<usize>) -> f64;
    fn lognorm(&self) -> f64;
    fn marginal_point(&self, l: usize, t: usize) -> f64;
    fn marginal_path(&self, path: Vec<usize>, begin: usize, end: usize)  -> f64;   
    fn score(&self, path: Vec<usize>) -> f64;
}