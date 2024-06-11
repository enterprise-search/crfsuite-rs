use super::data::{Float, Instance};

pub trait Tagger {
    fn set_instance(&self, instance: Instance);
    fn len(&self) -> usize;
    fn viterbi(&self, labels: Vec<usize>) -> Float;
    fn lognorm(&self) -> Float;
    fn marginal_point(&self, l: usize, t: usize) -> Float;
    fn  marginal_path(&self, path: Vec<usize>, begin: usize, end: usize)  -> Float;   
    fn  score(&self, path: Vec<usize>) -> Float;
}