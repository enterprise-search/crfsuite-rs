use std::path::Path;

use crate::quark::Quark;

use super::data::Data;


pub trait Trainer {
    fn train(&self, data: Data, fpath: Path, holdout: usize);
}