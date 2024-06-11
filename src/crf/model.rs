use std::path::PathBuf;

use crate::{quark::Quark, Tagger};

pub trait Model {
    fn get_tagger(&self) -> Tagger;
    fn get_labels(&self) -> Quark;
    fn get_attrs(&self) -> Quark;
    fn dump(&self, path: PathBuf);
}