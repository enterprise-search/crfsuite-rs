use std::path::PathBuf;

use crate::{quark::Quark};

use super::tagger::Tagger;

pub trait Model {
    fn get_tagger(&self) -> impl Tagger;
    fn get_labels(&self) -> &Quark;
    fn get_attrs(&self) -> &Quark;
    fn dump(&self, path: PathBuf);
}