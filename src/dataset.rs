use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf};


use crate::{quark::{Quark, TextVectorizer}, Attribute, Error, Item};

#[repr(C)]
#[derive(Debug)]
pub struct Attr {
    id: i32,
    value: f64,
}

pub type Token = Vec<Attr>;

#[derive(Debug, Default)]
pub struct Sequence {
    pub items: Vec<Token>,
    pub labels: Vec<usize>,
}

impl Sequence {
    pub fn push(&mut self, item: Token, label: usize) {
        self.items.push(item);
        self.labels.push(label);
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }    
}

#[derive(Debug, Default)]
pub struct Dataset {
    pub v: Vec<Sequence>,
    pub n_labels: usize,
    pub n_attrs: usize,
}

impl Dataset {
    pub fn read_file(&mut self, file: File, mut attrv: impl TextVectorizer, mut labelv: impl TextVectorizer) -> Result<(), std::io::Error> {
        let mut seq = Sequence::default();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Token = attrs.split('\t').map(|s| Attr { id: attrv.find_or_insert(s) as i32, value: 1.0 }).collect();
                    seq.push(item, labelv.find_or_insert(&label));
                } else {
                    log::warn!("invalid line: {line}");
                }
            } else {
                if !(seq.is_empty()) {
                    self.v.push(seq);
                    seq = Sequence::default();
                }
            }
        }
        self.n_attrs = attrv.len();
        self.n_labels = labelv.len();
        Ok(())
    }
}
