use std::convert::TryFrom;
use std::io::Read;
use std::str::Lines;
use std::usize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::quark::{Quark, TextVectorizer};

pub type Item = Vec<Attr>;

#[repr(C)]
#[derive(Debug)]
pub struct Attr {
    pub id: u32,
    pub value: f64,
}

#[derive(Debug)]
pub struct Sequence {
    pub items: Vec<Item>,
    pub labels: Vec<usize>,
    pub weight: f64,
    group: usize,
}

impl Default for Sequence {
    fn default() -> Self {
        Self {
            items: Default::default(),
            labels: Default::default(),
            weight: 1.0,
            group: Default::default(),
        }
    }
}

impl Sequence {
    pub fn push(&mut self, item: Item, label: usize) {
        self.items.push(item);
        self.labels.push(label);
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.labels.clear();
    }
}

#[derive(Debug, Default)]
pub struct Dataset {
    pub seqs: Vec<Sequence>,
    m_labels: Quark,
    m_attrs: Quark,
}

impl From<Lines<'_>> for Dataset {
    fn from(value: Lines) -> Self {
        let mut m_labels = Quark::default();
        let mut m_attrs = Quark::default();
        let mut v = Vec::new();
        let mut seq = Sequence::default();
        for line in value {
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Item = attrs
                        .split('\t')
                        .map(|s| Attr {
                            id: m_attrs.find_or_insert(s) as u32,
                            value: 1.0,
                        })
                        .collect();
                    seq.push(item, m_labels.find_or_insert(&label));
                } else {
                    log::warn!("invalid line: {line}");
                }
            } else {
                if !(seq.is_empty()) {
                    v.push(seq);
                    seq = Sequence::default();
                }
            }
        }
        Self { seqs: v, m_labels, m_attrs }
    }
}

impl TryFrom<File> for Dataset {
    type Error = std::io::Error;
    
    fn try_from(value: File) -> Result<Self, Self::Error> {
        let mut seq = Sequence::default();
        let mut m_labels = Quark::default();
        let mut m_attrs = Quark::default();
        let mut v = Vec::new();
        for line in BufReader::new(value).lines() {
            let line = line?;
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Item = attrs
                        .split('\t')
                        .map(|s| Attr {
                            id: m_attrs.find_or_insert(s) as u32,
                            value: 1.0,
                        })
                        .collect();
                    seq.push(item, m_labels.find_or_insert(&label));
                } else {
                    log::warn!("invalid line: {line}");
                }
            } else {
                if !(seq.is_empty()) {
                    v.push(seq);
                    seq = Sequence::default();
                }
            }
        }
        Ok(Self { seqs: v, m_labels, m_attrs })
    }
}

impl Dataset {
    pub fn read_file<Reader: Read>(&mut self, file: Reader) -> Result<(), std::io::Error> {
        let mut seq = Sequence::default();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Item = attrs
                        .split('\t')
                        .map(|s| Attr {
                            id: self.m_attrs.find_or_insert(s) as u32,
                            value: 1.0,
                        })
                        .collect();
                    seq.push(item, self.m_labels.find_or_insert(&label));
                } else {
                    log::warn!("invalid line: {line}");
                }
            } else {
                if !(seq.is_empty()) {
                    self.seqs.push(seq);
                    seq = Sequence::default();
                }
            }
        }
        Ok(())
    }

    pub fn max_seq_length(&self) -> usize {
        self.seqs.iter().map(|x| x.len()).max().unwrap_or_default()
    }

    pub fn total_items(&self) -> usize {
        self.seqs.iter().map(|x| x.len()).sum()
    }

    pub fn len(&self) -> usize {
        self.seqs.len()
    }

    pub(crate) fn get(&self, i: usize) -> &Sequence {
        &self.seqs[i]
    }

    pub fn num_labels(&self) -> usize {
        self.m_labels.len()
    }

    pub fn num_attrs(&self) -> usize {
        self.m_attrs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_from_lines() {
        let s = "P\thello\tworld
        Q\thi\ttherer\n\n";
        let ds = Dataset::from(s.lines());
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.num_attrs(), 4);
        assert_eq!(ds.num_labels(), 2);
    }
}