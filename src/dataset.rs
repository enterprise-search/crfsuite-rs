use std::{fs::File, io::{BufRead, BufReader}};
use std::usize;

pub type Item = Vec<Attr>;

#[repr(C)]
#[derive(Debug)]
pub struct Attr {
    pub id: i32,
    pub value: f64,
}

#[derive(Debug)]
pub struct Sequence {
    pub items: Vec<Item>,
    pub labels: Vec<usize>,
    weight: f64,
    group: usize,
}

impl Default for Sequence {
    fn default() -> Self {
        Self { items: Default::default(), labels: Default::default(), weight: 1.0, group: Default::default() }
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
    pub v: Vec<Sequence>,
    pub n_labels: usize,
    pub n_attrs: usize,
}

impl Dataset {
    pub fn read_file<AttrToId: FnMut(&str) -> usize, LabelToId: FnMut(&str) -> usize>(&mut self, file: File, mut attr_to_id: AttrToId, mut label_to_id: LabelToId) -> Result<(), std::io::Error> {
        let mut seq = Sequence::default();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Item = attrs.split('\t').map(|s| Attr { id: attr_to_id(s) as i32, value: 1.0 }).collect();
                    seq.push(item, label_to_id(&label));
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
        Ok(())
    }

    pub fn max_length(&self) -> usize {
        self.v.iter().map(|x| x.len()).max().unwrap_or_default()
    }

    pub fn total_items(&self) -> usize {
        self.v.iter().map(|x| x.len()).sum()
    }
}
