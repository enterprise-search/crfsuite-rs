use std::{fs::File, io::{BufRead, BufReader}};

use crate::{Attribute, Item};

#[derive(Debug, Default)]
pub struct Sequence {
    pub items: Vec<Item>,
    pub labels: Vec<String>,
}

impl Sequence {
    pub fn push(&mut self, item: Item, label: String) {
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
pub struct Dataset(pub Vec<Sequence>);

impl From<File> for Dataset {
    fn from(value: File) -> Self {
        let mut dataset = Dataset::default();
        let mut sequence = Sequence::default();
        for line in BufReader::new(value).lines() {
            let line = line.expect("failed to read line");
            if !line.is_empty() {
                if let Some((label, attrs)) = line.split_once('\t') {
                    let item: Item = attrs.split('\t').map(Attribute::from).collect();
                    sequence.push(item, label.to_string());
                } else {
                    log::warn!("invalid line: {line}");
                }
            } else {
                if !(sequence.is_empty()) {
                    dataset.0.push(sequence);
                    sequence = Sequence::default();
                }
            }
        }
        dataset
    }
}