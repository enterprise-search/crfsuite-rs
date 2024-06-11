use std::usize;

use crate::quark::Quark;

pub type Float = f64;

pub struct Attr {
    pub aid: usize,
    pub value: Float,
}

impl Attr {
    pub fn new(aid: usize, value: Float) -> Self {
        Self { aid, value }
    }
}

pub type Item = Vec<Attr>;

pub struct Instance {
    pub items: Vec<Item>,
    labels: Vec<usize>,
    weight: Float,
    group: usize,
}

impl Instance {
    pub fn append(&mut self, item: Item, label: usize) {
        self.items.push(item);
        self.labels.push(label);
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.labels.clear();
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.labels.is_empty()
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self { items: Default::default(), labels: Default::default(), weight: 1.0, group: Default::default() }
    }
}

pub struct Data {
    instances: Vec<Instance>,
    attrs: Quark,
    labels: Quark,
}

impl Data {
    pub fn append(&mut self, instance: Instance) {
        if !instance.is_empty() {
            self.instances.push(instance);
        }
    }

    pub fn max_length(&self) -> usize {
        self.instances.iter().map(|x| x.len()).max().unwrap_or_default()
    }

    pub fn total_items(&self) -> usize {
        self.instances.iter().map(|x| x.len()).sum()
    }
}

pub struct Dataset {
    data: Data,
    perm: Vec<usize>,
}