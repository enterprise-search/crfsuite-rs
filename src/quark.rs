use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub trait StringTable {
    fn to_str(&self, id: usize) -> Option<&str>;
    fn to_id(&self, s: &str) -> Option<usize>;
    fn len(&self) -> usize;
}

pub trait TextVectorizer {
    fn find_or_insert(&mut self, key: &str) -> usize;
}

#[derive(Debug, Default)]
pub struct Quark {
    v: Vec<String>,
    m: HashMap<String, usize>,
}

impl From<Vec<String>> for Quark {
    fn from(value: Vec<String>) -> Self {
        let m = value.iter().enumerate().map(|(i,s)| (s.to_string(),i)).collect();
        Self { v: value, m }
    }
}

impl StringTable for Quark {
    fn to_str(&self, id: usize) -> Option<&str> {
        self.v.get(id).map(|x| x.as_str())
    }

    fn to_id(&self, s: &str) -> Option<usize> {
        self.m.get(s).copied()
    }
    
    fn len(&self) -> usize {
        self.v.len()
    }
}

impl TextVectorizer for Quark {
    fn find_or_insert(&mut self, key: &str) -> usize {
        if self.m.contains_key(key) {
            return self.m[key];
        }
        let idx = self.v.len();
        self.m.insert(key.to_string(), idx);
        self.v.push(key.to_string());
        idx
    }
}

impl Quark {
    pub fn new(v: &[(String)]) -> Self {
        Self { v: v.to_vec(), m: v.iter().enumerate().map(|(i,s)| (s.to_string(), i)).collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_by_str() {
        let mut quark = Quark::default();
        for (s, id) in [("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("two", 2), ("one", 1), ("zero", 0), ("four", 4)] {
            assert_eq!(id, quark.find_or_insert(s), "{} != {}", s, id);
        }
    }

    #[test]
    fn find_by_id() {
        let mut quark = Quark::default();
        quark.find_or_insert("zero");
        quark.find_or_insert("one");
        assert_eq!(quark.to_str(0), Some("zero"));
        assert_eq!(quark.to_str(1), Some("one"));
        assert_eq!(quark.to_str(2), None);
    }
}