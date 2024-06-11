use std::{collections::HashMap, ffi::{c_char, CString}, ptr::null};

#[derive(Debug, Default)]
pub struct Quark {
    v: Vec<String>,
    m: HashMap<String, usize>,
}

impl Quark {
    pub fn new(v: &[(String)]) -> Self {
        Self { v: v.to_vec(), m: v.iter().enumerate().map(|(i,s)| (s.to_string(), i)).collect() }
    }
    pub fn find_by_id(&self, id: usize) -> Option<String> {
        if self.v.len() > id {
            return Some(self.v[id].to_string());
        }
        return None
    }

    pub fn find(&self, key: &str) -> Option<usize> {
        self.m.get(key).copied()
    }

    pub fn find_or_insert(&mut self, key: &str) -> usize {
        if self.m.contains_key(key) {
            return self.m[key];
        }
        let idx = self.v.len();
        self.m.insert(key.to_string(), idx);
        self.v.push(key.to_string());
        idx
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
        assert_eq!(quark.find_by_id(0), Some("zero".to_string()));
        assert_eq!(quark.find_by_id(1), Some("one".into()));
        assert_eq!(quark.find_by_id(2), None);
    }
}