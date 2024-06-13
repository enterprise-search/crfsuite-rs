use std::{
    fs::File, io::{BufRead, BufReader}, iter::zip, path::Path
};

use crfsuite::{Attribute, Item, Model, Tagger};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[derive(Debug, Default)]
struct Sequence {
    items: Vec<Item>,
    labels: Vec<String>,
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
struct Dataset(Vec<Sequence>);


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

fn predict(tagger: &mut Tagger, dataset: &Dataset) {
    for sequence in &dataset.0 {
        let pred_labels = tagger.tag(&sequence.items).expect("failed to tag");
        let mut n = 0;
        for (label, pred) in zip(&sequence.labels, pred_labels) {
            // println!("{:?}\t{:?}", label, pred);
            if *label != pred {
                n += 1;
            }
        }
        // eprintln!("not match: {}/{}", n, sequence.len());
    }    
}

fn predict_benchmark(c: &mut Criterion) {
    let model = "ner";
    let model = Model::from_file(model).expect("failed to load model");
    let mut tagger = model.tagger().expect("failed to get tagger from model");

    let fpath = Path::new("test.data");
    let f = File::open(fpath).expect("failed to open the stream for the input data");    
    let dataset = Dataset::from(f);
    
    c.bench_function("test", |b| b.iter(|| predict(black_box(&mut tagger), black_box(&dataset))));
}

criterion_group!(benchmarks, predict_benchmark);
criterion_main!(benchmarks);
