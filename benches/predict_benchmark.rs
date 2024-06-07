use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crfsuite::{Attribute, Item, Model};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn predict(fpath: &Path) {
    let model = "ner";
    let model = Model::from_file(model).expect("failed to load model");
    let mut tagger = model.tagger().expect("failed to get tagger from model");
    let f = File::open(fpath).expect("failed to open the stream for the input data");
    let mut items = Vec::new();
    let mut labels = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.expect("failed to read line");
        if !line.is_empty() {
            if let Some((label, attrs)) = line.split_once('\t') {
                let item: Item = attrs.split('\t').map(Attribute::from).collect();
                items.push(item);
                labels.push(label.to_string());
            } else {
                log::warn!("invalid line: {line}");
            }
        } else {
            if !(labels.is_empty() || items.is_empty()) {
                let pred_labels = tagger.tag(&items).expect("failed to tag");
                for (label, pred) in labels.iter().zip(pred_labels) {
                    // println!("{:?}\t{:?}", label, pred);
                }
                items.clear();
                labels.clear();
            }
        }
    }
    if !(labels.is_empty() || items.is_empty()) {
        let pred_labels = tagger.tag(&items).expect("failed to tag");
        for (label, pred) in labels.iter().zip(pred_labels) {
            // println!("{:?}\t{:?}", label, pred);
        }
        items.clear();
        labels.clear();
    }
}

fn predict_benchmark(c: &mut Criterion) {
    let fpath = Path::new("test.data");
    c.bench_function("test", |b| b.iter(|| predict(black_box(&fpath))));
}

criterion_group!(benchmarks, predict_benchmark);
criterion_main!(benchmarks);
