use std::{
    fs::File,
    io::{BufRead, BufReader},
    iter::zip, time::Duration,
};

use crfsuite::{
    crf::{data::Instance, tagger::Tagger},
    Performance,
};

use crfsuite::crf::{
    crf1d::model::Crf1dModel,
    data::{Attr, Item},
    model::Model,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn predict(tagger: &mut dyn Tagger, dataset: &Vec<Instance>) {
    for sequence in dataset {
        tagger.set_instance(sequence);
        let mut prediction = vec![0; sequence.len()];
        let score = tagger.viterbi(&mut prediction);

        let mut n = 0;
        for (label, pred) in zip(&sequence.labels, prediction) {
            if *label != pred {
                n += 1;
            }
        }
        // if n > 0 {
        //     eprintln!("not match: {}/{}", n, sequence.len());
        // }
    }
}

fn pred_benchmark(c: &mut Criterion) {
    let model_path = "ner";
    let model = Crf1dModel::from_path(model_path.into()); //.expect("failed to load model");
    let mut tagger = model.get_tagger(); //.expect("failed to get tagger from model");

    let mut performance = Performance::default();
    performance.num_labels = model.num_labels();

    let input_path = "test.data";
    let f = File::open(input_path).expect("failed to open the stream for the input data");
    let mut instance = Instance::default();
    let m_labels = model.get_labels();
    let m_attrs = model.get_attrs();
    let mut labels = Vec::new();
    let mut dataset = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.expect("failed to read line");
        if !line.is_empty() {
            if let Some((label, attrs)) = line.split_once('\t') {
                let item: Item = attrs
                    .split('\t')
                    .map(|s| m_attrs.find(s))
                    .flatten()
                    .map(|i| Attr::new(i, 1.0))
                    .collect();
                instance.append(item, m_labels.find(label).unwrap_or(labels.len()));
                labels.push(label.to_string());
            } else {
                log::warn!("invalid line: {line}");
            }
        } else {
            if !(instance.is_empty()) {
                dataset.push(instance);
                instance = Instance::default();
            }
        }
    }
    let est = performance.evaluate();
    println!("perf: {}, est: {:?}", performance, est);
    c.bench_function("pred", |b| {
        b.iter(|| predict(black_box(&mut tagger), black_box(&dataset)))
    });
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = pred_benchmark
}
criterion_main!(benchmarks);
