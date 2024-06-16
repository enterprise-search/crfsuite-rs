use std::{convert::TryFrom, fs::File, iter::zip, path::Path, time::Duration};

use crfsuite::{Dataset, Model, Tagger};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn predict(tagger: &mut Tagger, dataset: &Dataset) {
    for seq in &dataset.seqs {
        let pred_labels = tagger.tag(&seq.items).expect("failed to tag");
        let mut n = 0;
        for (label, pred) in zip(&seq.labels, pred_labels) {
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
    let dataset = Dataset::try_from(f).expect("failed to read file");

    c.bench_function("predict", |b| {
        b.iter(|| predict(black_box(&mut tagger), black_box(&dataset)))
    });
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = predict_benchmark
}

criterion_main!(benchmarks);
