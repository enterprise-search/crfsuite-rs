use std::{convert::TryFrom, fs::File, time::{Duration, Instant}};

use crfsuite::{
    crf::{lbfgs, trainer::Crf1dEncoder},
    Dataset,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn train(ds: &Dataset) {
    let begin = Instant::now();
    let fpath = "ner_new.json";
    let holdout = usize::MAX;
    let encoder = Crf1dEncoder::default();
    lbfgs::train(encoder, &ds, fpath.into(), holdout);
    println!("took: {:?}", begin.elapsed());
}

fn train_benchmark(c: &mut Criterion) {
    let f = File::open("test.data").expect("failed to open file");
    let ds = Dataset::try_from(f).expect("failed to read file");
    assert_eq!(1915, ds.len(), "read count mismatch");
    c.bench_function("train", |b| b.iter(|| train(black_box(&ds))));
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = train_benchmark
}
criterion_main!(benchmarks);
