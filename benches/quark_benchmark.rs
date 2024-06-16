use std::{convert::TryFrom, fs::File, path::Path, time::Duration};

use crfsuite::Dataset;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn quark(fpath: &Path) {
    let f = File::open(fpath).expect("failed to open file");
    let ds = Dataset::try_from(f).expect("failed to read file");
    assert_eq!(ds.len(), 1915);
}

fn quark_benchmark(c: &mut Criterion) {
    let fpath = Path::new("test.data");
    c.bench_function("quark", |b| b.iter(|| quark(black_box(&fpath))));
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = quark_benchmark
}

criterion_main!(benchmarks);
