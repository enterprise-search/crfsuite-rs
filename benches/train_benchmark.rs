use std::{fs::File, io::BufRead, path::Path};

use crfsuite::{Attribute, Item, Trainer};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn train(fpath: &Path) {
    let mut trainer = Trainer::new(false);
    trainer
        .select(crfsuite::Algorithm::LBFGS, crfsuite::GraphicalModel::CRF1D)
        .expect("failed to select algorithm and graphical model");
    let parameters = vec![
        "c1=0.1",
        "c2=0.1",
        "max_iterations=100",
        "feature.possible_transitions=1",
    ];
    parameters.iter().for_each(|s| {
        if let Some((name, value)) = s.split_once("=") {
            trainer
                .set(name, value)
                .expect("failed to set parameter: {s}");
        }
    });

    log::info!("reading dataset from: {:?}", fpath);
    let f = File::open(fpath).expect("failed to open: {fpath}");
    let mut count = 0;
    let mut items = Vec::new();
    let mut labels = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
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
                trainer
                    .append(&items, &labels, 1)
                    .expect("failed to append item");
                count += items.len();
                items.clear();
                labels.clear();
            }
        }
    }
    if !(labels.is_empty() || items.is_empty()) {
        trainer
            .append(&items, &labels, 1)
            .expect("failed to append item");
        count += items.len();
        items.clear();
        labels.clear();
    }
    log::info!("read {} items", count);
    let model = "/tmp/bench";
    trainer.train(model, -1).expect("failed to train");
}

fn train_benchmark(c: &mut Criterion) {
    let fpath = Path::new("test.data");
    c.bench_function("train", |b| b.iter(|| train(black_box(&fpath))));
}

criterion_group!(benchmarks, train_benchmark);
criterion_main!(benchmarks);
