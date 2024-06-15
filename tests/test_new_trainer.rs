use std::fs::File;

use crfsuite::{
    crf::{
        lbfgs::{self, Lbfgs},
        trainer::{Crf1dTrainer, TagEncoder},
    },
    Dataset,
};

#[test]
fn train() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();
    let input = File::open("test.data").expect("failed to open file");
    let mut ds = Dataset::default();
    ds.read_file(input).expect("failed to read input file");
    assert_eq!(1915, ds.len(), "read count mismatch");
    let fpath = "ner_new.json";
    let holdout = usize::MAX;
    let encoder = TagEncoder::new();
    lbfgs::train(encoder, &ds, fpath.into(), holdout);
}

#[test]
fn math() {
    let f = 12.34_f64;
    println!("{}", f.exp());
}