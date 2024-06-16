use std::{fs::File, time::Instant};

use crfsuite::{
    crf::{
        lbfgs,
        trainer::TagEncoder,
    },
    Dataset,
};

#[test]
fn train() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();
    let begin = Instant::now();
    let input = File::open("test.data").expect("failed to open file");
    let mut ds = Dataset::default();
    ds.read_file(input).expect("failed to read input file");
    assert_eq!(1915, ds.len(), "read count mismatch");
    let fpath = "ner_new.json";
    let holdout = usize::MAX;
    let encoder = TagEncoder::new();
    lbfgs::train(encoder, &ds, fpath.into(), holdout);
    println!("took: {:?}", begin.elapsed());
}

#[test]
fn math() {
    let f = 12.34_f64;
    println!("{}", f.exp());
}