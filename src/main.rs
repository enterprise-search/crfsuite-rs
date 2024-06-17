use std::{convert::TryFrom, fs::File, time::{Duration, Instant}};

use crfsuite::{
    crf::{lbfgs, trainer::Crf1dEncoder},
    Dataset,
};

fn train(ds: &Dataset) {
    let begin = Instant::now();
    let fpath = "ner_new.json";
    let holdout = usize::MAX;
    let encoder = Crf1dEncoder::default();
    lbfgs::train(encoder, &ds, fpath.into(), holdout);
    println!("took: {:?}", begin.elapsed());
}

fn main() {
    println!("begin");
    let f = File::open("test.data").expect("failed to open file");
    let ds = Dataset::try_from(f).expect("failed to read file");
    for i in 0..100 {
        let ts = Instant::now();
        train(&ds);
        println!("iteration {i} took: {:?}", ts.elapsed());
    }
    assert_eq!(1915, ds.len(), "read count mismatch");
    println!("OK\n");
}