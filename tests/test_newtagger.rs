use std::{
    fs::File,
    io::{BufRead, BufReader},
};
extern crate crfsuite;
use crfsuite::{
    crf::crf1d::model::Crf1dModel,
    dataset::{Attr, Item},
    quark::StringTable,
    Evaluation, Sequence,
};

use crate::crfsuite::crf::{model::Model, tagger::Tagger};

#[test]
fn tag() {
    let model_path = "ner.json";
    // let model_path = "../crfsuite/ner";
    let input_path = "test.data";
    let model = Crf1dModel::from_json(model_path.into()); //.expect("failed to load model");
    let mut tagger = model.get_tagger(); //.expect("failed to get tagger from model");
    let mut evaluation = Evaluation::default();
    evaluation.num_labels = model.num_labels();

    let mut seq = Sequence::default();
    let m_labels = model.get_labels();
    let m_attrs = model.get_attrs();
    let f = File::open(input_path).expect("failed to open the stream for the input data");
    let mut labels = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.expect("failed to read line");
        if !line.is_empty() {
            if let Some((label, attrs)) = line.split_once('\t') {
                let item: Item = attrs
                    .split('\t')
                    .map(|s| m_attrs.to_id(s))
                    .flatten()
                    .map(|i| Attr {
                        id: i as i32,
                        value: 1.0,
                    })
                    .collect();
                seq.push(item, m_labels.to_id(label).unwrap_or(labels.len()));
                labels.push(label.to_string());
            } else {
                log::warn!("invalid line: {line}");
            }
        } else {
            if !(seq.is_empty()) {
                tagger.set_seq(&seq);
                let mut prediction = Vec::new();
                prediction.resize(seq.len(), 0);
                let score = tagger.viterbi(&mut prediction); //.expect("failed to tag");
                                                             // println!("score = {score}, {:?}, \nrefs: {:?}, \npred: {:?}", labels, seq.labels, prediction);
                                                             // println!("output: {:?}", prediction);
                let prediction = prediction
                    .iter()
                    .map(|i| m_labels.to_str(*i).unwrap_or("N/A"))
                    .collect::<Vec<_>>();
                evaluation.accumulate(&labels, &prediction);
                let quiet = false;
                if !quiet {
                    // output_result
                }
                seq.clear();
                labels.clear();
            }
        }
    }
    let est = evaluation.evaluate();
    println!("perf: {}, est: {:?}", evaluation, est);
    assert!(
        (est.precision - 0.998457).abs() < 0.00001,
        "{}",
        est.precision
    );
    assert!((est.recall - 0.997050).abs() < 0.00001, "{}", est.recall);
}
