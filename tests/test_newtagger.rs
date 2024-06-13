use std::{
    fs::File,
    io::{BufRead, BufReader},
};
extern crate crfsuite;
use crfsuite::{
    crf::crf1d::{model::Crf1dModel, tagger::Crf1dTagger},
    Dataset, Performance,
};

use crate::crfsuite::crf::{
    data::{Attr, Instance, Item},
    model::Model,
    tagger::Tagger,
};

#[test]
fn init_tagger() {
    let buffer = [0; 10];
    let model = Crf1dModel::from_memory(buffer.to_vec());
    let tagger = Crf1dTagger::new(&model);
}

#[test]
fn tag() {
    let model_path = "ner";
    let input_path = "test.data";
    let model = Crf1dModel::from_path(model_path.into()); //.expect("failed to load model");
    let mut tagger = model.get_tagger(); //.expect("failed to get tagger from model");
    let mut performance = Performance::default();
    performance.num_labels = model.num_labels();
    let f = File::open(input_path).expect("failed to open the stream for the input data");
    let mut instance = Instance::default();
    let m_labels = model.get_labels();
    let m_attrs = model.get_attrs();
    let mut labels = Vec::new();
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
                tagger.set_instance(&instance);
                let mut prediction = Vec::new();
                prediction.resize(instance.len(), 0);
                let score = tagger.viterbi(&mut prediction); //.expect("failed to tag");
                                                             // println!("score = {score}, {:?}, \nrefs: {:?}, \npred: {:?}", labels, instance.labels, prediction);
                                                             // println!("output: {:?}", prediction);
                let prediction = prediction
                    .iter()
                    .map(|i| m_labels.find_by_id(*i).unwrap_or("N/A".into()))
                    .collect::<Vec<_>>();
                performance.accumulate(&labels, &prediction);
                let quiet = false;
                if !quiet {
                    // output_result
                }
                instance.clear();
                labels.clear();
            }
        }
    }
    //double sec = (clk1 - clk0) / (double)CLOCKS_PER_SEC;
    let est = performance.evaluate();
    println!("perf: {}, est: {:?}", performance, est);
    assert!(
        (est.precision - 0.998457).abs() < 0.00001,
        "{}",
        est.precision
    );
    assert!((est.recall - 0.997050).abs() < 0.00001, "{}", est.recall);
    // fprintf(fpo, "Elapsed time: %f [sec] (%.1f [instance/sec])\n", sec, N / sec);
}
