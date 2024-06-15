use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf};

use clap::Parser;
use crfsuite::{Attribute, Item, Model, Evaluation};

/// Assign suitable labels to the instances in the data set given by a file (FILE)
/// If the argument DATA is omitted or '-', this utility reads a data from STDIN
/// Evaluate the performance of the model on labeled instances (with -t option)
#[derive(Debug, Parser)]
struct Argv {
    /// read a model from a file (MODEL)
    #[arg(short, long, required = true, value_name = "MODEL")]
    model: String,
    /// report the performance of the model on the data
    #[arg(short = 't', long = "test")]
    evaluate: bool,
    /// output the reference labels in the input data
    #[arg(short, long)]
    reference: bool,
    /// output the probability of the label sequences
    #[arg(short, long)]
    probability: bool,
    /// output the marginal probabilitiy of items for their predicted label
    #[arg(short = 'i', long)]
    marginal: bool,
    /// output the marginal probabilities of items for all labels
    #[arg(short = 'l', long)]
    marginal_all: bool,
    /// suppress tagging results (useful for test mode)\n");
    #[arg(short, long)]
    quiet: bool,
    /// assign suitable labels to the instances in the data set given by a file (FILE)
    #[arg(value_name="<FILE>")]
    datasets: Vec<PathBuf>,
}

fn main() {
    env_logger::init();
    let argv = Argv::parse();
    log::info!("{:?}", argv);
    let model = Model::from_file(&argv.model).expect("failed to load model");
    let mut tagger = model.tagger().expect("failed to get tagger from model");
    let mut evaluation = Evaluation::default();
    evaluation.num_labels = tagger.labels().expect("failed to get labels").len();
    for fpath in argv.datasets {
        let f = File::open(fpath).expect("failed to open the stream for the input data");
        let mut items = Vec::new();
        let mut labels = Vec::new();
        for line in BufReader::new(f).lines() {
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
                    let prediction = tagger.tag(&items).expect("failed to tag");
                    if argv.evaluate {
                        evaluation.accumulate(&labels, &prediction.iter().map(AsRef::as_ref).collect::<Vec<_>>().as_slice());
                    }
                    if !argv.quiet {
                        // output_result
                    }
                    for (label, pred) in labels.iter().zip(prediction) {
                        // println!("{:?}\t{:?}", label, pred);
                    }
                    items.clear();
                    labels.clear();
                }
            }
        }
    }
    if argv.evaluate {
        //double sec = (clk1 - clk0) / (double)CLOCKS_PER_SEC;
        evaluation.evaluate();
        println!("{}", evaluation);
        // fprintf(fpo, "Elapsed time: %f [sec] (%.1f [instance/sec])\n", sec, N / sec);
    }
}