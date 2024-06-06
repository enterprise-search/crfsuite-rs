use std::{fs::File, io::BufRead, path::PathBuf};

use clap::Parser;
use crfsuite::{Algorithm, Attribute, GraphicalModel, Item, Trainer};

#[derive(Debug, Parser)]
#[command(version)]
#[command(propagate_version = true)]
struct Argv {
    /// specify a graphical model
    #[arg(short = 't', long = "type", default_value_t = GraphicalModel::CRF1D)]
    graphical_model: GraphicalModel,
    #[arg(short, long, default_value_t = Algorithm::LBFGS)]
    algorithm: Algorithm,
    /// store the model to FILE
    #[arg(short, long, value_name = "FILE")]
    model: String,
    /// split the instances into N groups; this option is useful for holdout evaluation and cross validation
    #[arg(short = 'g', long, value_name = "N", default_value_t = 0)]
    split: u32,
    /// use the M-th data for holdout evaluation and the rest for training
    #[arg(short = 'e', long, value_name = "M", default_value_t = -1)]
    holdout: i32,
    /// set the algorithm-specific parameter NAME to VALUE
    #[arg(short, long, value_name = "NAME=VALUE")]
    parameters: Vec<String>,
    /// repeat holdout evaluations for #i in {1, ..., N} groups (N-fold cross validation)
    #[arg(short = 'x')]
    cross_validate: bool,
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    #[arg(required = true)]
    datasets: Vec<PathBuf>,
}

fn main() {
    env_logger::init();

    let argv = Argv::parse();
    log::info!("argv: {:?}", argv);
    let mut trainer = Trainer::new(argv.verbose > 0);
    trainer
        .select(argv.algorithm, argv.graphical_model)
        .expect("failed to select algorithm and graphical model");
    argv.parameters.iter().for_each(|s| {
        if let Some((name, value)) = s.split_once("=") {
            trainer
                .set(name, value)
                .expect("failed to set parameter: {s}");
        }
    });
    argv.datasets.iter().for_each(|fpath| {
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
    });
    trainer.train(&argv.model, -1).expect("failed to train");
    log::info!("write model to {}", argv.model);
}
