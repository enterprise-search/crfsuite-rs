use std::{fs::File, io::BufRead, path::PathBuf};

use clap::Parser;
use crfsuite::{Algorithm, Attribute, GraphicalModel, Item, Trainer};

#[derive(Debug, Parser)]
#[command(version)]
#[command(propagate_version = true)]
struct Argv {
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    #[arg(short = 't', long = "type", default_value_t = GraphicalModel::CRF1D)]
    graphical_model: GraphicalModel,
    #[arg(short, long, default_value_t = Algorithm::LBFGS)]
    algorithm: Algorithm,
    #[arg(short, long)]
    model: String,
    #[arg(short)]
    parameters: Vec<String>,
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
        let f = File::open(fpath).expect("failed to open: {fpath}");
        for line in std::io::BufReader::new(f)
            .lines()
            .flatten()
            .filter(|s| !s.is_empty())
        {
            if let Some((label, attrs)) = line.split_once('\t') {
                let item: Item = attrs.split('\t').map(Attribute::from).collect();
                trainer
                    .append(&[item], &[label], 1)
                    .expect("failed to append item");
            } else {
                log::warn!("invalid line: {line}");
            }
        }
    });
    trainer.train(&argv.model, -1).expect("failed to train");
    log::info!("write model to {}", argv.model);
}
