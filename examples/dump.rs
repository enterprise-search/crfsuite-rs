use std::{io::stdin, os::fd::{AsFd, AsRawFd}};

use clap::Parser;
use crfsuite::Model;

/// output the model stored in the file (MODEL) in a plain-text format
#[derive(Debug, Parser)]
struct Argv {
    #[arg(short, long, required = true, value_name = "MODEL")]
    model: String,
}

fn main() {
    let argv = Argv::parse();
    let model = Model::from_file(&argv.model).expect("failed to load model");
    model.dump(stdin().as_fd().as_raw_fd()).expect("failed to dump model");
}