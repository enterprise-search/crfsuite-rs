use std::{ffi::CStr, mem::MaybeUninit, path::PathBuf, slice, time::Instant};

use clap::Parser;
use libc::c_void;
use liblbfgs_sys::{
    lbfgs, lbfgs_free, lbfgs_malloc, lbfgs_parameter_init, lbfgs_parameter_t, lbfgs_strerror,
};

use crate::Dataset;

use super::trainer::TagEncoder;

#[repr(C)]
#[derive(Debug)]
struct Ctx {
    encoder: TagEncoder,
    c2: f64,
    trainset: *const Dataset,
    best_w: Vec<f64>,
    timestamp: Instant,
}

impl Ctx {
    pub fn new(encoder: TagEncoder, trainset: *const Dataset) -> Self {
        Self {
            c2: 0.1,
            trainset,
            best_w: vec![0.0; encoder.num_features()],
            encoder: encoder,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Debug)]
pub struct Lbfgs {
    encoder: TagEncoder,
    c2: f64,
    trainset: *const Dataset,
    best_w: Vec<f64>,
}

#[derive(Debug, Parser)]
struct training_option_t {
    #[arg(default_value_t = 0.0)]
    c1: f64,
    #[arg(default_value_t = 1.0)]
    c2: f64,
    #[arg(default_value_t = 6)]
    memory: i32,
    #[arg(default_value_t = 1e-5)]
    epsilon: f64,
    #[arg(default_value_t = 10)]
    stop: i32,
    #[arg(default_value_t = 1e-5)]
    delta: f64,
    #[arg(default_value_t = i32::MAX)]
    max_iterations: i32,
    // linesearch: String,
    #[arg(default_value_t = 20)]
    linesearch_max_iterations: i32,
}

#[no_mangle]
unsafe extern "C" fn proc_evaluate(
    instance: *mut c_void,
    x: *const f64,
    g: *mut f64,
    n: i32,
    step: f64,
) -> f64 {
    log::debug!("evaluate step: {step}, inst: {:?}", instance);
    let this = (instance as *mut Ctx).as_mut().expect("null instance");
    log::debug!("ctx c2: {:?}, best_w.len: {}", this.c2, this.best_w.len());
    let trainset = this.trainset.as_ref().expect("null dataset");

    let n = n as usize;
    let gs = slice::from_raw_parts_mut(g, n);
    let xs = slice::from_raw_parts(x, n);

    /* Compute the objective value and gradients. */
    let mut f = this.encoder.objective_and_gradients_batch(trainset, xs, gs);

    /* L2 regularization. */
    let mut norm = 0.0;    
    if this.c2 > 0.0 {
        let c22 = this.c2 * 2.0;
        for i in 0..n {
            gs[i] += c22 * xs[i];
            norm += xs[i] * xs[i];
        }
        f += this.c2 * norm;
    }

    f
}

#[no_mangle]
unsafe extern "C" fn proc_progress(
    instance: *mut c_void,
    x: *const f64,
    g: *const f64,
    fx: f64,
    xnorm: f64,
    gnorm: f64,
    step: f64,
    n: i32,
    k: i32,
    ls: i32,
) -> i32 {
    let this = (instance as *mut Ctx).as_mut().expect("null instance");

    /* Compute the duration required for this iteration. */
    let elapsed = this.timestamp.elapsed();
    this.timestamp = Instant::now();

    let n = n as usize;
    let xs = slice::from_raw_parts(x, n);
    /* Store the feature weight in case L-BFGS terminates with an error. */
    let mut num_active_features = 0;
    for i in 0..n {
        this.best_w[i as usize] = xs[i];
        if xs[i] != 0.0 {
            num_active_features += 1;
        }
    }

    /* Report the progress. */
    log::info!("========> iteration: {k} (loss: {fx:.4}, feature norm: {xnorm:.4}, error norm: {gnorm:.4}, num_active_features: {num_active_features}, line search trials/step: {ls}/{step}, took: {elapsed:?})");

    /* Send the tagger with the current parameters. */
    // TODO

    /* Continue. */
    0
}

pub fn train(mut encoder: TagEncoder, ds: &Dataset, fpath: PathBuf, holdout: usize) {
    let N = ds.len();
    let L = ds.num_labels();
    let A = ds.num_attrs();
    encoder.set_data(ds);
    let K = encoder.num_features();
    let trainset = std::ptr::addr_of!(*ds);
    assert!(!trainset.is_null(), "null ds ptr");
    assert!(K > 0, "number of features should be positive");
    log::info!("L: {L}, A: {A}, N: {N}, K: {K}");
    let mut params = MaybeUninit::<lbfgs_parameter_t>::uninit();
    let mut params = unsafe {
        lbfgs_parameter_init(params.as_mut_ptr());
        params.assume_init()
    };
    // let opt = training_option_t::parse();
    {
        // TODO: hardcode params
        params.m = 6;
        params.epsilon = 1e-5;
        params.past = 10;
        params.delta = 1e-5;
        params.max_iterations = 100;
        params.orthantwise_c = 0.1;
        params.linesearch = 2;
        params.max_linesearch = 20;
    }
    let w = unsafe { lbfgs_malloc(K as i32) };
    assert!(!w.is_null(), "lbfgs_malloc failed");
    log::info!("params: {:?}", params);
    let params = std::ptr::addr_of_mut!(params);
    let ctx = Ctx::new(encoder, trainset);
    let instance = std::ptr::addr_of!(ctx);
    log::info!("pass instance: {:?}", instance);
    let r = unsafe {
        lbfgs(
            K as i32,
            w,
            std::ptr::null_mut(),
            Some(proc_evaluate),
            Some(proc_progress),
            instance as *mut c_void,
            params,
        )
    };
    if r != 0 {
        let s = unsafe { lbfgs_strerror(r) };
        let s = unsafe { CStr::from_ptr(s) };
        log::error!("lbfgs error: {:?}", s);
    }
    // output ctx.best_w;
    unsafe { lbfgs_free(w) };
}
