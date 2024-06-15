use std::{ffi::CStr, mem::MaybeUninit, path::PathBuf};

use clap::Parser;
use libc::c_void;
use liblbfgs_sys::{lbfgs, lbfgs_free, lbfgs_malloc, lbfgs_parameter_init, lbfgs_parameter_t, lbfgs_strerror};

use crate::Dataset;

use super::trainer::{Crf1dTrainer, TagEncoder};


#[derive(Debug)]
pub struct Lbfgs {
    encoder: TagEncoder,
    c2: f64,
    trainset: * const Dataset,
    best_w: Vec<f64>,
}
impl Lbfgs {
    pub fn new(encoder: TagEncoder, trainset: * const Dataset) -> Self {
        Self { c2: 0.1, trainset, best_w: Vec::with_capacity(encoder.num_features()), encoder: encoder }
    }
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
    let this = (instance as * mut Lbfgs).as_mut().expect("null instance");
    let gm = &this.encoder;
    let trainset = this.trainset.as_ref().expect("null dataset");

    /* Compute the objective value and gradients. */
    let mut f = gm.objective_and_gradients_batch(trainset, x, g);
    
    /* L2 regularization. */
    let mut norm = 0.0;
    if this.c2 > 0.0 {
        let c22 = this.c2 * 2.0;
        for i in 0..n {
            let iv = *x.offset(i as isize);
            *g.offset(i as isize) += c22 * iv;
            norm += iv * iv;
        }
        f += this.c2 * norm;
    }

    return f;
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
    todo!()
}

impl Crf1dTrainer for Lbfgs {
    fn train(&mut self, ds: &Dataset, fpath: PathBuf, holdout: usize) {
        let N = ds.len();
        let L = ds.num_labels();
        let A = ds.num_attrs();
        self.encoder.set_data(ds);
        let K = self.encoder.num_features();
        assert!(K > 0, "number of features should be positive");
        log::info!("K: {K}");
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
        }
        let w = unsafe { lbfgs_malloc(K as i32) };
        assert!(!w.is_null(), "lbfgs_malloc failed");
        println!("params: {:?}", params);
        let params = std::ptr::addr_of_mut!(params);
        let instance = std::ptr::addr_of!(self);
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
        for i in 0..K {
            self.best_w.push(unsafe { *w.offset(i as isize) });
        }
        unsafe { lbfgs_free(w) };
    }
}
