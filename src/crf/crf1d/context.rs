use std::{iter::zip, ops::Sub};

use libc::size_t;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Opt {
    CTXF_VITERBI = 0x01,
    CTXF_MARGINALS = 0x02,
    CTXF_ALL = 0xFF,
}

#[derive(Debug, PartialEq)]
pub(crate) enum ResetOpt {
    RF_STATE = 0x01,
    RF_TRANS = 0x02,
    RF_ALL = 0xFF,
}

#[derive(Debug, Default)]
pub(crate) struct Crf1dContext {
    /// Flag specifying the functionality.
    flag: Vec<Opt>,
    /**
     * The total number of distinct labels (L).
     */
    pub num_labels: usize,

    /**
     * The number of items (T) in the instance.
     */
    num_items: usize,

    /**
     * The maximum number of labels.
     */
    cap_items: usize,
    /**
     * Logarithm of the normalization factor for the instance.
     *  This is equivalent to the total scores of all paths in the lattice.
     */
    log_norm: f64,
    /**
     * State scores.
     *  This is a [T][L] matrix whose element [t][l] presents total score
     *  of state features associating label #l at #t.
     */
    pub state: Vec<f64>,

    /**
     * Transition scores.
     *  This is a [L][L] matrix whose element [i][j] represents the total
     *  score of transition features associating labels #i and #j.
     */
    pub trans: Vec<f64>,

    /**
     * Alpha score matrix.
     *  This is a [T][L] matrix whose element [t][l] presents the total
     *  score of paths starting at BOS and arraiving at (t, l).
     */
    alpha_score: Vec<f64>,

    /**
     * Beta score matrix.
     *  This is a [T][L] matrix whose element [t][l] presents the total
     *  score of paths starting at (t, l) and arraiving at EOS.
     */
    beta_score: Vec<f64>,

    /**
     * Scale factor vector.
     *  This is a [T] vector whose element [t] presents the scaling
     *  coefficient for the alpha_score and beta_score.
     */
    scale_factor: Vec<f64>,

    /**
     * Row vector (work space).
     *  This is a [T] vector used internally for a work space.
     */
    row: Vec<f64>,

    /**
     * Backward edges.
     *  This is a [T][L] matrix whose element [t][j] represents the label #i
     *  that yields the maximum score to arrive at (t, j).
     *  This member is available only with CTXF_VITERBI flag enabled.
     */
    backward_edge: Vec<i32>,

    /**
     * Exponents of state scores.
     *  This is a [T][L] matrix whose element [t][l] presents the exponent
     *  of the total score of state features associating label #l at #t.
     *  This member is available only with CTXF_MARGINALS flag.
     */
    exp_state: Vec<f64>,

    /**
     * Exponents of transition scores.
     *  This is a [L][L] matrix whose element [i][j] represents the exponent
     *  of the total score of transition features associating labels #i and #j.
     *  This member is available only with CTXF_MARGINALS flag.
     */
    exp_trans: Vec<f64>,

    /**
     * Model expectations of states.
     *  This is a [T][L] matrix whose element [t][l] presents the model
     *  expectation (marginal probability) of the state (t,l)
     *  This member is available only with CTXF_MARGINALS flag.
     */
    pub mexp_state: Vec<f64>,

    /**
     * Model expectations of transitions.
     *  This is a [L][L] matrix whose element [i][j] presents the model
     *  expectation of the transition (i--j).
     *  This member is available only with CTXF_MARGINALS flag.
     */
    pub mexp_trans: Vec<f64>,
}

impl Crf1dContext {
    pub fn new(flag: &[Opt], L: usize, T: usize) -> Self {
        let mut this = Self {
            flag: flag.to_vec(),
            trans: vec![0.0; L * L],
            num_labels: L,
            num_items: 0,
            ..Default::default()
        };
        if this.flag.contains(&Opt::CTXF_MARGINALS) {
            this.exp_trans.resize(L * L, 0.0);
            this.mexp_trans.resize(L * L, 0.0);
        }
        this.crf1dc_set_num_items(T);
        this.num_items = 0;
        this
    }

    pub fn crf1dc_set_num_items(&mut self, T: usize) {
        let L = self.num_labels;
        self.num_items = T;
        if self.cap_items < T {
            self.alpha_score.resize(T * L, 0.0);
            self.beta_score.resize(T * L, 0.0);
            for i in 0..T * L {
                self.alpha_score[i] = 0.0;
                self.beta_score[i] = 0.0;
            }
            self.scale_factor.resize(T, 0.0);
            for i in 0..T {
                self.scale_factor[i] = 0.0;
            }
            self.row.resize(L, 0.0);
            for i in 0..L {
                self.row[i] = 0.0;
            }

            if self.flag.contains(&Opt::CTXF_VITERBI) {
                self.backward_edge.resize(T * L, 0);
                for i in 0..T * L {
                    self.backward_edge[i] = 0;
                }
            }

            self.state.resize(T * L, 0.0);
            for i in 0..T * L {
                self.state[i] = 0.0;
            }
            if self.flag.contains(&Opt::CTXF_MARGINALS) {
                self.exp_state.resize(T * L, 0.0);
                self.mexp_state.resize(T * L, 0.0);
                for i in 0..T * L {
                    self.exp_state[i] = 0.0;
                    self.mexp_state[i] = 0.0;
                }
            }

            self.cap_items = T;
        }
    }

    pub(crate) fn reset(&mut self, opts: &[ResetOpt]) {
        let T = self.num_items;
        let L = self.num_labels;

        if opts.contains(&ResetOpt::RF_STATE) {
            for i in 0..self.state.len() {
                self.state[i] = 0.0;
            }
        }
        if opts.contains(&ResetOpt::RF_TRANS) {
            for i in 0..L * L {
                self.trans[i] = 0.0;
            }
        }

        if self.flag.contains(&Opt::CTXF_MARGINALS) {
            for i in 0..T * L {
                self.mexp_state[i] = 0.0;
            }
            for i in 0..L * L {
                self.mexp_trans[i] = 0.0;
            }
            self.log_norm = 0.0;
        }
    }

    pub fn crf1dc_exp_transition(&mut self) {
        let L = self.num_labels;
        for i in 0..L * L {
            self.exp_trans[i] = self.trans[i].exp();
        }
    }

    pub(crate) fn viterbi(&mut self, labels: &mut [usize]) -> f64 {
        let T = self.num_items;
        let L = self.num_labels;

        /* This function assumes state and trans scores to be in the logarithm domain. */
        /* Compute the scores at (0, *). */
        for j in 0..L {
            self.alpha_score[self.num_labels * 0 + j] = self.state[self.num_labels * 0 + j];
        }
        /* Compute the scores at (t, *). */
        for t in 1..T {
            /* Compute the score of (t, j). */
            for j in 0..L {
                let mut max_score = f64::MIN;
                let mut argmax_score = -1;
                for i in 0..L {
                    /* Transit from (t-1, i) to (t, j). */
                    let score = ((self.alpha_score)[(self.num_labels) * (t - 1) + (i)])
                        + ((self.trans)[(self.num_labels) * (i) + (j)]);

                    /* Store this path if it has the maximum score. */
                    if max_score < score {
                        max_score = score;
                        argmax_score = i as i32;
                    }
                }
                /* Backward link (#t, #j) -> (#t-1, #i). */
                if argmax_score >= 0 {
                    self.backward_edge[self.num_labels * (t) + j] = argmax_score;
                }
                /* Add the state score on (t, j). */
                self.alpha_score[self.num_labels * (t) + j] =
                    max_score + self.state[self.num_labels * (t) + j];
            }
        }

        /* Find the node (#T, #i) that reaches EOS with the maximum score. */
        let mut max_score = f64::MIN;
        /* Set a score for T-1 to be overwritten later. Just in case we don't
        end up with something beating -FLOAT_MAX. */
        labels[T - 1] = 0;
        for i in 0..L {
            let prev = self.alpha_score[(self.num_labels) * (T - 1) + (i)];
            if max_score < prev {
                max_score = prev;
                labels[T - 1] = i; /* Tag the item #T. */
            }
        }
        /* Tag labels by tracing the backward links. */
        for t in (0..T - 1).rev() {
            let i = labels[t + 1];
            labels[t] = self.backward_edge[(self.num_labels) * (t + 1) + i] as usize;
        }

        /* Return the maximum score (without the normalization factor subtracted). */
        max_score
    }

    pub(crate) fn crf1dc_lognorm(&self) -> f64 {
        self.log_norm
    }

    pub(crate) fn crf1dc_exp_state(&mut self) {
        for i in 0..self.state.len() {
            self.exp_state[i] = self.state[i].exp();
        }
    }

    pub(crate) fn crf1dc_alpha_score(&mut self) {
        let L = self.num_labels;
        let T = self.num_items;

        /* Compute the alpha scores on nodes (0, *). alpha[0][j] = state[0][j] */
        for i in 0..L {
            (self.alpha_score)[(self.num_labels) * (0) + (i)] = (self.exp_state)[(self.num_labels) * (0) + (i)];
        }
        let mut sum = 0.0;
        for i in 0..L {            
            sum += (self.alpha_score)[(self.num_labels) * (0) + (i)];
        }        
        self.scale_factor[0] = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        for i in 0..L {
            (self.alpha_score)[(self.num_labels) * (0) + (i)] *= self.scale_factor[0];
        }

        /* Compute the alpha scores on nodes (t, *).
            alpha[t][j] = state[t][j] * \sum_{i} alpha[t-1][i] * trans[i][j]
        */        
        for t in 1..T {
            for i in 0..L {
                ((self.alpha_score)[(self.num_labels) * (t) + (i)]) = 0.0;
            }
            for i in 0..L {
                for j in 0..L {
                    (self.alpha_score)[(self.num_labels) * (t) + (j)] += (self.alpha_score)[(self.num_labels) * (t - 1) + (i)] * (self.exp_trans)[(self.num_labels) * (i) + (j)];
                }
            }
            for i in 0..L {
                (self.alpha_score)[(self.num_labels) * (t) + (i)] *= (self.exp_state)[(self.num_labels) * (t) + (i)];
            }
            sum = 0.0;
            for i in 0..L {
                sum += (self.alpha_score)[(self.num_labels) * (t) + (i)];
            }
            self.scale_factor[t] = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            for i in 0..L {
                (self.alpha_score)[(self.num_labels) * (t) + (i)] *= self.scale_factor[t];
            }
        }

        /* Compute the logarithm of the normalization factor here.
        norm = 1. / (C[0] * C[1] ... * C[T-1])
        log(norm) = - \sum_{t = 0}^{T-1} log(C[t]).
        */
        sum = 0.0;
        for i in 0..T {
            sum += self.scale_factor[i].ln();
        }
        self.log_norm = -sum;
    }

    pub(crate) fn crf1dc_beta_score(&mut self) {
        let T = self.num_items;
        let L = self.num_labels;

        /* Compute the beta scores at (T-1, *). */
        for i in 0..L {
            (self.beta_score)[(self.num_labels) * (T - 1) + (i)] = self.scale_factor[T-1];
        }        

        /* Compute the beta scores at (t, *). */
        for t in (0..T - 1).rev() {
            for i in 0..L {
                self.row[i] = (self.beta_score)[(self.num_labels) * (t + 1) + (i)];
            }
            for i in 0..L {
                self.row[i] *= (self.exp_state)[(self.num_labels) * (t + 1) + (i)];
            }

            /* Compute the beta score at (t, i). */
            for i in 0..L {
                let mut s = 0.0;
                for j in 0..L {
                    s += (self.exp_trans)[(self.num_labels) * (i) + (j)] * self.row[j];
                }
                (self.beta_score)[(self.num_labels) * (t) + (i)] = s;
            }
            for i in 0..L {
                (self.beta_score)[(self.num_labels) * (t) + (i)] *= self.scale_factor[t];
            }
        }
    }

    pub(crate) fn crf1dc_marginals(&mut self) {
        let L = self.num_labels;
        let T = self.num_items;

        /*
        Compute the model expectations of states.
            p(t,i) = fwd[t][i] * bwd[t][i] / norm
                   = (1. / C[t]) * fwd'[t][i] * bwd'[t][i]
        */
        for t in 0..T {
            for i in 0..L {
                (self.mexp_state)[(self.num_labels) * (t) + (i)] =
                    (self.alpha_score)[(self.num_labels) * (t) + (i)];
            }
            for i in 0..L {
                (self.mexp_state)[(self.num_labels) * (t) + (i)] *=
                    (self.beta_score)[(self.num_labels) * (t) + (i)];
            }
            for i in 0..L {
                (self.mexp_state)[(self.num_labels) * (t) + (i)] *= 1.0 / self.scale_factor[t];
            }
        }

        /*
        Compute the model expectations of transitions.
            p(t,i,t+1,j)
                = fwd[t][i] * edge[i][j] * state[t+1][j] * bwd[t+1][j] / norm
                = (fwd'[t][i] / (C[0] ... C[t])) * edge[i][j] * state[t+1][j] * (bwd'[t+1][j] / (C[t+1] ... C[T-1])) * (C[0] * ... * C[T-1])
                = fwd'[t][i] * edge[i][j] * state[t+1][j] * bwd'[t+1][j]
        The model expectation of a transition (i -> j) is the sum of the marginal
        probabilities p(t,i,t+1,j) over t.
        */
        for t in 0..T - 1 {
            /* row[j] = state[t+1][j] * bwd'[t+1][j] */
            for i in 0..L {
                self.row[i] = (self.beta_score)[(self.num_labels) * (t + 1) + (i)];
            }
            for i in 0..L {
                self.row[i] *= (self.exp_state)[(self.num_labels) * (t + 1) + (i)];
            }

            for i in 0..L {
                for j in 0..L {
                    (self.mexp_trans)[(self.num_labels) * (i) + (j)] += (self.alpha_score)
                        [(self.num_labels) * (t) + (i)]
                        * (self.exp_trans)[(self.num_labels) * (i) + (j)]
                        * self.row[j];
                }
            }
        }
    }

    pub(crate) fn crf1dc_score(&self, labels: &Vec<usize>) -> f64 {
        assert!(!labels.is_empty(), "empty labels");
        let L = self.num_labels;
        let T = self.num_items;

        /* Stay at (0, labels[0]). */
        let mut i = labels[0];
        let mut r = (self.state)[(self.num_labels) * (0) + (i)];

        /* Loop over the rest of items. */
        for t in 1..T {
            let j = labels[t];

            /* Transit from (t-1, i) to (t, j). */
            r += (self.trans)[(self.num_labels) * (i) + (j)];
            r += (self.state)[(self.num_labels) * (t) + (j)];
            i = j;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init() {
        let L = 9;
        let T = 12;
        let ctx = Crf1dContext::new(&[Opt::CTXF_MARGINALS, Opt::CTXF_VITERBI], L, T);
        assert_eq!(ctx.num_items, 0);
        assert_eq!(ctx.cap_items, T);
    }

    #[test]
    fn reset() {
        let L = 9;
        let T = 12;
        let mut ctx = Crf1dContext::new(&[Opt::CTXF_MARGINALS, Opt::CTXF_VITERBI], L, T);
        assert_eq!(ctx.num_items, 0);
        assert_eq!(ctx.cap_items, T);
        ctx.reset(&[ResetOpt::RF_STATE]);
        assert_eq!(ctx.log_norm, 0.0);
    }
}