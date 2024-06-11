#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Opt {
    CTXF_VITERBI = 0x01,
    CTXF_MARGINALS = 0x02,
    CTXF_ALL = 0xFF,
}

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
    state: Vec<f64>,

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
    mexp_state: Vec<f64>,

    /**
     * Model expectations of transitions.
     *  This is a [L][L] matrix whose element [i][j] presents the model
     *  expectation of the transition (i--j).
     *  This member is available only with CTXF_MARGINALS flag.
     */
    mexp_trans: Vec<f64>,
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
        this
    }

    pub fn crf1dc_set_num_items(&mut self, T: usize) {
        let L = self.num_labels;
        self.num_items = T;
        if self.cap_items < T {
            self.alpha_score.resize(T * L, 0.0);
            self.beta_score.resize(T * L, 0.0);
            self.scale_factor.resize(T, 0.0);
            self.row.resize(L, 0.0);

            if self.flag.contains(&Opt::CTXF_VITERBI) {
                self.backward_edge.resize(T * L, 0);
            }

            self.state.resize(T * L, 0.0);
            if self.flag.contains(&Opt::CTXF_MARGINALS) {
                self.exp_state.resize(T * L, 0.0);
                self.mexp_state.resize(T * L, 0.0);
            }

            self.cap_items = T;
        }
    }

    pub(crate) fn reset(&self, reset: ResetOpt) {
        todo!()
    }

    pub fn crf1dc_exp_transition(&mut self) {
        todo!()
    }
}
