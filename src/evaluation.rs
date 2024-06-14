use std::{collections::HashMap, fmt::Display, iter::zip};

#[derive(Debug)]
struct Measure {
    n_match: usize,
    n_total: usize,
}

impl Measure {
    #[inline]
    pub fn precision(&self) -> f64 {
        self.n_match as f64 / self.n_total as f64
    }

    #[inline]
    pub fn recall() -> f64 {
        todo!()
    }

    #[inline]
    pub fn fmeasure() -> f64 {
        todo!()
    }
}

impl Display for Measure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.n_match, self.n_total)
    }
}

/// Label-wise performance values.
#[derive(Debug, Default)]
struct LabelMeasure {
    /// Number of correct predictions.
    num_correct: usize,
    /** Number of occurrences of the label in the gold-standard data. */
    num_observation: usize,
    /** Number of predictions. */
    num_prediction: usize,
    /** Precision. */
    precision: f64,
    /** Recall. */
    recall: f64,
    /** F1 score. */
    fmeasure: f64,
}

/// An overall performance values.
#[derive(Debug, Default)]
pub struct Evaluation {
    /** Number of labels. */
    pub num_labels: usize,
    /** Array of label-wise evaluations. */
    tbl: HashMap<String, LabelMeasure>,

    /** Number of correctly predicted items. */
    item_total_correct: usize,
    /** Total number of items. */
    item_total_num: usize,
    /** Total number of occurrences of labels in the gold-standard data. */
    item_total_observation: usize,
    /** Total number of predictions. */
    item_total_prediction: usize,
    /** Item-level accuracy. */
    item_accuracy: f64,

    /** Number of correctly predicted instances. */
    inst_total_correct: usize,
    /** Total number of instances. */
    inst_total_num: usize,
    /** Instance-level accuracy. */
    inst_accuracy: f64,

    /** Macro-averaged precision. */
    macro_precision: f64,
    /** Macro-averaged recall. */
    macro_recall: f64,
    /** Macro-averaged F1 score. */
    macro_fmeasure: f64,
}

#[derive(Debug)]
pub struct Estimation {
    pub precision: f64,
    pub recall: f64,
}

impl Evaluation {
    pub fn accumulate(&mut self, reference: &[String], prediction: &[&str]) {
        let mut matched = 0;
        for (r, p) in zip(reference, prediction) {
            self.tbl.entry(r.to_string()).or_default().num_observation += 1;
            self.tbl.entry(p.to_string()).or_default().num_prediction += 1;
            if *r == *p {
                self.tbl.entry(r.to_string()).or_default().num_correct += 1;
                matched += 1;
            }
            self.item_total_num += 1;
        }

        if matched == prediction.len() {
            self.inst_total_correct += 1;
        }
        self.inst_total_num += 1;
    }

    pub fn evaluate(&mut self) -> Estimation {
        for (label, lev) in &mut self.tbl {
            if lev.num_observation == 0 {
                continue;
            }
            self.item_total_correct += lev.num_correct;
            self.item_total_prediction += lev.num_prediction;
            self.item_total_observation += lev.num_observation;

            lev.precision = 0.0;
            lev.recall = 0.0;
            lev.fmeasure = 0.0;

            if lev.num_prediction > 0 {
                lev.precision = lev.num_correct as f64 / lev.num_prediction as f64;
            }
            if lev.num_observation > 0 {
                lev.recall = lev.num_correct as f64 / lev.num_observation as f64;
            }
            if lev.precision + lev.recall > 0.0 {
                lev.fmeasure = lev.precision * lev.recall * 2.0 / (lev.precision + lev.recall);
            }
            self.macro_precision += lev.precision;
            self.macro_recall += lev.recall;
            self.macro_fmeasure += lev.fmeasure;
        }

        self.macro_precision /= self.num_labels as f64;
        self.macro_recall /= self.num_labels as f64;
        self.macro_fmeasure /= self.num_labels as f64;

        if self.item_total_num > 0 {
            self.item_accuracy = self.item_total_correct as f64 / self.item_total_num as f64;
        }
        if self.inst_total_num > 0 {
            self.inst_accuracy = self.inst_total_correct as f64 / self.inst_total_num as f64;
        }
        Estimation { precision: self.macro_precision, recall: self.macro_recall }
    }
}

impl Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance by label (#match, #model, #ref) (precision, recall, F1):").unwrap();
        for (label, lev) in &self.tbl {
            if lev.num_observation == 0 {
                writeln!(f, "\t{}: ({}, {}, {}) (******, ******, ******)", label, lev.num_correct, lev.num_prediction, lev.num_observation).unwrap();
            } else {
                writeln!(f, "\t{}: ({}, {}, {}) ({:.4}, {:.4}, {:.4})", label, lev.num_correct, lev.num_prediction, lev.num_observation,
                lev.precision, lev.recall, lev.fmeasure
            ).unwrap();
            }
        }
        writeln!(f, "Macro-average precision, recall, F1: ({}, {}, {})", self.macro_precision, self.macro_recall, self.macro_fmeasure).unwrap();
        writeln!(f, "Item accuracy: {}/{} => {}", self.item_total_correct, self.item_total_num, self.item_accuracy).unwrap();
        writeln!(f, "Sequence accuracy: {}/{} => {}", self.inst_total_correct, self.inst_total_num, self.inst_accuracy)
    }
}