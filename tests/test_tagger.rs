extern crate crfsuite;

use crfsuite::{Attribute, Error, Model};

#[test]
fn test_open_model() {
    let model = Model::from_file("tests/model.crfsuite").unwrap();
    let tagger = model.tagger().unwrap();
    let _labels = tagger.labels().unwrap();
}

#[test]
fn test_open_not_existing_model_does_not_panic() {
    let ret = Model::from_file("tests/does-not-exists.crfsuite");
    match ret {
        Err(Error::InvalidModel(..)) => {}
        _ => panic!("test fail"),
    }
}

#[test]
fn test_dump_model() {
    let model = Model::from_file("tests/model.crfsuite").unwrap();
    model.dump_file("tests/model.dump").unwrap();
}

#[test]
fn test_create_model_from_memory() {
    let model_memory = include_bytes!("model.crfsuite");
    let model = Model::from_memory(&model_memory[..]).unwrap();
    let mut tagger = model.tagger().unwrap();
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![],
        vec![Attribute::new("clean", 1.0)],
    ];
    let yseq = [
        "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
    ];
    let res = tagger.tag(&xseq).unwrap();
    assert_eq!(res, yseq);

    tagger.probability(&yseq).unwrap();
    tagger.marginal("sunny", 1i32).unwrap();
}

#[test]
fn test_create_model_from_memory_invalid_model() {
    let ret = Model::from_memory(b"");
    match ret {
        Err(Error::InvalidModel(..)) => {}
        _ => panic!("test fail"),
    }

    let ret = Model::from_memory(b"abcdefg");
    match ret {
        Err(Error::InvalidModel(..)) => {}
        _ => panic!("test fail"),
    }

    let ret = Model::from_memory(b"lCRF");
    match ret {
        Err(Error::InvalidModel(..)) => {}
        _ => panic!("test fail"),
    }

    let ret = Model::from_memory(b"lCRFabcdefg");
    match ret {
        Err(Error::InvalidModel(..)) => {}
        _ => panic!("test fail"),
    }
}

#[test]
fn test_tag() {
    let model = Model::from_file("tests/model.crfsuite").unwrap();
    let mut tagger = model.tagger().unwrap();
    let xseq = vec![
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![Attribute::new("walk", 1.0)],
        vec![Attribute::new("walk", 1.0), Attribute::new("clean", 0.5)],
        vec![Attribute::new("shop", 0.5), Attribute::new("clean", 0.5)],
        vec![Attribute::new("walk", 0.5), Attribute::new("clean", 1.0)],
        vec![Attribute::new("clean", 1.0), Attribute::new("shop", 0.1)],
        vec![Attribute::new("walk", 1.0), Attribute::new("shop", 0.5)],
        vec![],
        vec![Attribute::new("clean", 1.0)],
    ];
    let yseq = [
        "sunny", "sunny", "sunny", "rainy", "rainy", "rainy", "sunny", "sunny", "rainy",
    ];
    let res = tagger.tag(&xseq).unwrap();
    assert_eq!(res, yseq);

    tagger.probability(&yseq).unwrap();
    tagger.marginal("sunny", 1i32).unwrap();
}

// test result for test.data (esp.testa)
// Performance by label (#match, #model, #ref) (precision, recall, F1):
//         B-ORG: (1105, 1323, 1700) (0.8352, 0.6500, 0.7311)
//         O: (45076, 46851, 45356) (0.9621, 0.9938, 0.9777)
//         I-ORG: (890, 1047, 1366) (0.8500, 0.6515, 0.7377)
//         I-PER: (685, 813, 859) (0.8426, 0.7974, 0.8194)
//         I-MISC: (293, 413, 654) (0.7094, 0.4480, 0.5492)
//         I-LOC: (238, 322, 337) (0.7391, 0.7062, 0.7223)
//         B-LOC: (728, 1004, 984) (0.7251, 0.7398, 0.7324)
//         B-PER: (770, 894, 1222) (0.8613, 0.6301, 0.7278)
//         B-MISC: (182, 256, 445) (0.7109, 0.4090, 0.5193)
// Macro-average precision, recall, F1: (0.8039834744027092, 0.6695541387706868, 0.7240860816513606)
// Item accuracy: 49967/52923 => 0.9441452676530053
// Sequence accuracy: 1061/1915 => 0.554046997389034

/*
Performance by label (#match, #model, #ref) (precision, recall, F1):
    B-LOC: (983, 984, 984) (0.9990, 0.9990, 0.9990)
    I-LOC: (336, 336, 337) (1.0000, 0.9970, 0.9985)
    O: (45353, 45356, 45356) (0.9999, 0.9999, 0.9999)
    B-ORG: (1698, 1702, 1700) (0.9976, 0.9988, 0.9982)
    I-ORG: (1366, 1375, 1366) (0.9935, 1.0000, 0.9967)
    B-PER: (1221, 1222, 1222) (0.9992, 0.9992, 0.9992)
    I-PER: (859, 859, 859) (1.0000, 1.0000, 1.0000)
    B-MISC: (442, 442, 445) (1.0000, 0.9933, 0.9966)
    I-MISC: (645, 647, 654) (0.9969, 0.9862, 0.9915)
Macro-average precision, recall, F1: (0.998457, 0.997050, 0.997748)
Item accuracy: 52903 / 52923 (0.9996)
Instance accuracy: 1906 / 1915 (0.9953)
Elapsed time: 0.716984 [sec] (2670.9 [instance/sec])
 */