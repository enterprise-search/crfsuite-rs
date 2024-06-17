# crfsuite-rs

[![Rust](https://github.com/messense/crfsuite-rs/workflows/Rust/badge.svg)](https://github.com/messense/crfsuite-rs/actions?query=workflow%3ARust)
[![Python](https://github.com/messense/crfsuite-rs/workflows/Python/badge.svg)](https://github.com/messense/crfsuite-rs/actions?query=workflow%3APython)
[![codecov](https://codecov.io/gh/messense/crfsuite-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/messense/crfsuite-rs)
[![Crates.io](https://img.shields.io/crates/v/crfsuite.svg)](https://crates.io/crates/crfsuite)
[![docs.rs](https://docs.rs/crfsuite/badge.svg)](https://docs.rs/crfsuite/)
[![PyPI](https://img.shields.io/pypi/v/crfsuite)](https://pypi.org/project/crfsuite)

Rust binding to [crfsuite](https://github.com/chokkan/crfsuite)

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
crfsuite = "0.3"
```

Add ``extern crate crfsuite`` to your crate root and your're good to go!

## Python package

There is also a Python package named `crfsuite`, you can install it via `pip`:

```bash
pip install -U crfsuite
```

Usage example:

```python
from crfsuite import Model

if __name__ == '__main__':
    model = Model('path/to/crfsuite/model.crf')
    tagged = model.tag(["abc", "def"])
    print(tagged)
```

## Run

```shell
RUST_LOG=info cargo run --example train -- -p c1=0.1 -p c2=0.1 -p max_iterations=100 -p feature.possible_transitions=1 -m ner train.data -v -v
RUST_LOG=info cargo run --example tag -- -t -m ner test.data
```

### Benchmark

```shell
cargo bench --bench predict_benchmark
python3 -m locust --host http://localhost:8080 -f locustfile.py
```

## Internal

- interface refine
    - fit (train)
    - evaluate

```
model.fit(train_images, 
        train_labels,
        epochs=50, 
        batch_size=batch_size, 
        callbacks=[cp_callback],
        validation_data=(test_images, test_labels),
        verbose=0)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
model.save('my_model.keras')
model = load_model('my_model.keras')
new_model.summary()
new_model.predict(test_images)
```
- ref https://www.tensorflow.org/tutorials/keras/text_classification
- dataset refine
    - train/validation/test
- structure:
    - tower::service API

## Performance

- Optimization
    - https://nnethercote.github.io/perf-book/title-page.html

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.

