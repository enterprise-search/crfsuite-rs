trait Transformer {
    fn encode(&self, s: &str) -> usize;
    fn decode(&self, value: usize) -> &str;
}