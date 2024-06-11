pub trait Dict {
    fn get_or_create(&mut self, key: &str) -> usize;
    fn to_id(&self, key: &str) -> Option<usize>;
    fn to_string(&self, id: usize) -> Option<String>;
    fn len(&self) -> usize;
}