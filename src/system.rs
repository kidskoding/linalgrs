use std::collections::HashMap;
use crate::matrix::Matrix;

pub struct Systems<'a> {
    pub coefficients: Matrix<'a>,
    pub constants: Vec<i64>
}
impl<'a> Systems<'a> {
    pub fn solve() -> Result<HashMap<char, i64>, &'a str> {
        
        Ok(HashMap::new())
    }
}