/// A struct representing that of a `Matrix` in linear algebra
///
/// In Linear Algebra, a `Matrix` is a rectangular array of numbers,
/// symbols, or expressions, arranged in rows and columns. Matrices are
/// used to represent and solve systems of linear equations, perform
/// linear transformations, and more
pub struct Matrix<'a> {
    /// `mat` represents a vector of `&[i64]` slices, 
    /// where each slice represents a row in the matrix
    pub mat: Vec<&'a [i64]>,
    
    /// `rows` stores the number of rows in the matrix
    pub rows: usize,
    
    /// `cols` stores the number of columns in the matrix
    pub cols: usize,
}

impl Matrix<'static> {
    /// Creates a new instance of this `Matrix`
    /// 
    /// ### Returns
    /// - A newly constructed `Matrix` object
    pub fn new() -> Matrix<'static> {
        Matrix {
            mat: Vec::new(),
            rows: 0,
            cols: 0,
        }
    }
    
    /// Appends an array to this `Matrix`
    ///
    /// ### Parameters
    /// - arr: A `&[i64]` slice
    /// 
    /// ### Returns
    /// - An updated `Matrix` object that adds `arr` to the `mat` `Vec`
    pub fn append<'a, 'b>(mut matrix: Matrix<'b>, arr: &'b [i64]) -> Matrix<'b> {
        matrix.mat.push(arr);
        matrix.rows = matrix.mat.len();
        matrix.cols = arr.len();
        
        matrix
    }
    
    /// Compute the shape of this `Matrix`
    /// 
    /// ### Returns
    /// - A tuple of two positive integers - `(usize, usize)` - representing
    /// the rows and columns of the matrix
    pub fn shape(&mut self) -> (usize, usize) {
        self.rows = self.mat.len();
        self.cols = if self.rows > 0 {
            self.mat[0].len()
        } else {
            0
        };
        
        (self.rows, self.cols)
    }
    
    /// Compute the determinant of this `Matrix`
    /// 
    /// ### Returns
    /// - A `Result` object determining whether the determinant could be calculated
    ///     - `Err(&str)` - if the `Matrix`'s shape is invalid
    ///     - `Ok(i64)` - if this `Matrix`'s shape is a `(2, 2)`
    pub fn determinant(&mut self) -> Result<i64, &str> {
        if self.shape() == (2, 2) {
            let first = 0;
            let last = self.mat.len() - 1;
            
            let ad = self.mat[first][first] * self.mat[last][last];
            let bc = self.mat[first][last] * self.mat[last][first];
            
            return Ok(ad - bc);
        }
        
        Err("Unable to compute determinant. The shape must be (2, 2)")
    }
}