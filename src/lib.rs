pub mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn test_matrix() {
        let mut mat = Matrix::new();
        let arr = &[1, 2, 3];
        mat = Matrix::append(mat, arr);
        assert_eq!(mat.mat, vec![arr]);
    }
    
    #[test]
    fn test_shape() {
        let mut mat = Matrix::new();
        let arr = &[1, 2, 3];
        mat = Matrix::append(mat, arr);
        assert_eq!(mat.shape(), (1, 3))
    }
    
    #[test]
    fn test_determinant() {
        let mut mat = Matrix::new();
        let top = &[1, 2];
        let bottom = &[3, 4];
        
        mat = Matrix::append(mat, top);
        mat = Matrix::append(mat, bottom);
        
        assert_eq!(mat.determinant().unwrap(), -2);
    }
    
    #[test]
    fn test_determinant_err() {
        let mut mat = Matrix::new();
        let top = &[1, 2, 3];
        
        mat = Matrix::append(mat, top);
        
        assert_eq!(mat.determinant().is_err(), true);
    }
}