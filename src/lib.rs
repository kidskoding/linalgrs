pub mod matrix;
pub mod system;
pub mod matrix_utilities;
mod number;

#[cfg(test)]
mod tests {
    mod matrix_tests {
        use std::sync::Arc;
        use crate::matrix::Matrix;
        use crate::matrix_utilities::MatrixUtilities;

        #[test]
        fn test_matrix() {
            let mut mat = Matrix::new();
            let arr = [1, 2, 3];
            mat = MatrixUtilities::append(mat, &arr);
            
            let expected: Vec<Arc<[i64]>> = vec![Arc::new(arr)];
            assert_eq!(mat.mat, expected);
        }
        #[test]
        fn test_shape() {
            let mut mat = Matrix::new();
            let arr = [1, 2, 3];
            mat = MatrixUtilities::append(mat, &arr);
            assert_eq!(mat.shape(), (1, 3))
        }
        #[test]
        fn test_multiply_by_scalar() {
            let mat = Matrix::new();
            let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
            let mat = MatrixUtilities::append_multiple(mat, arr);
            let mat = MatrixUtilities::multiply_by_scalar(mat, 2);
            assert_eq!(mat.mat, vec![Arc::from(&[2, 4, 6][..]), Arc::from(&[8, 10, 12][..])])
        }
        #[test]
        fn test_sub_matrix() {
            let mut mat = Matrix::new();
            let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];

            mat = MatrixUtilities::append_multiple(mat, arr);

            let mat = mat.sub_matrix(0..2, 0..2);
            assert_eq!(mat.is_ok(), true);
            
            let sub_mat = mat.unwrap();
            let expected: Vec<Arc<[i64]>> 
                = vec![Arc::from(&[1, 2][..]), Arc::from(&[4, 5][..])];
            assert_eq!(sub_mat.mat, expected);
        }
        mod determinant_test {
            use crate::matrix::Matrix;
            use crate::matrix_utilities::MatrixUtilities;

            #[test]
            fn test_determinant() {
                let mut mat = Matrix::new();
                let top = &[1, 2];
                let bottom = &[3, 4];

                mat = MatrixUtilities::append_multiple(mat, &[top, bottom]);

                assert_eq!(mat.determinant().unwrap(), -2);
            }
            #[test]
            fn test_determinant_err() {
                let mut mat = Matrix::new();
                let top = &[1, 2, 3];

                mat = MatrixUtilities::append(mat, top);

                assert_eq!(mat.determinant().is_err(), true);
            }
        }
        mod gaussian_test {
            use std::sync::Arc;
            use crate::matrix::Matrix;
            use crate::system::System;

            #[test]
            fn test_gaussian_elimination() {
                let coefficients = Matrix {
                    rows: 3,
                    cols: 3,
                    mat: vec![
                        Arc::from([2.0, 1.0, -1.0]),
                        Arc::new([-3.0, -1.0, 2.0]),
                        Arc::new([-2.0, 1.0, 2.0]),
                    ],
                };
                let constants = vec![8.0, -11.0, -3.0];
                let mut system = System {
                    coefficients,
                    variables: vec![],
                    constants,
                };

                let result = System::gaussian_elimination(&mut system);

                assert!(result.is_ok());
                let solution = result.unwrap();
                let expected_solution = vec![2.0, 3.0, -1.0];
                assert_eq!(solution, expected_solution);
            } 
        }
    }
}