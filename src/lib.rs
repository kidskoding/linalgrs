pub mod matrix;
pub mod matrix_utilities;
pub mod number;

#[cfg(test)]
mod tests {
    mod matrix_tests {
        use std::sync::Arc;
        use crate::matrix::Matrix;
        use crate::matrix_utilities::MatrixUtilities;
        
        #[test]
        fn test_matrix() {
            let mut mat = Matrix::default();
            let arr = [1, 2, 3];
            mat = MatrixUtilities::append(mat, &arr);
            
            let expected: Vec<Arc<[i64]>> = vec![Arc::new(arr)];
            assert_eq!(mat.mat, expected);
        }
        #[test]
        fn test_shape() {
            let mut mat = Matrix::default();
            let arr = [1, 2, 3];
            mat = MatrixUtilities::append(mat, &arr);
            assert_eq!(mat.shape(), (1, 3))
        }
        
        mod gaussian_elimination_tests {
            use crate::matrix::Matrix;
            use crate::matrix_utilities::MatrixUtilities;
            use std::sync::Arc;
            
            #[test]
            fn test_row_echelon_form() {
                let matrix = Matrix {
                    mat: vec![
                        Arc::from([1.0, 2.0, -1.0].as_slice()),
                        Arc::from([2.0, 3.0, 1.0].as_slice()),
                        Arc::from([3.0, 5.0, 0.0].as_slice()),
                    ],
                    rows: 3,
                    cols: 3,
                };

                let expected = vec![
                    Arc::from([1.0, 2.0, -1.0].as_slice()),
                    Arc::from([0.0, 1.0, -3.0].as_slice()),
                    Arc::from([0.0, 0.0, 0.0].as_slice()),
                ];

                let result = MatrixUtilities::row_echelon_form(matrix);
                assert_eq!(result.mat, expected);
            }
            
            #[test]
            fn test_rref() {
                let mat = Matrix {
                    mat: vec![
                        Arc::from(vec![1.0, 2.0, -1.0]),
                        Arc::from(vec![0.0, 1.0, -3.0]),
                        Arc::from(vec![0.0, 0.0, 0.0]),
                    ],
                    rows: 3,
                    cols: 3,
                };
                
                let expected_rref = vec![
                    Arc::from(vec![1.0, 0.0, 5.0]),
                    Arc::from(vec![0.0, 1.0, -3.0]),
                    Arc::from(vec![0.0, 0.0, 0.0]) 
                ];
                
                let result = MatrixUtilities::rref(mat);
                
                assert_eq!(result.mat, expected_rref);
            }
        }
        
        mod matrix_operations_tests {
            use std::sync::Arc;
            use crate::matrix::Matrix;
            use crate::matrix_utilities::MatrixUtilities;

            #[test]
            fn test_add_matrix() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let result = MatrixUtilities::add(mat.clone(), mat.clone());
                assert_eq!(result.unwrap().mat, vec![Arc::from([2, 4, 6]), Arc::from([8, 10, 12])])
            }
            #[test]
            fn test_add_matrix_different_shape() {
                let mat = Matrix::default();

                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let mat2 = Matrix::default();
                let arr2: &[&[i64]] = &[&[1, 2], &[3, 4]];
                let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

                let result = MatrixUtilities::add(mat, mat2);
                assert!(result.is_err());
            }
            #[test]
            fn test_subtract_matrix() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let result = MatrixUtilities::subtract(mat.clone(), mat.clone());
                assert_eq!(result.unwrap().mat, vec![Arc::from([0, 0, 0]), Arc::from([0, 0, 0])])
            }
            #[test]
            fn test_multiply_by_scalar() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);
                let mat = MatrixUtilities::multiply_by_scalar(mat, 2);
                assert_eq!(mat.mat, vec![Arc::from(&[2, 4, 6][..]), Arc::from(&[8, 10, 12][..])])
            }
            #[test]
            fn test_multiply_by_scalar_with_decimals() {
                let mat = Matrix::default();
                let arr: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let scalar = 2.5;
                let mat = MatrixUtilities::multiply_by_scalar(mat, scalar);

                let expected: Vec<Arc<[f64]>> = vec![
                    Arc::from(&[2.5, 5.0, 7.5][..]),
                    Arc::from(&[10.0, 12.5, 15.0][..])
                ];
                assert_eq!(mat.mat, expected);
            }
            #[test]
            fn test_multiply_matrix() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 4, -2], &[3, 5, -6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let mat2 = Matrix::default();
                let arr2: &[&[i64]] = &[&[5, 2, 8, -1], &[3, 6, 4, 5], &[-2, 9, 7, -3]];
                let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

                let result = MatrixUtilities::multiply(mat, mat2);
                assert!(result.is_ok());
                assert_eq!(result.unwrap().mat, vec![Arc::from([21, 8, 10, 25]),
                                                     Arc::from([42, -18, 2, 40])])
            }
            #[test]
            fn test_multiply_matrix_error() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 4], &[3, 5]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let mat2 = Matrix::default();
                let arr2: &[&[i64]] = &[&[5, 2], &[3, 6], &[3, 4]];
                let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

                let result = MatrixUtilities::multiply(mat, mat2);
                assert!(result.is_err());
            }
            #[test]
            fn test_sub_matrix() {
                let mut mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];

                mat = MatrixUtilities::append_multiple(mat, arr);

                let mat = mat.sub_matrix(0..2, 0..2);
                assert!(mat.is_ok());

                let sub_mat = mat.unwrap();
                let expected: Vec<Arc<[i64]>>
                    = vec![Arc::from(&[1, 2][..]), Arc::from(&[4, 5][..])];
                assert_eq!(sub_mat.mat, expected);
            }
            #[test]
            fn test_dot_product() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let mat2 = Matrix::default();
                let arr2: &[&[i64]] = &[&[1], &[2], &[3]];
                let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

                let result = MatrixUtilities::dot(mat, mat2);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), 14);
            }
            #[test]
            fn test_dot_product_error() {
                let mat = Matrix::default();
                let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
                let mat = MatrixUtilities::append_multiple(mat, arr);

                let mat2 = Matrix::default();
                let arr2: &[&[i64]] = &[&[1], &[2], &[3]];
                let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

                let result = MatrixUtilities::dot(mat, mat2);
                assert!(result.is_err());
            }
        }

        mod determinant_tests {
            use crate::matrix::Matrix;
            use crate::matrix_utilities::MatrixUtilities;

            #[test]
            fn test_determinant() {
                let mut mat = Matrix::default();
                let top = &[1, 2];
                let bottom = &[3, 4];

                mat = MatrixUtilities::append_multiple(mat, &[top, bottom]);

                assert_eq!(mat.determinant().unwrap(), -2);
            }
            #[test]
            fn test_determinant_err() {
                let mut mat = Matrix::default();
                let top = &[1, 2, 3];

                mat = MatrixUtilities::append(mat, top);

                assert!(mat.determinant().is_err());
            }
        }
    }
}