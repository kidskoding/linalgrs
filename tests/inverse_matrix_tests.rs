mod test_inverse_matrices {
    use std::sync::Arc;
    use float_cmp::approx_eq;
    use linalgrs::matrix::Matrix;
    use linalgrs::matrix_utilities::MatrixUtilities;

    #[test]
    fn test_inverse() {
        let matrix = Matrix {
            mat: vec![
                Arc::from([4.0, 7.0]),
                Arc::from([2.0, 6.0]),
            ],
            rows: 2,
            cols: 2,
        };

        let expected_inverse = Matrix {
            mat: vec![
                Arc::from([0.6, -0.7]),
                Arc::from([-0.2, 0.4]),
            ],
            rows: 2,
            cols: 2,
        };

        let result = MatrixUtilities::inverse(matrix);
        assert!(result.is_ok());
        let inverse = result.unwrap();
        for i in 0..inverse.rows {
            for j in 0..inverse.cols {
                assert!(
                    approx_eq!(f64, inverse.mat[i][j], expected_inverse.mat[i][j], epsilon = 1e-6),
                    "Mismatch at ({}, {}): got {}, expected {}",
                    i, j, inverse.mat[i][j], expected_inverse.mat[i][j]
                );
            }
        }
        assert_eq!(inverse.rows, expected_inverse.rows);
        assert_eq!(inverse.cols, expected_inverse.cols);
    }

    #[test]
    fn test_non_invertible_matrix() {
        let singular_matrix = Matrix {
            mat: vec![
                Arc::from([2.0, 4.0]),
                Arc::from([1.0, 2.0]),
            ],
            rows: 2,
            cols: 2,
        };

        let result = MatrixUtilities::inverse(singular_matrix);
        assert!(result.is_err());
    }
}