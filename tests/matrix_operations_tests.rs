mod matrix_operations_tests {
    use linalgrs::matrix::Matrix;
    use linalgrs::matrix_utilities::MatrixUtilities;
    use std::sync::Arc;
    use linalgrs::matrix;

    #[test]
    fn test_add_matrix() {
        let mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
        let mat = MatrixUtilities::append_multiple(mat, arr);

        let result = MatrixUtilities::add(&mat, &mat);
        assert_eq!(
            result.unwrap().mat,
            vec![Arc::from([2, 4, 6]), Arc::from([8, 10, 12])]
        )
    }
    #[test]
    fn test_add_matrix_different_shape() {
        let mat = Matrix::default();

        let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
        let mat = MatrixUtilities::append_multiple(mat, arr);

        let mat2 = Matrix::default();
        let arr2: &[&[i64]] = &[&[1, 2], &[3, 4]];
        let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

        let result = MatrixUtilities::add(&mat, &mat2);
        assert!(result.is_err());
    }
    #[test]
    fn test_subtract_matrix() {
        let mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
        let mat = MatrixUtilities::append_multiple(mat, arr);

        let result = MatrixUtilities::subtract(&mat, &mat);
        assert_eq!(
            result.unwrap().mat,
            vec![Arc::from([0, 0, 0]), Arc::from([0, 0, 0])]
        )
    }
    #[test]
    fn test_multiply_by_scalar() {
        let mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6]];
        let mat = MatrixUtilities::append_multiple(mat, arr);
        let mat = MatrixUtilities::multiply_by_scalar(mat, 2);
        assert_eq!(
            mat.mat,
            vec![Arc::from(&[2, 4, 6][..]), Arc::from(&[8, 10, 12][..])]
        )
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
            Arc::from(&[10.0, 12.5, 15.0][..]),
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

        let result = MatrixUtilities::multiply(&mat, &mat2);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().mat,
            vec![Arc::from([21, 8, 10, 25]), Arc::from([42, -18, 2, 40])]
        )
    }
    #[test]
    fn test_multiply_matrix_error() {
        let mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 4], &[3, 5]];
        let mat = MatrixUtilities::append_multiple(mat, arr);

        let mat2 = Matrix::default();
        let arr2: &[&[i64]] = &[&[5, 2], &[3, 6], &[3, 4]];
        let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

        let result = MatrixUtilities::multiply(&mat, &mat2);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix {
            mat: vec![Arc::from([1, 2, 3]), Arc::from([4, 5, 6])],
            rows: 2,
            cols: 3,
        };

        let transposed = MatrixUtilities::transpose(&mat);

        let expected = Matrix {
            mat: vec![Arc::from([1, 4]), Arc::from([2, 5]), Arc::from([3, 6])],
            rows: 3,
            cols: 2,
        };

        assert_eq!(transposed.mat, expected.mat);
        assert_eq!(transposed.rows, expected.rows);
        assert_eq!(transposed.cols, expected.cols);
    }

    #[test]
    fn test_dot_product() {
        let mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 2, 3]];
        let mat = MatrixUtilities::append_multiple(mat, arr);

        let mat2 = Matrix::default();
        let arr2: &[&[i64]] = &[&[1], &[2], &[3]];
        let mat2 = MatrixUtilities::append_multiple(mat2, arr2);

        let result = MatrixUtilities::dot(&mat, &mat2);
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

        let result = MatrixUtilities::dot(&mat, &mat2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gauss_jordan_elimination_unique_solution() {
        let matrix = Matrix {
            mat: vec![
                Arc::from(vec![2.0, 1.0, -1.0, 8.0]),
                Arc::from(vec![-3.0, -1.0, 2.0, -11.0]),
                Arc::from(vec![-2.0, 1.0, 2.0, -3.0]),
            ],
            rows: 3,
            cols: 4,
        };

        let result = MatrixUtilities::gauss_jordan_elimination(matrix);
        assert!(result.is_ok());
        let pivot_vars = result.unwrap();
        assert_eq!(pivot_vars.get(&'a'), Some(&2.0));
        assert_eq!(pivot_vars.get(&'b'), Some(&3.0));
        assert_eq!(pivot_vars.get(&'c'), Some(&-1.0));
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix = Matrix {
            mat: vec![
                Arc::from([4.0, 3.0].as_slice()),
                Arc::from([6.0, 3.0].as_slice()),
            ],
            rows: 2,
            cols: 2,
        };

        let (l, u) = MatrixUtilities::lu_decomposition(&matrix).unwrap();

        let expected_l = Matrix {
            mat: vec![
                Arc::from([1.0, 0.0].as_slice()),
                Arc::from([1.5, 1.0].as_slice()),
            ],
            rows: 2,
            cols: 2,
        };

        let expected_u = Matrix {
            mat: vec![
                Arc::from([4.0, 3.0].as_slice()),
                Arc::from([0.0, -1.5].as_slice()),
            ],
            rows: 2,
            cols: 2,
        };

        assert_eq!(l, expected_l);
        assert_eq!(u, expected_u);
    }

    #[test]
    fn test_lu_decomposition_non_square_matrix() {
        let matrix: Matrix<i32> = matrix!([2, 3, 1], [4, 7, 3]);

        let result = MatrixUtilities::lu_decomposition(&matrix);
        assert!(result.is_err(), "LU decomposition should fail for a non-square matrix.");
        assert_eq!(
            result.unwrap_err(),
            "Matrix must be square for LU decomposition.".to_string()
        );
    }
}