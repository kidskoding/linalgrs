mod matrix_tests {
    use linalgrs::matrix;
    use linalgrs::matrix::Matrix;
    use linalgrs::matrix_utilities::MatrixUtilities;
    use std::sync::Arc;

    #[test]
    fn test_matrix() {
        let mut mat = Matrix::default();
        let arr = [1, 2, 3];
        mat = MatrixUtilities::append(mat, &arr);

        let expected: Vec<Arc<[i64]>> = vec![Arc::new(arr)];
        assert_eq!(mat.mat, expected);
    }

    #[test]
    fn test_matrix_macro() {
        let mat = matrix!([1, 2, 3, 4], [5, 6, 7, 8]);
        let mat2 = Matrix::default();
        let mat_expected = MatrixUtilities::append_multiple(mat2, &[&[1, 2, 3, 4], &[5, 6, 7, 8]]);
        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_shape() {
        let mut mat = Matrix::default();
        let arr = [1, 2, 3];
        mat = MatrixUtilities::append(mat, &arr);
        assert_eq!(mat.shape(), (1, 3));
    }

    #[test]
    fn test_identity() {
        let expected = Matrix {
            mat: vec![
                Arc::from([1, 0, 0]),
                Arc::from([0, 1, 0]),
                Arc::from([0, 0, 1]),
            ],
            cols: 3,
            rows: 3,
        };

        let eye = MatrixUtilities::identity(3);
        assert_eq!(eye.mat, expected.mat);
        assert_eq!(eye.cols, expected.cols);
        assert_eq!(eye.rows, expected.rows);
    }

    #[test]
    fn test_sub_matrix() {
        let mut mat = Matrix::default();
        let arr: &[&[i64]] = &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]];

        mat = MatrixUtilities::append_multiple(mat, arr);

        let mat = mat.sub_matrix(0..2, 0..2);
        assert!(mat.is_ok());

        let sub_mat = mat.unwrap();
        let expected: Vec<Arc<[i64]>> =
            vec![Arc::from(&[1, 2][..]), Arc::from(&[4, 5][..])];
        assert_eq!(sub_mat.mat, expected);
    }

    #[test]
    fn test_dbg_derive() {
        let mat = MatrixUtilities::<i32>::identity(3usize);
        dbg!(mat);
    }

    #[test]
    fn test_display_derive() {
        let mat = MatrixUtilities::<i32>::identity(3usize);
        println!("{}", mat);
    }
}
