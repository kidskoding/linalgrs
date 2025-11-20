mod determinant_tests {
    use linalgrs::matrix::Matrix;
    use linalgrs::matrix_utilities::MatrixUtilities;
    use std::sync::Arc;

    #[test]
    fn test_determinant_1x1() {
        let mut matrix = Matrix {
            mat: vec![Arc::new([1])],
            rows: 1,
            cols: 1,
        };

        assert_eq!(MatrixUtilities::determinant(&mut matrix).unwrap(), 1);
    }

    #[test]
    fn test_determinant_2x2() {
        let mut matrix = Matrix {
            mat: vec![Arc::new([1, 2]), Arc::new([3, 4])],
            rows: 2,
            cols: 2,
        };

        assert_eq!(MatrixUtilities::determinant(&mut matrix).unwrap(), -2);
    }

    #[test]
    fn test_determinant_3x3() {
        let mut matrix = Matrix {
            mat: vec![
                Arc::new([1, 2, 3]),
                Arc::new([0, 1, 4]),
                Arc::new([5, 6, 0]),
            ],
            rows: 3,
            cols: 3,
        };

        let result = MatrixUtilities::determinant(&mut matrix);
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_determinant_4x4() {
        let mut matrix = Matrix {
            mat: vec![
                Arc::new([1, 0, 2, -1]),
                Arc::new([3, 0, 0, 5]),
                Arc::new([2, 1, 4, -3]),
                Arc::new([1, 0, 5, 0]),
            ],
            rows: 4,
            cols: 4,
        };
        assert_eq!(MatrixUtilities::determinant(&mut matrix).unwrap(), 30);
    }

    #[test]
    fn test_non_square_matrix() {
        let mut matrix = Matrix {
            mat: vec![Arc::new([1, 2, 3]), Arc::new([4, 5, 6])],
            rows: 2,
            cols: 3,
        };

        let result = MatrixUtilities::determinant(&mut matrix);
        assert_eq!(result, None);
    }
}
