mod gaussian_elimination_tests {
    use linalgrs::matrix::Matrix;
    use linalgrs::matrix_utilities::MatrixUtilities;
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
            Arc::from(vec![0.0, 0.0, 0.0]),
        ];

        let result = MatrixUtilities::rref(mat);

        assert_eq!(result.mat, expected_rref);
    }

    #[test]
    fn test_gaussian_elimination_unique_solution() {
        let matrix = Matrix {
            mat: vec![
                Arc::from(vec![2.0, 1.0, -1.0, 8.0]),
                Arc::from(vec![-3.0, -1.0, 2.0, -11.0]),
                Arc::from(vec![-2.0, 1.0, 2.0, -3.0]),
            ],
            rows: 3,
            cols: 4,
        };

        let result = MatrixUtilities::gaussian_elimination(matrix);
        assert!(result.is_ok());
        let pivot_vars = result.unwrap();
        assert_eq!(pivot_vars.get(&'a'), Some(&2.0));
        assert_eq!(pivot_vars.get(&'b'), Some(&3.0));
        assert_eq!(pivot_vars.get(&'c'), Some(&-1.0));
    }
    #[test]
    fn test_gaussian_elimination_no_solution() {
        let matrix = Matrix {
            mat: vec![
                Arc::from(vec![2.0, 1.0, -1.0, 8.0]),
                Arc::from(vec![-3.0, -1.0, 2.0, -11.0]),
                Arc::from(vec![2.0, 1.0, -1.0, 7.0]),
            ],
            rows: 3,
            cols: 4,
        };

        let result = MatrixUtilities::gaussian_elimination(matrix);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("No solution exists for the given matrix.".to_string())
        );
    }
    #[test]
    fn test_gaussian_elimination_infinitely_many_solutions() {
        let matrix = Matrix {
            mat: vec![
                Arc::from(vec![1.0, -1.0, 2.0, 0.0]),
                Arc::from(vec![0.0, 0.0, 0.0, 0.0]),
                Arc::from(vec![0.0, 0.0, 0.0, 0.0]),
            ],
            rows: 3,
            cols: 4,
        };

        let result = MatrixUtilities::gaussian_elimination(matrix);
        assert!(result.is_err());
        assert_eq!(
            result.err(),
            Some("Infinitely many solutions exist for the given matrix.".to_string())
        );
    }
}