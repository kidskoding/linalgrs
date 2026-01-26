mod vector_operations_tests {
    use std::collections::HashSet;

    use linalgrs::vector;
    use linalgrs::vector_utilities::VectorUtilities;

    #[test]
    fn test_linear_independence() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![3, 6, 8];
    
        let set = HashSet::from([vec1, vec2]);
        let result = VectorUtilities::is_linear_independent(&set);
        
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn test_linear_dependence() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![3, 6, 9];

        let set = HashSet::from([vec1, vec2]);
        let result = VectorUtilities::is_linear_independent(&set);
        
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap(), false);
    }

    #[test]
    fn test_invalid_linear_indepdences() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![3, 6, 9, 12];

        let set = HashSet::from([vec1, vec2]);
        let result = VectorUtilities::is_linear_independent(&set);
        
        assert_eq!(result.is_err(), true);
    }

    #[test]
    fn test_inner_product() {
        let u = vector![1, 2, 3];
        let v = vector![4, 5, 6];

        let result = VectorUtilities::<i32>::inner_product(&u, &v);
        
        assert_eq!(result.is_ok(), true);
        assert_eq!(result.unwrap(), 32);
    }

    #[test]
    fn test_invalid_inner_product() {
        let u = vector![1, 2, 3];
        let v = vector![1, 2, 3, 4];

        let result = VectorUtilities::<i32>::inner_product(&u, &v);

        assert_eq!(result.is_err(), true);
    }
}
