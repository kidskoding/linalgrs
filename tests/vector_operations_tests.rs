mod vector_operations_tests {
    use std::collections::HashSet;

    use linalgrs::vector;
    use linalgrs::vector_utilities::VectorUtilities;

    #[test]
    fn test_linear_independence() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![3, 6, 8];
    
        let set = HashSet::from([vec1, vec2]);
        assert_eq!(VectorUtilities::is_linear_independent(&set), true);
    }

    #[test]
    fn test_linear_dependence() {
        let vec1 = vector![1, 2, 3];
        let vec2 = vector![3, 6, 9];

        let set = HashSet::from([vec1, vec2]);
        assert_eq!(VectorUtilities::is_linear_independent(&set), false);
    }
}
