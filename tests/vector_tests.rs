mod vector_operations_test {
    use std::sync::Arc;

    use linalgrs::vector;
    use linalgrs::vector::Vector;

    #[test]
    fn test_vector() {
        let vector = Vector::new(&[1, 2, 3]);
        let vec = Arc::new([1, 2, 3]);

        assert_eq!(vector.data.as_ref(), vec.as_ref())
    }

    #[test]
    fn test_vector_display() {
        let vec = vector![1, 2, 3];
        assert_eq!(format!("{}", vec), "| 1 2 3 |");
    }

    #[test]
    fn test_debug_derive() {
        let vec = vector![1, 2, 3];
        dbg!(vec);
    }
}
