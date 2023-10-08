// Copyright 2023 Pigasin Daniil
#include <omp.h>
#include <gtest/gtest.h>
#include <vector>
#include "./matrix_mult.h"

TEST(Matrix_Multiplication_OMP, get_works) {
    SparseMatrix matrix(3, 3);
    EXPECT_NO_THROW(matrix.get(1, 1));
}

TEST(Matrix_Multiplication_OMP, get_throws_exception) {
    SparseMatrix matrix(3, 3);
    EXPECT_ANY_THROW(matrix.get(4, 4));
}

TEST(Matrix_Multiplication_OMP, multiplication_works) {
    SparseMatrix matrix1(3, 5);
    SparseMatrix matrix2(5, 6);
    EXPECT_NO_THROW(matrix1.multiply_seq(matrix2));
}

TEST(Matrix_Multiplication_OMP, multiplication_throws_exception) {
    SparseMatrix matrix1(3, 6);
    SparseMatrix matrix2(5, 6);
    EXPECT_ANY_THROW(matrix1.multiply_seq(matrix2));
}

TEST(Matrix_Multiplication_OMP, result_matrix_has_correct_size) {
    SparseMatrix matrix1(3, 6);
    SparseMatrix matrix2(6, 10);
    SparseMatrix result = matrix1.multiply_seq(matrix2);
    EXPECT_EQ(result.getM(), matrix1.getM());
    EXPECT_EQ(result.getN(), matrix2.getN());
}

TEST(Matrix_Multiplication_OMP, parallel_multiplication_returns_same_result) {
    SparseMatrix matrix1(3, 6);
    SparseMatrix matrix2(6, 10);
    SparseMatrix result1 = matrix1.multiply_parallel(matrix2);
    SparseMatrix result2 = matrix1.multiply_seq(matrix2);
    ASSERT_TRUE(result1 == result2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
