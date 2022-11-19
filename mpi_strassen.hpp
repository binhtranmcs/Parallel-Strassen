#include <mpi.h>

#include "omp_strassen.hpp"


void mpi_strassen(int **A, int **B, int **C, int n, int m, int p, int rank) {
    int n1 = (n + 1) / 2, m1 = (m + 1) / 2, p1 = (p + 1) / 2;


    // divide matrix A into 4 submatrix
    int **A11 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A11, n, m, 0, 0);
    MPI_Win win_A11;
    MPI_Win_create(&A11[0][0], sizeof(int) * (n1 * m1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A11);

    int **A12 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A12, n, m, 0, m1);
    MPI_Win win_A12;
    MPI_Win_create(&A12[0][0], sizeof(int) * (n1 * m1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A12);

    int **A21 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A21, n, m, n1, 0);
    MPI_Win win_A21;
    MPI_Win_create(&A21[0][0], sizeof(int) * (n1 * m1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A21);

    int **A22 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A22, n, m, n1, m1);
    MPI_Win win_A22;
    MPI_Win_create(&A22[0][0], sizeof(int) * (n1 * m1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A22);


    // divide matrix A into 4 submatrix
    int **B11 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B11, m, p, 0, 0);
    MPI_Win win_B11;
    MPI_Win_create(&B11[0][0], sizeof(int) * (m1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B11);

    int **B12 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B12, m, p, 0, p1);
    MPI_Win win_B12;
    MPI_Win_create(&B12[0][0], sizeof(int) * (m1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B12);

    int **B21 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B21, m, p, m1, 0);
    MPI_Win win_B21;
    MPI_Win_create(&B21[0][0], sizeof(int) * (m1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B21);

    int **B22 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B22, m, p, m1, p1);
    MPI_Win win_B22;
    MPI_Win_create(&B22[0][0], sizeof(int) * (m1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B22);


    // 7 submatrix results after running strassen
    int **M1 = new_matrix(n1, p1);
    MPI_Win win_M1;
    MPI_Win_create(&M1[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M1);
    
    int **M2 = new_matrix(n1, p1);
    MPI_Win win_M2;
    MPI_Win_create(&M2[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M2);
    
    int **M3 = new_matrix(n1, p1);
    MPI_Win win_M3;
    MPI_Win_create(&M3[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M3);
    
    int **M4 = new_matrix(n1, p1);
    MPI_Win win_M4;
    MPI_Win_create(&M4[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M4);
    
    int **M5 = new_matrix(n1, p1);
    MPI_Win win_M5;
    MPI_Win_create(&M5[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M5);
    
    int **M6 = new_matrix(n1, p1);
    MPI_Win win_M6;
    MPI_Win_create(&M6[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M6);
    
    int **M7 = new_matrix(n1, p1);
    MPI_Win win_M7;
    MPI_Win_create(&M7[0][0], sizeof(int) * (n1 * p1), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M7);


    // sending matrix from machine 0 to other machines
    MPI_Win_fence(0, win_A11);
    MPI_Win_fence(0, win_A12);
    MPI_Win_fence(0, win_A21);
    MPI_Win_fence(0, win_A22);
    MPI_Win_fence(0, win_B11);
    MPI_Win_fence(0, win_B12);
    MPI_Win_fence(0, win_B21);
    MPI_Win_fence(0, win_B22);
    if (rank == 0) {
        int a_sub_size = n1 * m1;
        int b_sub_size = m1 * p1;
        
        // send to machine 0 to process
        MPI_Put(&A11[0][0], a_sub_size, MPI_INT, 0, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(&A22[0][0], a_sub_size, MPI_INT, 0, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(&B11[0][0], b_sub_size, MPI_INT, 0, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(&B22[0][0], b_sub_size, MPI_INT, 0, 0, b_sub_size, MPI_INT, win_B22);

        // send to machine 1 to process
        MPI_Put(&A21[0][0], a_sub_size, MPI_INT, 1, 0, a_sub_size, MPI_INT, win_A21);
        MPI_Put(&A22[0][0], a_sub_size, MPI_INT, 1, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(&B11[0][0], b_sub_size, MPI_INT, 1, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(&B21[0][0], b_sub_size, MPI_INT, 1, 0, b_sub_size, MPI_INT, win_B21);

        // send to machine 2 to process
        MPI_Put(&A11[0][0], a_sub_size, MPI_INT, 2, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(&A21[0][0], a_sub_size, MPI_INT, 2, 0, a_sub_size, MPI_INT, win_A21);
        MPI_Put(&B11[0][0], b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(&B12[0][0], b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B12);
        MPI_Put(&B22[0][0], b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B22);

        // send to machine 3 to process
        MPI_Put(&A11[0][0], a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(&A12[0][0], a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A12);
        MPI_Put(&A22[0][0], a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(&B21[0][0], b_sub_size, MPI_INT, 3, 0, b_sub_size, MPI_INT, win_B21);
        MPI_Put(&B22[0][0], b_sub_size, MPI_INT, 3, 0, b_sub_size, MPI_INT, win_B22);
    }
    MPI_Win_fence(0, win_A11);
    MPI_Win_fence(0, win_A12);
    MPI_Win_fence(0, win_A21);
    MPI_Win_fence(0, win_A22);
    MPI_Win_fence(0, win_B11);
    MPI_Win_fence(0, win_B12);
    MPI_Win_fence(0, win_B21);
    MPI_Win_fence(0, win_B22);  


    // each machine calculate and send back result to machine 0
    MPI_Win_fence(0, win_M1);
    MPI_Win_fence(0, win_M2);
    MPI_Win_fence(0, win_M3);
    MPI_Win_fence(0, win_M4);
    MPI_Win_fence(0, win_M5);
    MPI_Win_fence(0, win_M6);
    MPI_Win_fence(0, win_M7);
    int c_sub_size = n1 * p1;
    if (rank == 0) {
        int **M11 = new_matrix(n1, m1);
        int **M12 = new_matrix(m1, p1);
        
        // calculate M1
        matrix_add(A11, A22, M11, n1, m1);
        matrix_add(B11, B22, M12, m1, p1);
        omp_strassen(M11, M12, M1, n1, m1, p1);
        MPI_Put(&M1[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M1);

        delete_matrix(M11);
        delete_matrix(M12);
    }
    if (rank == 1) {
        int **M21 = new_matrix(n1, m1);
        int **M42 = new_matrix(m1, p1);
        
        // calculate M2
        matrix_add(A21, A22, M21, n1, m1);
        omp_strassen(M21, B11, M2, n1, m1, p1);
        MPI_Put(&M2[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M2);
        // calculate M4
        matrix_sub(B21, B11, M42, m1, p1);
        omp_strassen(A22, M42, M4, n1, m1, p1);
        MPI_Put(&M4[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M4);

        delete_matrix(M21);
        delete_matrix(M42);
    }
    if (rank == 2) {
        int **M32 = new_matrix(m1, p1);
        int **M61 = new_matrix(n1, m1);
        int **M62 = new_matrix(m1, p1);
        
        // calculate M3
        matrix_sub(B12, B22, M32, m1, p1);
        omp_strassen(A11, M32, M3, n1, m1, p1);
        MPI_Put(&M3[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M3);
        // calcualte M6
        matrix_sub(A21, A11, M61, n1, m1);
        matrix_add(B11, B12, M62, m1, p1);
        omp_strassen(M61, M62, M6, n1, m1, p1);
        MPI_Put(&M6[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M6);

        delete_matrix(M32);
        delete_matrix(M61);
        delete_matrix(M62);
    }
    if (rank == 3) {
        int **M51 = new_matrix(n1, m1);
        int **M71 = new_matrix(n1, m1);
        int **M72 = new_matrix(m1, p1);
        
        // calculate M5
        matrix_add(A11, A12, M51, n1, m1);
        omp_strassen(M51, B22, M5, n1, m1, p1);
        MPI_Put(&M5[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M5);
        //calculate M7
        matrix_sub(A12, A22, M71, n1, m1);
        matrix_add(B21, B22, M72, m1, p1);
        omp_strassen(M71, M72, M7, n1, m1, p1);
        MPI_Put(&M7[0][0], c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M7);

        delete_matrix(M51);
        delete_matrix(M72);
        delete_matrix(M71);
    }
    MPI_Win_fence(0, win_M1);
    MPI_Win_fence(0, win_M2);
    MPI_Win_fence(0, win_M3);
    MPI_Win_fence(0, win_M4);
    MPI_Win_fence(0, win_M5);
    MPI_Win_fence(0, win_M6);
    MPI_Win_fence(0, win_M7);

    // merge result
    if (rank == 0) {
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < p1; ++j) {
                C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                if (p1 + j < p) C[i][p1 + j] = M3[i][j] + M5[i][j];
                if (n1 + i < n) {
                    C[n1 + i][j] = M2[i][j] + M4[i][j];
                    if (p1 + j < p) {
                        C[n1 + i][p1 + j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
                    }
                }
            }
        }
    }

    delete_matrix(A11);
    delete_matrix(A12);
    delete_matrix(A21);
    delete_matrix(A22);

    delete_matrix(B11);
    delete_matrix(B12);
    delete_matrix(B21);
    delete_matrix(B22);

    delete_matrix(M1);
    delete_matrix(M2);
    delete_matrix(M3);
    delete_matrix(M4);
    delete_matrix(M5);
    delete_matrix(M6);
    delete_matrix(M7);

    // free windows
    MPI_Win_free(&win_A11);
    MPI_Win_free(&win_A12);
    MPI_Win_free(&win_A21);
    MPI_Win_free(&win_A22);

    MPI_Win_free(&win_B11);
    MPI_Win_free(&win_B12);
    MPI_Win_free(&win_B21);
    MPI_Win_free(&win_B22);

    MPI_Win_free(&win_M1);
    MPI_Win_free(&win_M2);
    MPI_Win_free(&win_M3);
    MPI_Win_free(&win_M4);
    MPI_Win_free(&win_M5);
    MPI_Win_free(&win_M6);
    MPI_Win_free(&win_M7);
}
