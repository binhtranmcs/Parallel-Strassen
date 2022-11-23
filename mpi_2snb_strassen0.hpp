#include <mpi.h>
#include "omp_strassen.hpp"

void mpi_2snb_strassen(int **A, int **B, int **C, int n, int m, int p, int rank) {
    int n1 = (n + 1) / 2, m1 = (m + 1) / 2, p1 = (p + 1) / 2;
    
    //divide matrix A into 4 submatrix
    int **A11 = new_matrix(n1,m1);
    int **A12 = new_matrix(n1,m1);
    int **A21 = new_matrix(n1,m1);
    int **A22 = new_matrix(n1,m1);
    int **B11 = new_matrix(m1,p1);
    int **B12 = new_matrix(m1,p1);
    int **B21 = new_matrix(m1,p1);
    int **B22 = new_matrix(m1,p1);
    if (rank == 0) {
        init_sub_matrix(A,A11,n,m,0,0);
        init_sub_matrix(A,A12,n,m,0,m1);
        init_sub_matrix(A,A21,n,m,n1,0);
        init_sub_matrix(A,A22,n,m,n1,m1);

        init_sub_matrix(B, B11, m, p, 0, 0);
        init_sub_matrix(B, B12, m, p, 0, p1);
        init_sub_matrix(B, B21, m, p, m1, 0);
        init_sub_matrix(B, B22, m, p, m1, p1);
    }

    // submatrix results of strassen
    int **M1 = new_matrix(n1,p1);
    int **M2 = new_matrix(n1, p1);
    int **M3 = new_matrix(n1, p1);
    int **M4 = new_matrix(n1, p1);
    int **M5 = new_matrix(n1, p1);
    int **M6 = new_matrix(n1, p1);
    int **M7 = new_matrix(n1, p1);
    
    int a_sub_size = n1 * p1;
    int b_sub_size = m1 * p1;
    int c_sub_size = n1 * p1;

    //target process
    if (rank == 1) {
        MPI_Request request[4];

        MPI_Irecv(&A21[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&A22[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[1]);
        MPI_Irecv(&B11[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&B21[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[3]);

        for (int i = 0; i < 4; ++i) MPI_Wait(&request[i], MPI_STATUS_IGNORE);

        int **M21 = new_matrix(n1, m1);
        int **M42 = new_matrix(m1, p1);
        // calculate M2
        matrix_add(A21, A22, M21, n1, m1);
        omp_strassen(M21, B11, M2, n1, m1, p1);
        // calculate M4
        matrix_sub(B21, B11, M42, m1, p1);
        omp_strassen(A22, M42, M4, n1, m1, p1);

        MPI_Send(&M2[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&M4[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete_matrix(M21);
        delete_matrix(M42);
    } 

    if (rank == 2) {
        MPI_Request request[5];
        MPI_Irecv(&A11[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&A21[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[1]);
        MPI_Irecv(&B11[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&B12[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[3]);
        MPI_Irecv(&B22[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[4]);   

        for (int i = 0; i < 5; ++i) MPI_Wait(&request[i], MPI_STATUS_IGNORE);

        int **M32 = new_matrix(m1, p1);
        int **M61 = new_matrix(n1, m1);
        int **M62 = new_matrix(m1, p1);

        // calculate M3
        matrix_sub(B12, B22, M32, m1, p1);
        omp_strassen(A11, M32, M3, n1, m1, p1);
        // calcualte M6
        matrix_sub(A21, A11, M61, n1, m1);
        matrix_add(B11, B12, M62, m1, p1);
        omp_strassen(M61, M62, M6, n1, m1, p1);

        MPI_Send(&M3[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&M6[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete_matrix(M32);
        delete_matrix(M61);
        delete_matrix(M62);
    }

    if (rank == 3) {
        MPI_Request request[5];
        MPI_Irecv(&A11[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&A12[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[1]);
        MPI_Irecv(&A22[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&B21[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[3]);
        MPI_Irecv(&B22[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD, &request[4]);

        for (int i = 0; i < 5; ++i) MPI_Wait(&request[i], MPI_STATUS_IGNORE);

        int **M51 = new_matrix(n1, m1);
        int **M71 = new_matrix(n1, m1);
        int **M72 = new_matrix(m1, p1);

        // calculate M5
        matrix_add(A11, A12, M51, n1, m1);
        omp_strassen(M51, B22, M5, n1, m1, p1);
        //calculate M7
        matrix_sub(A12, A22, M71, n1, m1);
        matrix_add(B21, B22, M72, m1, p1);
        omp_strassen(M71, M72, M7, n1, m1, p1);

        MPI_Send(&M5[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&M7[0][0], c_sub_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete_matrix(M51);
        delete_matrix(M72);
        delete_matrix(M71);
    }

    //sending process
    if (rank == 0) {
        MPI_Request request[6];

        MPI_Irecv(&M2[0][0], c_sub_size, MPI_INT, 1, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&M4[0][0], c_sub_size, MPI_INT, 1, 0, MPI_COMM_WORLD, &request[1]);
        MPI_Irecv(&M3[0][0], c_sub_size, MPI_INT, 2, 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&M6[0][0], c_sub_size, MPI_INT, 2, 0, MPI_COMM_WORLD, &request[3]);
        MPI_Irecv(&M5[0][0], c_sub_size, MPI_INT, 3, 0, MPI_COMM_WORLD, &request[4]);
        MPI_Irecv(&M7[0][0], c_sub_size, MPI_INT, 3, 0, MPI_COMM_WORLD, &request[5]);

        // //send to machine 0 to process
        // MPI_Send(&A11[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD);
        // MPI_Send(&A22[0][0],a_sub_size,MPI_INT,0,0,MPI_COMM_WORLD);
        // MPI_Send(&B11[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD);
        // MPI_Send(&B22[0][0],b_sub_size,MPI_INT,0,0,MPI_COMM_WORLD);

        //send to machine 1 to process
        MPI_Send(&A21[0][0],a_sub_size,MPI_INT,1,0,MPI_COMM_WORLD);
        MPI_Send(&A22[0][0],a_sub_size,MPI_INT,1,0,MPI_COMM_WORLD);
        MPI_Send(&B11[0][0],b_sub_size,MPI_INT,1,0,MPI_COMM_WORLD);
        MPI_Send(&B21[0][0],b_sub_size,MPI_INT,1,0,MPI_COMM_WORLD);

        //send to machine 2 to process
        MPI_Send(&A11[0][0],a_sub_size,MPI_INT,2,0,MPI_COMM_WORLD);
        MPI_Send(&A21[0][0],a_sub_size,MPI_INT,2,0,MPI_COMM_WORLD);
        MPI_Send(&B11[0][0],b_sub_size,MPI_INT,2,0,MPI_COMM_WORLD);
        MPI_Send(&B12[0][0],b_sub_size,MPI_INT,2,0,MPI_COMM_WORLD);
        MPI_Send(&B22[0][0],b_sub_size,MPI_INT,2,0,MPI_COMM_WORLD);
        
        //send to machine 3 to process
        MPI_Send(&A11[0][0],a_sub_size,MPI_INT,3,0,MPI_COMM_WORLD);
        MPI_Send(&A12[0][0],a_sub_size,MPI_INT,3,0,MPI_COMM_WORLD);
        MPI_Send(&A22[0][0],a_sub_size,MPI_INT,3,0,MPI_COMM_WORLD);
        MPI_Send(&B21[0][0],b_sub_size,MPI_INT,3,0,MPI_COMM_WORLD);
        MPI_Send(&B22[0][0],b_sub_size,MPI_INT,3,0,MPI_COMM_WORLD);

        int **M11 = new_matrix(n1, m1);
        int **M12 = new_matrix(m1, p1);
        // calculate M1
        matrix_add(A11, A22, M11, n1, m1);
        matrix_add(B11, B22, M12, m1, p1);
        omp_strassen(M11, M12, M1, n1, m1, p1);
        delete_matrix(M11);
        delete_matrix(M12);
 
        for (int i = 0; i < 6; ++i) MPI_Wait(&request[i], MPI_STATUS_IGNORE);

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
}