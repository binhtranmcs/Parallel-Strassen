#include "bits/stdc++.h"
#include <omp.h>
#include <mpi.h>

const int THRESHOLD = 64;

void matrix_print(int* A, int n, int m) {
    printf("Matrix size: %d %d\n", n, m);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            std:: cout << A[2 + r * m + c] << " \n"[c == m - 1];
        }
    }
}

inline int get(int *A, int r, int c) {
    return A[2 + r * A[1] + c];
}

inline bool set(int *A, int r, int c, int val) {
    int n = A[0], m = A[1];
    if (r >= n || r < 0 || c >= m || c < 0) [[unlikely]] return false;
    std::swap(A[2 + r * m + c], val);
    return true;
}

inline bool add(int *A, int r, int c, int val) {
    int n = A[0], m = A[1];
    if (r >= n || r < 0 || c >= m || c < 0) [[unlikely]] return false;
    A[2 + r * m + c] += val;
    return true;
}

int* new_matrix(int n, int m) {
    int *A = new int[n * m + 2];
    std::fill(A, A + n * m + 2, 0);
    A[0] = n;
    A[1] = m;
    return A;
}

void delete_matrix(int *A) {
    delete[] A;
}

void matrix_add(int *A, int *B, int *C, int n, int m) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            set(C, r, c, get(A, r, c) + get(B, r, c));
        }
    }
}

void matrix_sub(int *A, int *B, int *C, int n, int m) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            set(C, r, c, get(A, r, c) - get(B, r, c));
        }
    }
}

void matrix_multiply(int *A, int *B, int *C, int n, int m, int p) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k) {
            for (int j = 0; j < p; ++j) {
                add(C, i, j, get(A, i, k) * get(B, k, j));
            }
        }
    }
}

void init_sub_matrix(int *A, int *A1, int n, int m, int sr, int sc) {
    int n1 = (n + 1) / 2, m1 = (m + 1) / 2;
    for (int i = sr; i < std::min(sr + n1, n); ++i) {
        for (int j = sc; j < std::min(sc + m1, m); ++j) {
            // std::cerr << i << ' ' << j << a[i][j] << '\n';
            set(A1, i - sr, j - sc, get(A, i, j));
        }
    }
}

void omp_strassen(int *A, int *B, int *C, int n, int m, int p) {
    if (std::max(n, std::max(m, p)) <= THRESHOLD) {
        matrix_multiply(A, B, C, n, m, p);
        return;
    }

    int n1 = (n + 1) / 2, m1 = (m + 1) / 2, p1 = (p + 1) / 2;

    int *A11 = new_matrix(n1, m1);
    init_sub_matrix(A, A11, n, m, 0, 0);
    int *A12 = new_matrix(n1, m1);
    init_sub_matrix(A, A12, n, m, 0, m1);
    int *A21 = new_matrix(n1, m1);
    init_sub_matrix(A, A21, n, m, n1, 0);
    int *A22 = new_matrix(n1, m1);
    init_sub_matrix(A, A22, n, m, n1, m1);

    int *B11 = new_matrix(m1, p1);
    init_sub_matrix(B, B11, m, p, 0, 0);
    int *B12 = new_matrix(m1, p1);
    init_sub_matrix(B, B12, m, p, 0, p1);
    int *B21 = new_matrix(m1, p1);
    init_sub_matrix(B, B21, m, p, m1, 0);
    int *B22 = new_matrix(m1, p1);
    init_sub_matrix(B, B22, m, p, m1, p1);

    int *M11 = new_matrix(n1, m1);
    int *M12 = new_matrix(m1, p1);
    int *M21 = new_matrix(n1, m1);
    int *M32 = new_matrix(m1, p1);
    int *M42 = new_matrix(m1, p1);
    int *M51 = new_matrix(n1, m1);
    int *M61 = new_matrix(n1, m1);
    int *M62 = new_matrix(m1, p1);
    int *M71 = new_matrix(n1, m1);
    int *M72 = new_matrix(m1, p1);

    int *M1 = new_matrix(n1, p1);
    int *M2 = new_matrix(n1, p1);
    int *M3 = new_matrix(n1, p1);
    int *M4 = new_matrix(n1, p1);
    int *M5 = new_matrix(n1, p1);
    int *M6 = new_matrix(n1, p1);
    int *M7 = new_matrix(n1, p1);

#pragma omp parallel
{
#pragma omp single
{
#pragma omp task untied
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_add(A11, A22, M11, n1, m1);
    matrix_add(B11, B22, M12, m1, p1);
    omp_strassen(M11, M12, M1, n1, m1, p1);
}
#pragma omp task untied
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_add(A21, A22, M21, n1, m1);
    omp_strassen(M21, B11, M2, n1, m1, p1);
}
#pragma omp task untied
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_sub(B12, B22, M32, m1, p1);
    omp_strassen(A11, M32, M3, n1, m1, p1);
}
#pragma omp task
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_sub(B21, B11, M42, m1, p1);
    omp_strassen(A22, M42, M4, n1, m1, p1);
}
#pragma omp task
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_add(A11, A12, M51, n1, m1);
    omp_strassen(M51, B22, M5, n1, m1, p1);
}
#pragma omp task
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_sub(A21, A11, M61, n1, m1);
    matrix_add(B11, B12, M62, m1, p1);
    omp_strassen(M61, M62, M6, n1, m1, p1);
}
#pragma omp task
{
    // printf("thread num: %d out of %d\n", omp_get_thread_num(), omp_get_max_threads());
    matrix_sub(A12, A22, M71, n1, m1);
    matrix_add(B21, B22, M72, m1, p1);
    omp_strassen(M71, M72, M7, n1, m1, p1);
}
#pragma omp taskwait
} // omp single
} // omp parallel

    // matrix_print(A, n, m);
    // matrix_print(B, m, p);
    // matrix_print(M1, n1, p1);
    // matrix_print(M2, n1, p1);
    // matrix_print(M3, n1, p1);
    // matrix_print(M4, n1, p1);
    // matrix_print(M5, n1, p1);
    // matrix_print(M6, n1, p1);
    // matrix_print(M7, n1, p1);
    // std::cout << '\n';

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < p1; ++j) {
            int tmp = get(M1, i, j) + get(M4, i, j) - get(M5, i, j) + get(M7, i, j);
            set(C, i, j, tmp);
            if (p1 + j < p) set(C, i, p1 + j, get(M3, i, j) + get(M5, i, j));
            if (n1 + i < n) {
                set(C, n1 + i, j, get(M2, i, j) + get(M4, i, j));
                if (p1 + j < p) {
                    int tmp = get(M1, i, j) - get(M2, i, j) + get(M3, i, j) + get(M6, i, j);
                    set(C, n1 + i, p1 + j, tmp);
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

    delete_matrix(M11);
    delete_matrix(M12);
    delete_matrix(M21);
    delete_matrix(M32);
    delete_matrix(M42);
    delete_matrix(M51);
    delete_matrix(M61);
    delete_matrix(M62);
    delete_matrix(M71);
    delete_matrix(M72);

    delete_matrix(M1);
    delete_matrix(M2);
    delete_matrix(M3);
    delete_matrix(M4);
    delete_matrix(M5);
    delete_matrix(M6);
    delete_matrix(M7);
}

void check_correctness(int* A, int* B, int* C, int n, int m, int p) {
    int* E = new_matrix(n, p);
    matrix_multiply(A, B, E, n, m, p);
    for (int r = 0; r < n; ++r) {
        for (int c = p; c < p; ++c) {
            if (get(C, r, c) != get(E, r, c)) {
                std::cout << "INCORRECT\n";
                std::cout << r << ' ' << c << '\n';
                std::cout << get(C, r, c) << ' ' << get(E, r, c) << '\n';
            }
        }
    }
    std::cout << "CORRECT\n";
}

void load_matrix(int* &A, int* &B, int* &C, int &n, int &m, int &p, int rank) {
    std::fstream fin;
    fin.open("gen_input.txt", std::ios::in);
    fin >> n >> m >> p;

    if (rank == 0) {
        A = new_matrix(n, m);
        B = new_matrix(m, p);
        C = new_matrix(n, p);

        int tmp;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < m; ++c) {
                fin >> tmp;
                set(A, r, c, tmp);
            }
        }
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < p; ++c) {
                fin >> tmp;
                set(B, r, c, tmp);
            }
        }
    }

    fin.close();
}

void mpi_strassen(int *A, int *B, int *C, int n, int m, int p, int rank) {
    int n1 = (n + 1) / 2, m1 = (m + 1) / 2, p1 = (p + 1) / 2;


    // divide matrix A into 4 submatrix
    int *A11 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A11, n, m, 0, 0);
    MPI_Win win_A11;
    MPI_Win_create(A11, sizeof(int) * (n1 * m1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A11);

    int *A12 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A12, n, m, 0, m1);
    MPI_Win win_A12;
    MPI_Win_create(A12, sizeof(int) * (n1 * m1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A12);

    int *A21 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A21, n, m, n1, 0);
    MPI_Win win_A21;
    MPI_Win_create(A21, sizeof(int) * (n1 * m1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A21);

    int *A22 = new_matrix(n1, m1);
    if (rank == 0) init_sub_matrix(A, A22, n, m, n1, m1);
    MPI_Win win_A22;
    MPI_Win_create(A22, sizeof(int) * (n1 * m1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A22);


    // divide matrix A into 4 submatrix
    int *B11 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B11, m, p, 0, 0);
    MPI_Win win_B11;
    MPI_Win_create(B11, sizeof(int) * (m1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B11);

    int *B12 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B12, m, p, 0, p1);
    MPI_Win win_B12;
    MPI_Win_create(B12, sizeof(int) * (m1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B12);

    int *B21 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B21, m, p, m1, 0);
    MPI_Win win_B21;
    MPI_Win_create(B21, sizeof(int) * (m1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B21);

    int *B22 = new_matrix(m1, p1);
    if (rank == 0) init_sub_matrix(B, B22, m, p, m1, p1);
    MPI_Win win_B22;
    MPI_Win_create(B22, sizeof(int) * (m1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B22);


    // 7 submatrix result after running strassen
    int *M1 = new_matrix(n1, p1);
    MPI_Win win_M1;
    MPI_Win_create(M1, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M1);
    
    int *M2 = new_matrix(n1, p1);
    MPI_Win win_M2;
    MPI_Win_create(M2, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M2);
    
    int *M3 = new_matrix(n1, p1);
    MPI_Win win_M3;
    MPI_Win_create(M3, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M3);
    
    int *M4 = new_matrix(n1, p1);
    MPI_Win win_M4;
    MPI_Win_create(M4, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M4);
    
    int *M5 = new_matrix(n1, p1);
    MPI_Win win_M5;
    MPI_Win_create(M5, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M5);
    
    int *M6 = new_matrix(n1, p1);
    MPI_Win win_M6;
    MPI_Win_create(M6, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M6);
    
    int *M7 = new_matrix(n1, p1);
    MPI_Win win_M7;
    MPI_Win_create(M7, sizeof(int) * (n1 * p1 + 2), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_M7);


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
        int a_sub_size = n1 * m1 + 2;
        int b_sub_size = m1 * p1 + 2;
        
        // send to machine 0 to process
        MPI_Put(A11, a_sub_size, MPI_INT, 0, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(A22, a_sub_size, MPI_INT, 0, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(B11, b_sub_size, MPI_INT, 0, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(B22, b_sub_size, MPI_INT, 0, 0, b_sub_size, MPI_INT, win_B22);

        // send to machine 1 to process
        MPI_Put(A21, a_sub_size, MPI_INT, 1, 0, a_sub_size, MPI_INT, win_A21);
        MPI_Put(A22, a_sub_size, MPI_INT, 1, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(B11, b_sub_size, MPI_INT, 1, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(B21, b_sub_size, MPI_INT, 1, 0, b_sub_size, MPI_INT, win_B21);

        // send to machine 2 to process
        MPI_Put(A11, a_sub_size, MPI_INT, 2, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(A21, a_sub_size, MPI_INT, 2, 0, a_sub_size, MPI_INT, win_A21);
        MPI_Put(B11, b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B11);
        MPI_Put(B12, b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B12);
        MPI_Put(B22, b_sub_size, MPI_INT, 2, 0, b_sub_size, MPI_INT, win_B22);

        // send to machine 3 to process
        MPI_Put(A11, a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A11);
        MPI_Put(A12, a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A12);
        MPI_Put(A22, a_sub_size, MPI_INT, 3, 0, a_sub_size, MPI_INT, win_A22);
        MPI_Put(B21, b_sub_size, MPI_INT, 3, 0, b_sub_size, MPI_INT, win_B21);
        MPI_Put(B22, b_sub_size, MPI_INT, 3, 0, b_sub_size, MPI_INT, win_B22);
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
    int c_sub_size = n1 * p1 + 2;
    if (rank == 0) {
        int *M11 = new_matrix(n1, m1);
        int *M12 = new_matrix(m1, p1);
        
        // calculate M1
        matrix_add(A11, A22, M11, n1, m1);
        matrix_add(B11, B22, M12, m1, p1);
        omp_strassen(M11, M12, M1, n1, m1, p1);
        MPI_Put(M1, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M1);

        delete_matrix(M11);
        delete_matrix(M12);
    }
    if (rank == 1) {
        int *M21 = new_matrix(n1, m1);
        int *M42 = new_matrix(m1, p1);
        
        // calculate M2
        matrix_add(A21, A22, M21, n1, m1);
        omp_strassen(M21, B11, M2, n1, m1, p1);
        MPI_Put(M2, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M2);
        // calculate M4
        matrix_sub(B21, B11, M42, m1, p1);
        omp_strassen(A22, M42, M4, n1, m1, p1);
        MPI_Put(M4, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M4);

        delete_matrix(M21);
        delete_matrix(M42);
    }
    if (rank == 2) {
        int *M32 = new_matrix(m1, p1);
        int *M61 = new_matrix(n1, m1);
        int *M62 = new_matrix(m1, p1);
        
        // calculate M3
        matrix_sub(B12, B22, M32, m1, p1);
        omp_strassen(A11, M32, M3, n1, m1, p1);
        MPI_Put(M3, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M3);
        // calcualte M6
        matrix_sub(A21, A11, M61, n1, m1);
        matrix_add(B11, B12, M62, m1, p1);
        omp_strassen(M61, M62, M6, n1, m1, p1);
        MPI_Put(M6, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M6);

        delete_matrix(M32);
        delete_matrix(M61);
        delete_matrix(M62);
    }
    if (rank == 3) {
        int *M51 = new_matrix(n1, m1);
        int *M71 = new_matrix(n1, m1);
        int *M72 = new_matrix(m1, p1);
        
        // calculate M5
        matrix_add(A11, A12, M51, n1, m1);
        omp_strassen(M51, B22, M5, n1, m1, p1);
        MPI_Put(M5, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M5);
        //calculate M7
        matrix_sub(A12, A22, M71, n1, m1);
        matrix_add(B21, B22, M72, m1, p1);
        omp_strassen(M71, M72, M7, n1, m1, p1);
        MPI_Put(M7, c_sub_size, MPI_INT, 0, 0, c_sub_size, MPI_INT, win_M7);

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
                int tmp = get(M1, i, j) + get(M4, i, j) - get(M5, i, j) + get(M7, i, j);
                set(C, i, j, tmp);
                if (p1 + j < p) set(C, i, p1 + j, get(M3, i, j) + get(M5, i, j));
                if (n1 + i < n) {
                    set(C, n1 + i, j, get(M2, i, j) + get(M4, i, j));
                    if (p1 + j < p) {
                        int tmp = get(M1, i, j) - get(M2, i, j) + get(M3, i, j) + get(M6, i, j);
                        set(C, n1 + i, p1 + j, tmp);
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

int main(int argc, char* argv[]) {
    // MPI init
    MPI_Init(&argc, &argv);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 4) {
        printf("This application is meant to be run with 4 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // get rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // OMP init
    omp_set_dynamic(0);
    omp_set_num_threads(24);

    // init input matrix
    int *A, *B, *C;
    int n, m, p;
    load_matrix(A, B, C, n, m, p, rank);

    // matrix multiplication with strassen algorithm
    double begin = MPI_Wtime();
    mpi_strassen(A, B, C, n, m, p, rank);
    double end = MPI_Wtime();

    if (rank == 0) {
        check_correctness(A, B, C, n, m, p);
        std::ofstream fout;
        fout.open("output.txt", std::ofstream::app);
        fout << "mpi_omp " << n << ' ' << m << ' ' << p << ": " << end - begin << '\n';
        fout.close();
    }

    MPI_Finalize();

    return 0;
}
