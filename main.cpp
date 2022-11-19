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

void load_matrix(int* &A, int* &B, int* &C, int &n, int &m, int &p) {
    std::fstream fin;
    fin.open("gen_input.txt", std::ios::in);
    fin >> n >> m >> p;

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

    fin.close();
}

void mpi_strassen(int *A, int *B, int *C, int n, int m, int p) {

}

// int main(int argc, char* argv[]) {
//     // MPI init
//     MPI_Init(&argc, &argv);
//     int comm_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//     if(comm_size != 4) {
//         printf("This application is meant to be run with 4 MPI processes, not %d.\n", comm_size);
//         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//     }
//     // Get rank
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     // OMP init
//     omp_set_dynamic(0);
//     omp_set_num_threads(24);

//     // // MPI template
//     // const int LEN = 10;
//     // int *window_buffer = new int[LEN];
//     // MPI_Win window;
//     // MPI_Win_create(window_buffer, sizeof(int) * LEN, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
//     // MPI_Win_fence(0, window);
//     // if (my_rank == 0) {
//     //     MPI_Put(window_buffer, LEN, MPI_INT, 1, 0, LEN, MPI_INT, window);
//     // }
//     // MPI_Win_fence(0, window);
//     // MPI_Win_free(&window);
//     // MPI_Finalize();

//     // int n = 3, m = 4;
//     // int len = n * m + 2;
//     // int *matrix = new int[len];
//     // matrix[0] = n;
//     // matrix[1] = m;
//     // MPI_Win win;
//     // MPI_Win_create(matrix, len * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
//     // MPI_Win_fence(0, win);
//     int *a, *b, *c;
//     int n, m, p;

//     if (rank == 0) {
//         std::ofstream fout;
//         fout.open("output.txt", std::ofstream::app);

//         load_matrix(a, b, c, n, m, p);

//         // double begin = omp_get_wtime();
//     }

//     mpi_strassen(a, b, c, n, m, p);

//     MPI_Win_fence(0, win);

//     MPI_Win_free(&win);
//     MPI_Finalize();

//     return 0;
// }

int main(int argc, char** argv) {
    std::ofstream fout;
    fout.open ("output.txt", std::ofstream::app);

    int *a, *b, *c;
    int n, m, p;
    load_matrix(a, b, c, n, m, p);

    omp_set_dynamic(0);
    omp_set_num_threads(24);
    
    double begin = omp_get_wtime();

    omp_strassen(a, b, c, n, m, p);

    double end = omp_get_wtime();

    fout << end - begin << '\n'; // still need to improve

    check_correctness(a, b, c, n, m, p);

    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < p; ++j) fout << c[i][j] << ' ';
    //     fout << '\n';
    // }

    fout.close();
}
