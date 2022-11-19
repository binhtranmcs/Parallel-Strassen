#include <iostream>


/**
 * @brief initialize a matrix of size n * m.
 * The matrix is actually a 1D array with size 2 + n * m.
 * The first 2 elements are the matrix size, which are n and m respectively.
 * The remaining n * m elements are elements of the matrix in row-wise order.
 * 
 * @param n 
 * @param m 
 * @return int* 
 */
int* new_matrix(int n, int m) {
    int *A = new int[n * m + 2];
    std::fill(A, A + n * m + 2, 0);
    A[0] = n;
    A[1] = m;
    return A;
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

void matrix_print(int* A, int n, int m) {
    printf("Matrix size: %d %d\n", n, m);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            std::cout << get(A, r, c) << " \n"[c == m - 1];
        }
    }
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
