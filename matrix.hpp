#include <iostream>


/**
 * @brief initialize a matrix of size n * m.
 * The matrix memory layout is a 1D array of size n * m in row-wise order.
 * 
 * @param n 
 * @param m 
 * @return int** 
 */
int** new_matrix(int n, int m) {
    int *arr = new int[n * m];
    std::fill(arr, arr + n * m, 0);
    int** matrix = new int*[n];
    for (int i = 0; i < n; ++i) {
        *(matrix + i) = arr;
        arr += m;
    }
    return matrix;
}

void matrix_print(int** a, int n, int m) {
    std::cout << "Matrix:" << n << ' ' << m << '\n';
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std:: cout << a[i][j] << " \n"[j == m - 1];
        }
    }
}

void delete_matrix(int** matrix) {
    delete[] *matrix;
    delete[] matrix;
}

void matrix_add(int** a, int** b, int** c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) c[i][j] = a[i][j] + b[i][j];
    }
}

void matrix_sub(int** a, int** b, int** c, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) c[i][j] = a[i][j] - b[i][j];
    }
}

void matrix_multiply(int** a, int** b, int** c, int n, int m, int p) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < m; ++k) {
            for (int j = 0; j < p; ++j) c[i][j] += a[i][k] * b[k][j];
        }
    }
}

void init_sub_matrix(int** a, int** a1, int n, int m, int sr, int sc) {
    int n1 = (n + 1) / 2, m1 = (m + 1) / 2;
    for (int i = sr; i < std::min(sr + n1, n); ++i) {
        for (int j = sc; j < std::min(sc + m1, m); ++j) {
            a1[i - sr][j - sc] = a[i][j];
        }
    }
}
