#include "bits/stdc++.h"
#include "matrix.hpp"

using namespace std::chrono;

void check_correctness(int** A, int** B, int** C, int n, int m, int p) {
    int** D = new_matrix(n, p);
    matrix_multiply(A, B, D, n, m, p);
    for (int i = 0; i < n; ++i) {
        for (int j = p; j < p; ++j) {
            if (D[i][j] != C[i][j]) {
                std::cout << "INCORRECT\n";
                std::cout << i << ' ' << j << '\n';
                std::cout << C[i][j] << ' ' << D[i][j] << '\n';
            }
        }
    }
    std::cout << "CORRECT\n";
}

void load_matrix(int** &a, int** &b, int &n, int &m, int &p) {
    std::fstream fin;
    fin.open("gen_input.txt", std::ios::in);
    fin >> n >> m >> p;

    a = new_matrix(n, m);
    b = new_matrix(m, p);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) fin >> a[i][j];
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) fin >> b[i][j];
    }

    fin.close();
}

const int THRESHOLD = 64;
void strassen(int** A, int ** B, int** C, int n, int m, int p) {
    if (std::max(n, std::max(m, p)) <= THRESHOLD) {
        matrix_multiply(A, B, C, n, m, p);
        return;
    }

    int n1 = (n + 1) / 2, m1 = (m + 1) / 2, p1 = (p + 1) / 2;

    int **A11 = new_matrix(n1, m1);
    init_sub_matrix(A, A11, n, m, 0, 0);
    int **A12 = new_matrix(n1, m1);
    init_sub_matrix(A, A12, n, m, 0, m1);
    int **A21 = new_matrix(n1, m1);
    init_sub_matrix(A, A21, n, m, n1, 0);
    int **A22 = new_matrix(n1, m1);
    init_sub_matrix(A, A22, n, m, n1, m1);

    int **B11 = new_matrix(m1, p1);
    init_sub_matrix(B, B11, m, p, 0, 0);
    int **B12 = new_matrix(m1, p1);
    init_sub_matrix(B, B12, m, p, 0, p1);
    int **B21 = new_matrix(m1, p1);
    init_sub_matrix(B, B21, m, p, m1, 0);
    int **B22 = new_matrix(m1, p1);
    init_sub_matrix(B, B22, m, p, m1, p1);

    int **M11 = new_matrix(n1, m1);
    int **M12 = new_matrix(m1, p1);
    int **M21 = new_matrix(n1, m1);
    int **M32 = new_matrix(m1, p1);
    int **M42 = new_matrix(m1, p1);
    int **M51 = new_matrix(n1, m1);
    int **M61 = new_matrix(n1, m1);
    int **M62 = new_matrix(m1, p1);
    int **M71 = new_matrix(n1, m1);
    int **M72 = new_matrix(m1, p1);

    int **M1 = new_matrix(n1, p1);
    int **M2 = new_matrix(n1, p1);
    int **M3 = new_matrix(n1, p1);
    int **M4 = new_matrix(n1, p1);
    int **M5 = new_matrix(n1, p1);
    int **M6 = new_matrix(n1, p1);
    int **M7 = new_matrix(n1, p1);

    //compute M1
    matrix_add(A11, A22, M11, n1, m1);
    matrix_add(B11, B22, M12, m1, p1);
    strassen(M11, M12, M1, n1, m1, p1);

    //compute M2
    matrix_add(A21, A22, M21, n1, m1);
    strassen(M21, B11, M2, n1, m1, p1);

    //compute M3
    matrix_sub(B12, B22, M32, m1, p1);
    strassen(A11, M32, M3, n1, m1, p1);

    //compute M4
    matrix_sub(B21, B11, M42, m1, p1);
    strassen(A22, M42, M4, n1, m1, p1);

    //compute M5
    matrix_add(A11, A12, M51, n1, m1);
    strassen(M51, B22, M5, n1, m1, p1);

    //compute M6
    matrix_sub(A21, A11, M61, n1, m1);
    matrix_add(B11, B12, M62, m1, p1);
    strassen(M61, M62, M6, n1, m1, p1);

    //compute M7
    matrix_sub(A12, A22, M71, n1, m1);
    matrix_add(B21, B22, M72, m1, p1);
    strassen(M71, M72, M7, n1, m1, p1);

    //compute C with seven matrix
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < p1; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            if (p1 + j < p) C[i][p1 + j] = M3[i][j] + M5[i][j];
            if (n1 + i < n) {
                C[n1 + i][j] = M2[i][j] + M4[i][j];
                if (p1 + j < p) C[n1 + i][p1 + j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }
        }
    }

    //deallocate some matrix temporary

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

int main() {
    
    std::ofstream fout;
    fout.open ("output.txt", std::ofstream::app);

    int **a, **b;
    int n, m, p;
    load_matrix(a, b, n, m, p);

    int **c = new_matrix(n, p);

    std::chrono::time_point<std::chrono::system_clock> start, end;

    double elapsed_seconds;
    start = std::chrono::system_clock::now();
    strassen(a, b, c, n, m, p);
    end = std::chrono::system_clock::now();
    elapsed_seconds = duration_cast<microseconds> (end - start).count() * pow(10, -6);
    fout << "naive " << n << ' ' << m << ' ' << p << ": " << elapsed_seconds << '\n';
    
    check_correctness(a, b, c, n, m, p);
    fout.close();

    return 0;
}