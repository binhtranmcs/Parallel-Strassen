#include "bits/stdc++.h"
#include "omp_strassen.hpp"

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

int main(int argc, char** argv) {
    std::ofstream fout;
    fout.open ("output.txt", std::ofstream::app);

    int **a, **b;
    int n, m, p;
    load_matrix(a, b, n, m, p);

    int **c = new_matrix(n, p);

    omp_set_dynamic(0);
    omp_set_num_threads(24);
    
    double begin = omp_get_wtime();

    omp_strassen(a, b, c, n, m, p);

    double end = omp_get_wtime();

    fout << "omp " << n << ' ' << m << ' ' << p << ": " << end - begin << '\n';

    // check_correctness(a, b, c, n, m, p);

    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < p; ++j) fout << c[i][j] << ' ';
    //     fout << '\n';
    // }

    fout.close();
}
