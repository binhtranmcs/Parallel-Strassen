#include "bits/stdc++.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Need 3 arguments n, m, p!\n";
        return 0;
    }

    std::ofstream fout;
    fout.open ("gen_input.txt", std::ofstream::app);

    const int MAX = 1e3;

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    int p = std::atoi(argv[3]);

    fout << n << ' ' << m << ' ' << p << '\n';

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) fout << rand() % MAX << " \n"[j == m - 1];
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) fout << rand() % MAX << " \n"[j == p - 1];
    }
}