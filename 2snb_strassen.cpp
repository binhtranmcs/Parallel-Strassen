#include <fstream>

#include "mpi_2snb_strassen.hpp"


void check_correctness(int** A, int** B, int** C, int n, int m, int p) {
    int **E = new_matrix(n, p);
    matrix_multiply(A, B, E, n, m, p);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < p; ++c) {
            if (C[r][c] != E[r][c]) {
                std::cout << "INCORRECT\n";
                std::cout << r << ' ' << c << '\n';
                std::cout << C[r][c] << ' ' << E[r][c] << '\n';
            }
        }
    }
    std::cout << "CORRECT\n";
}

void load_matrix(int** &A, int** &B, int** &C, int &n, int &m, int &p, int rank) {
    std::fstream fin;
    fin.open("gen_input.txt", std::ios::in);
    fin >> n >> m >> p;

    if (rank == 0) {
        A = new_matrix(n, m);
        B = new_matrix(m, p);
        C = new_matrix(n, p);

        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < m; ++c) {
                fin >> A[r][c];
            }
        }
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < p; ++c) {
                fin >> B[r][c];
            }
        }
    }

    fin.close();
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
    int **A, **B, **C;
    int n, m, p;
    load_matrix(A, B, C, n, m, p, rank);

    // matrix multiplication with strassen algorithm
    double begin = MPI_Wtime();
    mpi_2snb_strassen(A, B, C, n, m, p, rank);
    double end = MPI_Wtime();

    if (rank == 0) {
        std::ofstream fout;
        fout.open("output.txt", std::ofstream::app);
        fout << "mpi_2snb_omp " << n << ' ' << m << ' ' << p << ": " << end - begin << '\n';
        fout.close();
        // check_correctness(A, B, C, n, m, p);
    }

    MPI_Finalize();
    return 0;
}
