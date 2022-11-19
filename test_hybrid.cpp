#include <bits/stdc++.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int get(int *matrix, int r, int c) {
    return matrix[2 + r * matrix[1] + c];
}

bool set(int *matrix, int r, int c, int val) {
    int n = matrix[0], m = matrix[1];
    if (r >= n || r < 0 || c >= m || c < 0) return false;
    matrix[2 + r * m + c] = val;
    return true;
}

int main(int argc, char* argv[])
{
    // MPI init
    MPI_Init(&argc, &argv);
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 4) {
        printf("This application is meant to be run with 4 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Get my rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // // OMP init
    // omp_set_dynamic(0);
    // omp_set_num_threads(24);

    // // MPI template
    // const int LEN = 10;
    // int *window_buffer = new int[LEN];
    // MPI_Win window;
    // MPI_Win_create(window_buffer, sizeof(int) * LEN, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    // MPI_Win_fence(0, window);
    // if (my_rank == 0) {
    //     MPI_Put(window_buffer, LEN, MPI_INT, 1, 0, LEN, MPI_INT, window);
    // }
    // MPI_Win_fence(0, window);
    // MPI_Win_free(&window);
    // MPI_Finalize();

    // int n = 3, m = 4;
    // int len = n * m + 2;
    // int *matrix = new int[len];
    // matrix[0] = n;
    // matrix[1] = m;
    // MPI_Win win;
    // MPI_Win_create(matrix, len * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // MPI_Win_fence(0, win);

    if (rank == 0) {
        std::ofstream fout;
        fout.open ("output.txt", std::ofstream::app);

        int *a, *b;
        int n, m, p;
        load_matrix(a, b, n, m, p);

        int **c = new_matrix(n, p);

        double begin = omp_get_wtime();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) set(matrix, i, j, i + j);
        }
        MPI_Put(matrix, len, MPI_INT, 1, 0, len, MPI_INT, win);
    }

    MPI_Win_fence(0, win);

    if (rank == 1) {
        int n = matrix[0], m = matrix[1];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) std::cout << get(matrix, i, j) << ' ';
            std::cout << '\n';
        }
    }

    MPI_Win_free(&win);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
