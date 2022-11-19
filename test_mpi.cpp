#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
 
/**
 * @brief Illustrate how to put data into a target window.
 * @details This application consists of two MPI processes. MPI process 1
 * exposes a window containing an integer. MPI process 0 puts the value 12345
 * in it via MPI_Put. After the MPI_Put is issued, synchronisation takes place
 * via MPI_Win_fence and the MPI process 1 prints the value in its window.
 **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Check that only 2 MPI processes are spawn
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 4)
    {
        printf("This application is meant to be run with 4 MPI processes, not %d.\n", comm_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
    // Create the window
    int LEN = rand() % 10 + 10;
    std::cout << "LEN: " << LEN << '\n';
    int *window_buffer = new int[LEN];
    MPI_Win window;
    MPI_Win_create(window_buffer, sizeof(int) * LEN, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

    MPI_Win_fence(0, window);

    if (my_rank == 0) {
        for (int i = 0; i < LEN; ++i) window_buffer[i] = rand() % 50;
        MPI_Put(window_buffer, LEN, MPI_INT, 1, 0, LEN, MPI_INT, window);
        int x = 12;
        MPI_Put(&x, 1, MPI_INT, 2, 0, 1, MPI_INT, window);
        MPI_Put(window_buffer, LEN / 3, MPI_INT, 3, 0, LEN / 3, MPI_INT, window);
    }
    else {
        memset(window_buffer, 0, sizeof(int) * LEN);
    }

    MPI_Win_fence(0, window);

    MPI_Win_fence(0, window);

    if (my_rank == 1) {
        for (int i = 0; i < LEN; ++i) window_buffer[i] *= 2;
        MPI_Put(window_buffer + LEN / 2, LEN / 2, MPI_INT, 2, 3, LEN / 2, MPI_INT, window);
    }

    MPI_Win_fence(0, window);

    std::cout << "rank: " << my_rank << '\n';
    for (int i = 0; i < LEN; ++i) std::cout << window_buffer[i] << ' ';
    std::cout << '\n';

    // Destroy the window
    MPI_Win_free(&window);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}