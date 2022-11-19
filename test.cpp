// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>
 
// /**
//  * @brief Illustrate how to put data into a target window.
//  * @details This application consists of two MPI processes. MPI process 1
//  * exposes a window containing an integer. MPI process 0 puts the value 12345
//  * in it via MPI_Put. After the MPI_Put is issued, synchronisation takes place
//  * via MPI_Win_fence and the MPI process 1 prints the value in its window.
//  **/
// int main(int argc, char* argv[])
// {
//     MPI_Init(&argc, &argv);
 
//     // Check that only 2 MPI processes are spawn
//     int comm_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
//     if(comm_size != 4)
//     {
//         printf("This application is meant to be run with 2 MPI processes, not %d.\n", comm_size);
//         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//     }
 
//     // Get my rank
//     int my_rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
//     // Create the window
//     int window_buffer = 0;
//     MPI_Win window;
//     MPI_Win_create(&window_buffer, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

//     MPI_Win_fence(0, window);

//     int my_value;
//     if (my_rank == comm_size - 1) {
//         my_value = 10;
//     }
//     else {
//         my_value = window_buffer + 1;
//     }

//     if (my_rank > 0) {
//         MPI_Put(&my_value, 1, MPI_INT, my_rank - 1, 0, 1, MPI_INT, window);
//     }

//     MPI_Win_fence(0, window);

//     std::cout << "proc " << my_rank << " get " << window_buffer << '\n';
 
//     // Destroy the window
//     MPI_Win_free(&window);
 
//     MPI_Finalize();
 
//     return EXIT_SUCCESS;
// }

#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

void strass(int rank) {
    if (rank < -1) return;
    int np, iam;
#pragma omp parallel
{
#pragma omp single private(np, iam)
{
    for (int i = 0; i < 5; ++i) {
    #pragma omp task
    {
        np = omp_get_max_threads();
        iam = omp_get_thread_num();
        printf("Hybrid: Hello from thread %d out of %d from process %d\n", iam, np, rank);
        strass(rank - 1);
    }
    }   
}
}
}

int main(int argc, char *argv[])
{
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int iam = 0, np = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    omp_set_num_threads(5);

    // #pragma omp parallel default(shared) private(iam, np)
    // {
    //     np = omp_get_num_threads();
    //     iam = omp_get_thread_num();
    //     printf("Hybrid: Hello from thread %d out of %d from process %d out of %d on %s\n",
    //             iam, np, rank, numprocs, processor_name);
    // }

    if (rank == 0) {
        strass(rank);
    }

    MPI_Finalize();

    return 0;
}
