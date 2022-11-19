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

    strass(rank);

    MPI_Finalize();

    return 0;
}

// https://gist.github.com/huzhifeng/d1cda3f0474261eda72b36ca83f24e21
// #pragma omp parallel default(shared) private(iam, np)
// {
//     np = omp_get_num_threads();
//     iam = omp_get_thread_num();
//     printf("Hybrid: Hello from thread %d out of %d from process %d out of %d on %s\n",
//             iam, np, rank, numprocs, processor_name);
// }
