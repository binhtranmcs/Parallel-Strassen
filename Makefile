N = 1000
M = 1000
P = 1000

# mpi one-sided communication with omp for each machine
mpi_omp:
	mpicxx -fopenmp main.cpp -o mpi_omp.o
	mpirun -n 4 ./mpi_omp.o
	rm mpi_omp.o

# mpi p2p blocking communication with omp for each machine
mpi_2s_omp:
	mpicxx -fopenmp 2s_strassen.cpp -o 2s_strassen.o
	mpirun -n 4 ./2s_strassen.o
	rm 2s_strassen.o

# mpi p2p non-blocking communication with omp for each machine
mpi_2snb_omp:
	mpicxx -fopenmp 2snb_strassen.cpp -o 2snb_strassen.o
	mpirun -n 4 ./2snb_strassen.o
	rm 2snb_strassen.o

# generate 2 input matrices with size N*M and M*P
gen_test:
	g++ gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

# omp task parallel implementation
omp:
	g++ -fopenmp omp.cpp -o omp.o
	./omp.o
	rm omp.o

# naive implementation for strassen algorithm
naive:
	g++ naive_strassen.cpp -o naive_strassen.o
	./naive_strassen.o
	rm naive_strassen.o

clean:
	rm *.o
