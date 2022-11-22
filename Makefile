N = 1000
M = 1000
P = 1000

mpi_omp:
	mpic++ -std=c++17 -fopenmp main.cpp -o mpi_omp.o
	mpirun -n 4 ./mpi_omp.o
	rm mpi_omp.o

mpi_2s_omp:
	mpic++ -std=c++17 -fopenmp 2s_strassen.cpp -o 2s_strassen.o
	mpirun -n 4 ./2s_strassen.o
	rm 2s_strassen.o

mpi_2snb_omp:
	mpic++ -std=c++17 -fopenmp 2snb_strassen.cpp -o 2snb_strassen.o
	mpirun -n 4 ./2snb_strassen.o
	rm 2snb_strassen.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

omp:
	g++ -fopenmp omp.cpp -o omp.o
	./omp.o
	rm omp.o

naive:
	g++ -std=c++11 naive_strassen.cpp -o naive_strassen.o
	./naive_strassen.o
	rm naive_strassen.o