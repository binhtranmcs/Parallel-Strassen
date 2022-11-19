mpi_omp:
	mpic++ -std=c++17 -fopenmp main.cpp -o mpi_omp.o
	mpirun -n 4 ./mpi_omp.o
	rm mpi_omp.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

omp:
	mpic++ -fopenmp omp.cpp -o omp.o
	./omp.o
	rm omp.o
