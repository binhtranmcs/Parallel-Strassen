mpi_omp:
	mpic++ -std=c++17 -fopenmp main.cpp -o mpi_omp.o
	mpirun -n 4 ./mpi_omp.o
	rm mpi_omp.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

test:
	mpic++ -fopenmp test.cpp -o test.o
	mpirun -n 3 ./test.o
	rm test.o

test_mpi:
	mpic++ -fopenmp test_mpi.cpp -o test_mpi.o
	mpirun -n 4 ./test_mpi.o
	rm test_mpi.o

omp:
	mpic++ -fopenmp omp.cpp -o omp.o
	./omp.o
	rm omp.o

test_hybrid:
	mpic++ -fopenmp test_hybrid.cpp -o test_hybrid.o
	mpirun -n 4 ./test_hybrid.o
	rm test_hybrid.o
