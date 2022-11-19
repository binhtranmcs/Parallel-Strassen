strassen:
	mpic++ -std=c++11 -fopenmp main.cpp -o strassen.o
	mpirun -n 3 strassen.o
	rm strassen.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

test:
	mpic++ -fopenmp test.cpp -o test.o
	mpirun -n 3 ./test.o # --hostfile hostfile
	rm test.o

test_mpi:
	mpic++ -fopenmp test_mpi.cpp -o test_mpi.o
	mpirun -n 4 ./test_mpi.o # --hostfile hostfile
	rm test_mpi.o

test_hybrid:
	mpic++ -fopenmp test_hybrid.cpp -o test_hybrid.o
	mpirun -n 4 ./test_hybrid.o # --hostfile hostfile
	rm test_hybrid.o
