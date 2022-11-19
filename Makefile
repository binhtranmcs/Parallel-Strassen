strassen:
	g++ -std=c++11 -fopenmp main.cpp -o strassen.o
	./strassen.o
	rm strassen.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

test:
	mpic++ -fopenmp test.cpp -o test.o
	mpirun -n 4 ./test.o # --hostfile hostfile
	rm test.o