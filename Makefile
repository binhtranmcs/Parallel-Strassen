strassen:
	g++ -std=c++11 -fopenmp main.cpp -o strassen.o
	./strassen.o
	rm strassen.o

gen_test:
	g++ -std=c++11 gen_test.cpp -o gen_test.o
	./gen_test.o $N $M $P
	rm gen_test.o

tmp:
	mpic++ tmp.cpp -o tmp.o
	mpirun -np $(PROC) ./tmp.o # --hostfile hostfile
	rm tmp.o