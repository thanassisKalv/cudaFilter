CC=gcc -O2
NVCC=nvcc -O2
LIB_PATH=-L/usr/local/cuda/lib64

all: main.o  cuda_young.o

	$(CC) $(LIB_PATH) main.o cuda_young.o -lcuda -lcudart -lm -o main 

main.o: main.c

	$(CC) main.c -c


cuda_young.o: cuda_young.cu cuda_young.h
	
	$(NVCC) cuda_young.cu -c 

clean:
	rm -f *.o *.out *~
	rm -f *.bin
