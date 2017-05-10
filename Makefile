CC=gcc -O2 -DUSE_LIBJPEG
NVCC=nvcc -O2
LIB_PATH=-L/usr/local/cuda/lib64

all: main.o imageio.o young.o cuda_young.o

	$(CC) $(LIB_PATH) main.o imageio.o young.o cuda_young.o -lcuda -lcudart -lm -ljpeg  -o main 

main.o: main.c imageio.h

	$(CC) main.c -c

imageio.o: imageio.c imageio.h

	$(CC) imageio.c -c

young.o: young.c cuda_young.h

	$(CC) young.c -c 

cuda_young.o: cuda_young.cu cuda_young.h
	
	$(NVCC) cuda_young.cu -c 

clean:
	rm -f *.o *.out *~
	rm -f *.bin
