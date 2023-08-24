INC="./inc"
FLAGS = -I$(INC)
OMPFLAG = -fopenmp
NVCC = nvcc
MPI = mpicc
LDFLAGS = -I. -lm 
CFLAGS = -DNDEBUG
CC = g++
OBJFILE0 = cuda_Dijkstra.cu
OBJFILE1 = mpi_Dijkstra.c
TARGET0 = cuda
TARGET1 = mpi


all: cuda mpi

cuda: cuda_Dijkstra.cu
	$(NVCC) $(OBJFILE0) -o $(TARGET0)

mpi: mpi_Dijkstra.c
	mpicc $(OBJFILE1) -o $(TARGET1)

clean:
	rm -f $(TARGET0) $(TARGET1) *~
