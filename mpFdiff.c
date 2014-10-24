/*************************
Author: Rachel Beasley, Jason Burmark, Moses Lee
COMPILE: mpicc mpFdiff.c Fdutils.c -O3 -o mpFd -lm
EXECUTE: mpirun -n [number of nodes] ./mpFd [input file]

Performs 4 nearest neighbor updates on 2-D grid
Input file format:

# cycles
size of grid (including boundary)
# initial data points
   
3 integers per data point: i and j indices, data
*************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Fdiff.h"

// less likely to overflow
int find_pos(int division, int num_divisions, int size) {
	return division * (size / num_divisions) + division * (size % num_divisions) / num_divisions;
}

int main(int argc, char **argv) {
	int size, my_size[2];
	int numCycles;
	int i, j, n;
	int ok;
	int num_procs, my_rank, my_coord[2];
	int S_rank, N_rank, E_rank, W_rank;
	int tmp[2], dims[2], periods[2] = {0,0};
	double *u0, *u1, *tptr;
	double inTemp;
	int cycle = 0; // tag with cycle, but message order per other processor should be fixed
	int numInit;
	MPI_Comm CART_COMM;
	MPI_Datatype COL_DOUBLE;

	FILE *fp;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (0 == my_rank) {
		fp = fopen(argv[1], "r");

		ok = fscanf(fp, "%d", &numCycles);
		ok = fscanf(fp, "%d", &size);
		ok = fscanf(fp, "%d", &numInit);
		printf("# cycles %d size %d # initializations %d\n", numCycles, size, numInit);
		tmp[0] = numCycles;
		tmp[1] = size;
	}

	MPI_Bcast(tmp, 2, MPI_INT, 0, MPI_COMM_WORLD);

	numCycles = tmp[0];
	size = tmp[1];

	// consider non-square case, create most square blocks possible to reduce communication
	dims[0] = 1;
	dims[1] = 1;
	while (dims[0]*dims[1] < num_procs) dims[0]++,dims[1]++;
	if (dims[0]*dims[1] > num_procs) dims[0]--,dims[1]--;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &CART_COMM);
	MPI_Cart_coords(CART_COMM, my_rank, 2, my_coord);

	// find neighbors
	MPI_Cart_shift(CART_COMM, 0,  1, &N_rank, &S_rank);
	MPI_Cart_shift(CART_COMM, 1,  1, &W_rank, &E_rank);

	// consider giving edges more computations as they have fewer sendrecv
	my_size[0] = find_pos(my_coord[0]+1, dims[0], size) - find_pos(my_coord[0], dims[0], size);
	my_size[1] = find_pos(my_coord[1]+1, dims[1], size) - find_pos(my_coord[1], dims[1], size);

	// may be different sizes for different processors
	// horizontal neighbors should have same height
	// take halo into account with + 2
	// allows sending the halo columns without manually copying
	MPI_Type_vector(my_size[0], 1, my_size[1]+2, MPI_DOUBLE, &COL_DOUBLE);
	MPI_Type_commit(&COL_DOUBLE);

	u0 = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));
	u1 = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));













	initGrid(u0, u1, size);

	for (n=0; n<numInit; n++) {
		ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
		dataAt(u1, i, j, size) = inTemp;
	}

  //printGrid(u1, size);

	for (cycle=0; cycle<numCycles; cycle++) {
		updateGrid(u0, u1, size);
	//printGrid(u0, size);
		tptr = u0;
		u0 = u1;
		u1 = tptr;
	}

	dumpGrid(u1, size);

	MPI_Type_free(&COL_DOUBLE);
	MPI_Comm_free(&CART_COMM);
	MPI_Finalize();

}

