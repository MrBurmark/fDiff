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
int find_pos(int i, int P, int N) {
	return i * (N / P) + i * (N % P) / P;
}

int main(int argc, char **argv) {
	int size, my_size[2];
	int numCycles;
	int i, j, k, l, n;
	int ok, tag=0;
	int num_procs, my_rank, my_coord[2];
	int S_rank, N_rank, E_rank, W_rank;
	int tmp[2], dims[2], periods[2] = {0,0};
	int *all_sizes, *all_offsets;
	double *uall, *uold, *unew, *tptr;
	double inTemp;
	int cycle = 0; // tag with cycle, but message order per other processor should be fixed
	int numInit;
	MPI_Status status;
	MPI_Comm CART_COMM;
	MPI_Datatype COL_DOUBLE, BLOCK_DOUBLE, MID_DOUBLE;

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

		uall = (double *) calloc(size*size, sizeof(double));
	}

	MPI_Bcast(tmp, 2, MPI_INT, 0, MPI_COMM_WORLD);

	numCycles = tmp[0];
	size = tmp[1];

	// consider non-square case, create most square blocks possible to reduce communication
	dims[0] = 1;
	dims[1] = 1;
	while (dims[0]*dims[1] < num_procs) dims[0]++,dims[1]++;
	if (dims[0]*dims[1] > num_procs) dims[0]--,dims[1]--;

	// MPI_Dims_create(num_procs, 2, dims);

    all_sizes = (int *) calloc(dims[0]*dims[1], sizeof(int));
    all_offsets = (int *) calloc(dims[0]*dims[1], sizeof(int));

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &CART_COMM);
	MPI_Cart_coords(CART_COMM, my_rank, 2, my_coord);

	for (i=0; i < dims[0]; i++){
		for(j=0; j < dims[1]; j++) {
			all_sizes[i*dims[1]+j] = 1;
			all_offsets[i*dims[1]+j] = find_pos(i, dims[0], size) * size + find_pos(j, dims[1], size);
		}
    }

	if(DEBUG) {
		printf("my rank %i, my coords %i, %i, my offset %i\n", my_rank, my_coord[0], my_coord[1], all_offsets[my_rank]);
		MPI_Barrier(CART_COMM);
	}

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
	MPI_Type_vector(my_size[0],          1, my_size[1]+2, MPI_DOUBLE, &COL_DOUBLE);
	MPI_Type_commit(&COL_DOUBLE);

	MPI_Type_vector(my_size[0], my_size[1],         size, MPI_DOUBLE, &BLOCK_DOUBLE);
	MPI_Type_commit(&BLOCK_DOUBLE);

	MPI_Type_vector(my_size[0], my_size[1], my_size[1]+2, MPI_DOUBLE, &MID_DOUBLE);
	MPI_Type_commit(&MID_DOUBLE);


	if (0 == my_rank) {
		for (n=0; n<numInit; n++) {
			ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
			dataAt(uall, i, j, size) = inTemp;
		}
	}

	if(DEBUG && 0 == my_rank) {
		printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
		mpPrintGrid(uall, size, size);
	}

	uold = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));

	//MPI_Scatterv(uall, all_sizes, all_offsets, BLOCK_DOUBLE, uold + my_size[1]+3, 1, MID_DOUBLE, 0, CART_COMM);

	for (i=0; i < dims[0]; i++){
		for (j=0; j < dims[1]; j++){
			tmp[0] = i;
			tmp[1] = j;
			MPI_Cart_rank(CART_COMM, tmp, &k);

			if(DEBUG) {
				if (0 == my_rank){
					printf("dest rank %i, offset %i\n", k, all_offsets[k]);
				}
				MPI_Barrier(CART_COMM);
			}

			if (0 == k && 0 == my_rank)
				MPI_Sendrecv(uall+all_offsets[k], 1, BLOCK_DOUBLE, 0, tag, 
					uold+my_size[1]+3, 1, MID_DOUBLE, 0, tag, CART_COMM, &status);
			else if (0 == my_rank)
				MPI_Send(uall+all_offsets[k], 1, BLOCK_DOUBLE, k, tag, CART_COMM);
			else if (k == my_rank)
				MPI_Recv(uold+my_size[1]+3, 1, MID_DOUBLE, 0, tag, CART_COMM, &status);
		}
	}

	if (0 == my_rank) free(uall);

	unew = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));

	if(DEBUG) {
		printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
		mpPrintGrid(uold, my_size[0]+2, my_size[1]+2);
		MPI_Barrier(CART_COMM);
	}











	

  //printGrid(u1, size);

	// for (cycle=0; cycle<numCycles; cycle++) {
	// sendrecv
	// 	updateGrid(unew, uold, size); // mp it
	// //printGrid(unew, size);
	// 	tptr = unew;
	// 	unew = uold;
	// 	uold = tptr;
	// }











	free(unew);

	if (0 == my_rank) uall = (double *) calloc(size*size, sizeof(double));

	//MPI_Gatherv(uold + my_size[1]+3, 1, MID_DOUBLE, uall, all_sizes, all_offsets, BLOCK_DOUBLE, 0, CART_COMM);

	for (i=0; i < dims[0]; i++){
		for (j=0; j < dims[1]; j++){
			tmp[0] = i;
			tmp[1] = j;
			MPI_Cart_rank(CART_COMM, tmp, &k);

			if (0 == k && 0 == my_rank)
				MPI_Sendrecv(uold+my_size[1]+3, 1, MID_DOUBLE, 0, tag, 
					uall+all_offsets[k], 1, BLOCK_DOUBLE, 0, tag, CART_COMM, &status);
			else if (k == my_rank)
				MPI_Send(uold+my_size[1]+3, 1, MID_DOUBLE, 0, tag, CART_COMM);
			else if (0 == my_rank)
				MPI_Recv(uall+all_offsets[k], 1, BLOCK_DOUBLE, k, tag, CART_COMM, &status);
		}
	}

	if(DEBUG && 0 == my_rank) {
		printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
		mpPrintGrid(uall, size, size);
	}

	if (0 == my_rank)
		dumpGrid(uall, size);

	MPI_Type_free(&COL_DOUBLE);
	MPI_Type_free(&BLOCK_DOUBLE);
	MPI_Type_free(&MID_DOUBLE);
	MPI_Comm_free(&CART_COMM);
	MPI_Finalize();

}
