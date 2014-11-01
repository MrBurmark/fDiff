/*************************
Author: Rachel Beasley, Jason Burmark, Moses Lee
COMPILE: mpicc mpFdiff.c Fdutils.c -O3 -o mpFd -lm
RUN: mpirun -n [number of nodes] ./mpFd [input file]

Performs 4 nearest neighbor updates on 2-D grid
Input file format:

# cycles
size of grid (including boundary)
# initial data points
   
3 integers per data point: i and j indices, data
*************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "prt_time.h"
#include "Fdiff.h"

// less likely to overflow
int find_pos(int i, int P, int N) {
	if (i == P) return N;
	else if (0 == i) return 0;
	return i * (N / P) + i * (N % P) / P;
}

int main(int argc, char **argv) {
	int size, calc_size, my_size[2];
	int numCycles;
	int my_type;
	int i, j, k, l, n;
	int start[2], stop[2];
	int ok, tag=0;
	int num_procs, my_rank, their_rank, my_coord[2], their_coords[2];
	int S_rank, N_rank, E_rank, W_rank, num_neighbors, my_neighbors[4];
	int tmp[2], dims[2] = {0,0}, periods[2] = {0,0};
	int *all_sizes, *all_offsets, my_offset;
	double *uall, *uold, *unew, *tptr;
	double t[4] = {0.0,0.0,0.0,0.0}, tr[4], t_t0, t_t1, t_t2, t_t3, t_t4, ts;
	double inTemp;
	int cycle = 0; // tag with cycle, but message order per other processor should be fixed
	int numInit;
	MPI_Status status;
	MPI_Request request[8];
	MPI_Comm CART_COMM, SHARE_COMM;
	MPI_Group comm_world_group, group;
	MPI_Win WINold, WINnew, WINtmp;
	MPI_Datatype COL_DOUBLE, TMP_BLOCK_DOUBLE, BLOCK_DOUBLE, MID_DOUBLE;

	FILE *fp;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (0 == my_rank) {
		prt_time();

		fp = fopen(argv[1], "r");

		ok = fscanf(fp, "%d", &numCycles);
		ok = fscanf(fp, "%d", &size);
		ok = fscanf(fp, "%d", &numInit);
		printf("cycles %d, size %d, initializations %d, procs %d\n", numCycles, size, numInit, num_procs);
		tmp[0] = numCycles;
		tmp[1] = size;

		uall = (double *) calloc(size*size, sizeof(double));

		for (n=0; n<numInit; n++) {
			ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
			dataAt(uall, i, j, size) = inTemp;
		}
		fclose(fp);
#if RMA && FENCE
		printf("Using RMA with fence\n");
#elif RMA && !FENCE
		printf("Using RMA with start/post\n");
#elif BLOCKING
		printf("Using blocking sends and receives\n");
#else
		printf("Using non-blocking sends and receives\n");
#endif
#if REORDER
		printf("Reorder enabled\n");
#else
		printf("Reorder disabled\n");
#endif
	}

	MPI_Bcast(tmp, 2, MPI_INT, 0, MPI_COMM_WORLD);

	numCycles = tmp[0];
	size = tmp[1];
	calc_size = size - 2;

	MPI_Dims_create(num_procs, 2, dims);

	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, REORDER, &CART_COMM);
	MPI_Cart_coords(CART_COMM, my_rank, 2, my_coord);

	// break grid up so neighbors of opposite types
	// allows matching sends and recieves in proper order
	my_type = (my_coord[0] + my_coord[1])%2;

	my_offset = (find_pos(my_coord[0], dims[0], calc_size) + 1) * size + find_pos(my_coord[1], dims[1], calc_size) + 1;

	if (0 == my_rank) {
		printf("Using %ix%i cartesian grid\n", dims[0], dims[1]);

		all_sizes = (int *) calloc(dims[0]*dims[1], sizeof(int));
	    all_offsets = (int *) calloc(dims[0]*dims[1], sizeof(int));

		for (i=0; i < dims[0]; i++){
			for(j=0; j < dims[1]; j++) {

				their_coords[0] = i;
				their_coords[1] = j;

				MPI_Cart_rank(CART_COMM, their_coords, &their_rank);

				all_sizes[their_rank] = 1;
				all_offsets[their_rank] = (find_pos(i, dims[0], calc_size) + 1) * size + find_pos(j, dims[1], calc_size) + 1;
			}
		}
	}

	// find neighbors
	MPI_Cart_shift(CART_COMM, 0,  1, &N_rank, &S_rank);
	MPI_Cart_shift(CART_COMM, 1,  1, &W_rank, &E_rank);

	// consider giving edges more computations as they have fewer sendrecv
	my_size[0] = find_pos(my_coord[0]+1, dims[0], calc_size) - find_pos(my_coord[0], dims[0], calc_size);
	my_size[1] = find_pos(my_coord[1]+1, dims[1], calc_size) - find_pos(my_coord[1], dims[1], calc_size);


#if DEBUG
	if(0 == my_rank) {
		printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
		mpPrintGrid(uall, size, size);
	}
	fflush(stdout);
	MPI_Barrier(CART_COMM);
	for (i=0; i < num_procs; i++) {
		if (i == my_rank) {
			printf("my rank %i, my coords %i, %i, my sizes %ix%i\n", my_rank, my_coord[0], my_coord[1], my_size[0], my_size[1]);
		}
		fflush(stdout);
		MPI_Barrier(CART_COMM);
	}
#endif


	// may be different sizes for different processors
	// horizontal neighbors should have same height
	// take halo into account with + 2
	// allows sending the halo columns without manually copying
	MPI_Type_vector(my_size[0], my_size[1],         size, MPI_DOUBLE, &TMP_BLOCK_DOUBLE);
	MPI_Type_create_resized(TMP_BLOCK_DOUBLE, 0, sizeof(double), &BLOCK_DOUBLE);
	MPI_Type_commit(&BLOCK_DOUBLE);
	MPI_Type_free(&TMP_BLOCK_DOUBLE);

	MPI_Type_vector(my_size[0], my_size[1], my_size[1]+2, MPI_DOUBLE, &MID_DOUBLE);
	MPI_Type_commit(&MID_DOUBLE);

	MPI_Type_vector(my_size[0],          1, my_size[1]+2, MPI_DOUBLE, &COL_DOUBLE);
	MPI_Type_commit(&COL_DOUBLE);



#if !RMA

	uold = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));

	ts = MPI_Wtime();

	MPI_Scatterv(uall, all_sizes, all_offsets, BLOCK_DOUBLE, 
				 uold + my_size[1]+3, 1, MID_DOUBLE, 0, CART_COMM);

	ts = MPI_Wtime() - ts;

	unew = (double *) calloc((my_size[0]+2) * (my_size[1]+2), sizeof(double));


#elif RMA

	MPI_Alloc_mem((my_size[0]+2) * (my_size[1]+2) * sizeof(double), MPI_INFO_NULL, &uold);
	memset(uold, 0, (my_size[0]+2) * (my_size[1]+2) * sizeof(double));

	if (0 == my_rank) {
		MPI_Win_create(uall, size*size*sizeof(double), sizeof(double), MPI_INFO_NULL, CART_COMM, &WINold);
	} else {
		MPI_Win_create(NULL, 0, sizeof(double), MPI_INFO_NULL, CART_COMM, &WINold);
	}

	ts = MPI_Wtime();

	MPI_Win_fence(0, WINold);

	MPI_Get(uold + my_size[1]+3, 1, MID_DOUBLE, 0, my_offset, 1, BLOCK_DOUBLE, WINold);

	MPI_Win_fence(0, WINold);

	ts = MPI_Wtime() - ts;

	MPI_Win_free(&WINold);

	if (0 == my_rank) free(uall);

	MPI_Alloc_mem((my_size[0]+2) * (my_size[1]+2) * sizeof(double), MPI_INFO_NULL, &unew);
	memset(unew, 0, (my_size[0]+2) * (my_size[1]+2) * sizeof(double));

#if !FENCE

	num_neighbors = 0;

	if (N_rank != MPI_PROC_NULL) {
		my_neighbors[num_neighbors] = N_rank;
		num_neighbors++;
	}
	if (S_rank != MPI_PROC_NULL) {
		my_neighbors[num_neighbors] = S_rank;
		num_neighbors++;
	}
	if (E_rank != MPI_PROC_NULL) {
		my_neighbors[num_neighbors] = E_rank;
		num_neighbors++;
	}
	if (W_rank != MPI_PROC_NULL) {
		my_neighbors[num_neighbors] = W_rank;
		num_neighbors++;
	}

	MPI_Comm_group(CART_COMM, &comm_world_group);
	MPI_Group_incl(comm_world_group, num_neighbors, my_neighbors, &group);
	MPI_Group_free(&comm_world_group);

#endif

	MPI_Win_create(uold, (my_size[0]+2) * (my_size[1]+2) * sizeof(double), sizeof(double), MPI_INFO_NULL, CART_COMM, &WINold);
	MPI_Win_create(unew, (my_size[0]+2) * (my_size[1]+2) * sizeof(double), sizeof(double), MPI_INFO_NULL, CART_COMM, &WINnew);

#endif


#if DEBUG
	for (i=0; i < num_procs; i++) {
		if (i == my_rank) {
			printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
			mpPrintGrid(uold, my_size[0]+2, my_size[1]+2);
		}
		fflush(stdout);
		MPI_Barrier(CART_COMM);
	}
#endif


	// set up bounds
	start[0] = 1;
	start[1] = 1;
	stop[0] = my_size[0]+1;
	stop[1] = my_size[1]+1;

	MPI_Barrier(CART_COMM);



#if RMA

	for (cycle=0; cycle<numCycles; cycle++) {
#if PRINT_CYCLES
		if (0 == my_rank && 0 == cycle%PRINT_CYCLES)
			printf("cycle %i\n", cycle);
#endif

		t_t0 = MPI_Wtime();

#if FENCE
		MPI_Win_fence(0, WINold);
#else
		if (0 == my_type) {
			MPI_Win_start(group, 0, WINold);
			MPI_Win_post(group, 0, WINold);
		} else {
			MPI_Win_post(group, 0, WINold);
			MPI_Win_start(group, 0, WINold);
		}
#endif

		MPI_Get(uold+2*my_size[1]+3, 1, COL_DOUBLE, E_rank,
					 my_size[1] + 3, 1, COL_DOUBLE, WINold);

		MPI_Get(uold+my_size[1] + 2, 1, COL_DOUBLE, W_rank,
					 2*my_size[1]+2, 1, COL_DOUBLE, WINold);

		MPI_Get(uold +                           1, my_size[1], MPI_DOUBLE, N_rank,
					 (my_size[0])*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, WINold);

		MPI_Get(uold+(my_size[0]+1)*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank,
					 my_size[1]             +      3, my_size[1], MPI_DOUBLE, WINold);


		t_t1 = MPI_Wtime();

		mpUpdateGrid(unew, uold, my_size[1]+2, start[0]+1, stop[0]-1, start[1]+1, stop[1]-1);

		t_t2 = MPI_Wtime();

#if FENCE
		MPI_Win_fence(0, WINold);
#else
		if (0 == my_type) {
			MPI_Win_complete(WINold);
			MPI_Win_wait(WINold);
		} else {
			MPI_Win_wait(WINold);
			MPI_Win_complete(WINold);
		}
#endif

		t_t3 = MPI_Wtime();

		mpUpdateGridBorder(unew, uold, my_size[1]+2, start[0], stop[0], start[1], stop[1]);

		WINtmp = WINnew;
		WINnew = WINold;
		WINold = WINtmp;

		tptr = unew;
		unew = uold;
		uold = tptr;

		t_t4 = MPI_Wtime();

		t[0] += (t_t1 - t_t0) + (t_t3 - t_t2);
		t[1] += (t_t2 - t_t1) + (t_t4 - t_t3);

#if DEBUG
		for (i=0; i < num_procs; i++) {
			if (i == my_rank) {
				printf("my rank %i, my coords %i, %i, my computations %i\n", my_rank, my_coord[0], my_coord[1], (stop[0] - start[0]) * (stop[1] - start[1]));
				mpPrintGrid(uold, my_size[0]+2, my_size[1]+2);
			}
			fflush(stdout);
			MPI_Barrier(CART_COMM);
		}
#endif
	}



#elif BLOCKING

	for (cycle=0; cycle<numCycles; cycle++) {
#if PRINT_CYCLES
		if (0 == my_rank && 0 == cycle%PRINT_CYCLES)
			printf("cycle %i\n", cycle);
#endif

		t_t0 = MPI_Wtime();

		if (0 == my_type) { // E W N S
			MPI_Sendrecv(uold+2*my_size[1]+2, 1, COL_DOUBLE, E_rank, cycle,
						 uold+2*my_size[1]+3, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+my_size[1]+3, 1, COL_DOUBLE, W_rank, cycle,
						 uold+my_size[1]+2, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+my_size[1]+3, my_size[1], MPI_DOUBLE, N_rank, cycle,
						 uold+           1, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+(my_size[0])  *(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle,
						 uold+(my_size[0]+1)*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &status);
		} else { // W E S N
			MPI_Sendrecv(uold+my_size[1]+3, 1, COL_DOUBLE, W_rank, cycle,
						 uold+my_size[1]+2, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+2*my_size[1]+2, 1, COL_DOUBLE, E_rank, cycle,
						 uold+2*my_size[1]+3, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+(my_size[0])  *(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle,
						 uold+(my_size[0]+1)*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &status);

			MPI_Sendrecv(uold+my_size[1]+3, my_size[1], MPI_DOUBLE, N_rank, cycle,
						 uold+           1, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &status);
		}

		t_t1 = MPI_Wtime();

		mpUpdateGrid(unew, uold, my_size[1]+2, start[0], stop[0], start[1], stop[1]);
		
		tptr = unew;
		unew = uold;
		uold = tptr;

		t_t2 = MPI_Wtime();

		t[0] += t_t1 - t_t0;
		t[1] += t_t2 - t_t1;

#if DEBUG
		for (i=0; i < num_procs; i++) {
			if (i == my_rank) {
				printf("my rank %i, my coords %i, %i, my computations %i\n", my_rank, my_coord[0], my_coord[1], (stop[0] - start[0]) * (stop[1] - start[1]));
				mpPrintGrid(uold, my_size[0]+2, my_size[1]+2);
			}
			fflush(stdout);
			MPI_Barrier(CART_COMM);
		}
#endif
	}



#else

	for (cycle=0; cycle<numCycles; cycle++) {
#if PRINT_CYCLES
		if (0 == my_rank && 0 == cycle%PRINT_CYCLES)
			printf("cycle %i\n", cycle);
#endif

		t_t0 = MPI_Wtime();

		if (0 == my_type) { // E W N S
			MPI_Isend(uold+2*my_size[1]+2, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &request[4]);
			MPI_Irecv(uold+2*my_size[1]+3, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &request[0]);

			MPI_Isend(uold+my_size[1]+3, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &request[5]);
			MPI_Irecv(uold+my_size[1]+2, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &request[1]);

			MPI_Isend(uold+my_size[1]+3, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &request[6]);
			MPI_Irecv(uold+           1, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &request[2]);

			MPI_Isend(uold+(my_size[0])  *(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &request[7]);
			MPI_Irecv(uold+(my_size[0]+1)*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &request[3]);
		} else { // W E S N
			MPI_Irecv(uold+my_size[1]+2, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &request[4]);
			MPI_Isend(uold+my_size[1]+3, 1, COL_DOUBLE, W_rank, cycle, CART_COMM, &request[0]);
			
			MPI_Irecv(uold+2*my_size[1]+3, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &request[5]);
			MPI_Isend(uold+2*my_size[1]+2, 1, COL_DOUBLE, E_rank, cycle, CART_COMM, &request[1]);
			
			MPI_Irecv(uold+(my_size[0]+1)*(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &request[6]);
			MPI_Isend(uold+(my_size[0])  *(my_size[1]+2)+1, my_size[1], MPI_DOUBLE, S_rank, cycle, CART_COMM, &request[2]);
			
			MPI_Irecv(uold+           1, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &request[7]);
			MPI_Isend(uold+my_size[1]+3, my_size[1], MPI_DOUBLE, N_rank, cycle, CART_COMM, &request[3]);
		}

		t_t1 = MPI_Wtime();

		mpUpdateGrid(unew, uold, my_size[1]+2, start[0]+1, stop[0]-1, start[1]+1, stop[1]-1);

		t_t2 = MPI_Wtime();
		
		MPI_Waitall(8, request, MPI_STATUSES_IGNORE);

		t_t3 = MPI_Wtime();

		mpUpdateGridBorder(unew, uold, my_size[1]+2, start[0], stop[0], start[1], stop[1]);

		tptr = unew;
		unew = uold;
		uold = tptr;

		t_t4 = MPI_Wtime();

		t[0] += (t_t1 - t_t0) + (t_t3 - t_t2);
		t[1] += (t_t2 - t_t1) + (t_t4 - t_t3);

#if DEBUG
		for (i=0; i < num_procs; i++) {
			if (i == my_rank) {
				printf("my rank %i, my coords %i, %i, my computations %i\n", my_rank, my_coord[0], my_coord[1], (stop[0] - start[0]) * (stop[1] - start[1]));
				mpPrintGrid(uold, my_size[0]+2, my_size[1]+2);
			}
			fflush(stdout);
			MPI_Barrier(CART_COMM);
		}
#endif
	}

#endif

	MPI_Barrier(CART_COMM);



	t_t0 = MPI_Wtime();

#if RMA

	MPI_Win_free(&WINold);
	MPI_Win_free(&WINnew);

	MPI_Free_mem(unew);

	if (0 == my_rank) {
		MPI_Alloc_mem(size * size * sizeof(double), MPI_INFO_NULL, &uall);
		memset(uall, 0, size * size * sizeof(double));
	}

#else

	free(unew);

	if (0 == my_rank) uall = (double *) calloc(size*size, sizeof(double));

#endif

	t_t1 = MPI_Wtime();

	MPI_Gatherv(uold + my_size[1]+3, 1, MID_DOUBLE, 
				uall, all_sizes, all_offsets, BLOCK_DOUBLE, 0, CART_COMM);

	t_t2 = MPI_Wtime();

	t[2] += t_t1 - t_t0;
	t[3] += t_t2 - t_t1;

#if RMA

	MPI_Free_mem(uold);

#if !FENCE
	MPI_Group_free(&group);
#endif

#else

	free(uold);

#endif



#if TIME_REDUCE
	MPI_Reduce(t, tr, 4, MPI_DOUBLE, MPI_SUM, 0, CART_COMM);
	if (0 == my_rank) {
		printf("Averaging times from all processes\n");
		for(i=1;i<4;i++)
			t[i] = tr[i] / (double)num_procs;
	}
#endif


	if (0 == my_rank) {
#if DEBUG
		printf("my rank %i, my coords %i, %i\n", my_rank, my_coord[0], my_coord[1]);
		mpPrintGrid(uall, size, size);
		fflush(stdout);
#endif

		printf("scatter time %.9lf\n", ts);

		//printf("results for size %i test with %i cycles %i num procs\n", size, numCycles, num_procs);
		printf("total time, communication, computation, gather (mem, comm)\n%.9lf\n%.9lf\n%.9lf\n%.9lf (%.9lf, %.9lf)\n", t[0]+t[1]+t[2]+t[3], t[0], t[1], t[2]+t[3], t[2], t[3]);
		
		prt_time();

#if DUMPGRID
		dumpGrid(uall, size);
#endif

#if CHECK
		checkGrid(argc, argv, uall);
#endif

		free(all_offsets);
		free(all_sizes);
#if RMA
		MPI_Free_mem(uall);
#else
		free(uall);
#endif
	}


	MPI_Type_free(&COL_DOUBLE);
	MPI_Type_free(&BLOCK_DOUBLE);
	MPI_Type_free(&MID_DOUBLE);
	MPI_Comm_free(&CART_COMM);
	MPI_Finalize();

}

