/*************************

   File: Fdiff.c
   Compile: gcc Fdiff.c Fdutils.c -O3 -o Fd -lm
   Use: ./Fd [input file]

   Performs 4 nearest neighbor updates on 2-D grid
   Input file format:

   # cycles
   width of grid (including boundary)
   # initial data points
   
   3 integers per data point: i and j indices, data


*************************/

#include <stdio.h>
#include <stdlib.h>
#include "Fdiff.h"

#define PRINT_CYCLES 2

int main(int arg, char **argv) {
  int width;
  int numCycles;
  int ok;
  int i, j, n;
  double *u0, *u1, *tptr;
  double inTemp;
  int cycle = 0;
  int numInit;

  FILE *fp;

  fp = fopen(argv[1], "r");

  ok = fscanf(fp, "%d", &numCycles);
  ok = fscanf(fp, "%d", &width);
  ok = fscanf(fp, "%d", &numInit);
  printf("# cycles %d width %d # initializations %d\n", numCycles, width, numInit);

  u0 = calloc(width * width, sizeof(double));
  u1 = calloc(width * width, sizeof(double));

  initGrid(u0, u1, width);

  for (n=0; n<numInit; n++) {
    ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
    dataAt(u1, i, j, width) = inTemp;
  }
  
  //printGrid(u1, width);

  for (cycle=0; cycle<numCycles; cycle++) {
    if (0 == cycle%PRINT_CYCLES)
      printf("cycle %i\n", cycle);

    updateGrid(u0, u1, width);
    //printGrid(u0, width);
    tptr = u0;
    u0 = u1;
    u1 = tptr;
  }

  printGrid(u1, width);
  dumpGrid(u1, width);

}

