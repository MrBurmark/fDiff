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

int main(int arg, char **argv) {
  int width;
  int numCycles;
  int i, j, n;
  double *u0, *u1, *tptr;
  double inTemp;
  int cycle = 0;
  int numInit;

  FILE *fp;

  fp = fopen(argv[1], "r");

  fscanf(fp, "%d", &numCycles);
  fscanf(fp, "%d", &width);
  fscanf(fp, "%d", &numInit);
  printf("# cycles %d width %d # initializations %d\n", numCycles, width, numInit);

  u0 = calloc(width * width, sizeof(double));
  u1 = calloc(width * width, sizeof(double));

  initGrid(u0, u1, width);

  for (n=0; n<numInit; n++) {
    fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
    dataAt(u1, i, j, width) = inTemp;
  }
  
  //printGrid(u1, width);

  for (cycle=0; cycle<numCycles; cycle++) {
    updateGrid(u0, u1, width);
    //printGrid(u0, width);
    tptr = u0;
    u0 = u1;
    u1 = tptr;
  }

  dumpGrid(u1, width);

}

