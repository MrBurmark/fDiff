#ifndef _FDIFF_H_
#define _FDIFF_H_

#define DEBUG 0
#define PRINT_CYCLES 2
#define COMM_COMP_RATIO 1.0e3
#define THRESHOLD 1.0e-6

#define dataAt(DATA, I, J, W) DATA[(I) * (W) + J]

void updateGrid(double *, double *, int);
void mpUpdateGrid(double u[], double tu[], int w, int start0, int stop0, int start1, int stop1);
void printGrid(double *, int);
void mpPrintGrid(double g[], int h, int w);
void printMid(double g[], int w, int r);
void initGrid(double [], double [], int);
int checkGrid(int argc, char **argv, double* uall);

#endif
