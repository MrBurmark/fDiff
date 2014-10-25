#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Fdiff.h"

/*

	 tu[] is input grid
	 u[] is output grid
	 w is width of grid

	 compute 4-nearest neighbor updates 

*/

void updateGrid(double u[], double tu[], int w) {
	int i, j;
	double uv2;

	for (i=1; i<w-1; i++) {
		for (j=1; j<w-1; j++) {

			dataAt(u, i, j, w) = .25 * (dataAt(tu, i+1, j, w)
				+ dataAt(tu, i-1, j, w)
				+ dataAt(tu, i, j+1, w)
				+ dataAt(tu, i, j-1, w));
		}
	}
}

void mpUpdateGrid(double u[], double tu[], int w, int start0, int stop0, int start1, int stop1) {
	int i, j;

	for (i=start0; i < stop0; i++) {
		for (j=start1; j < stop1; j++) {

			dataAt(u, i, j, w) = .25 * (dataAt(tu, i+1, j, w)
				+ dataAt(tu, i-1, j, w)
				+ dataAt(tu, i, j+1, w)
				+ dataAt(tu, i, j-1, w));
		}
	}
}

void printGrid(double g[], int w) {
	int i, j;

	for (i=0; i<w; i++) {
		for (j=0; j<w; j++) {
			printf("%7.3f ", dataAt(g, i, j, w));
		}
		printf("\n");
	}
}

void mpPrintGrid(double g[], int h, int w) {
	int i, j;

	for (i=0; i<h; i++) {
		for (j=0; j<w; j++) {
			printf("%7.3f ", dataAt(g, i, j, w));
		}
		printf("\n");
	}
}

void dumpGrid(double g[], int w) {
	int i, j;
	FILE *fp;

	fp = fopen("dump.out", "w");

	for (i=0; i<w; i++) {
		for (j=0; j<w; j++) {
			fprintf(fp, "%f ", dataAt(g, i, j, w));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void initGrid(double u0[], double u1[], int w) {
	int i, j;

	for (i=0; i<w; i++) {
		for (j=0; j<w; j++) {
			dataAt(u0, i, j, w) = 0.;
			dataAt(u1, i, j, w) = 0.;
		}
	}
}

int checkGrid(int argc, char **argv, double* uall) {
	int width;
	int numCycles;
	int ok = 1;
	int i, j, n;
	double *u0, *u1, *tptr;
	double inTemp, diff;
	int cycle = 0;
	int numInit;

	FILE *fp;

	fp = fopen(argv[1], "r");

	ok = fscanf(fp, "%d", &numCycles);
	ok = fscanf(fp, "%d", &width);
	ok = fscanf(fp, "%d", &numInit);
	u0 = calloc(width * width, sizeof(double));
	u1 = calloc(width * width, sizeof(double));

	for (n=0; n<numInit; n++) {
		ok = fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
		dataAt(u1, i, j, width) = inTemp;
	}
	fclose(fp);

	for (cycle=0; cycle<numCycles; cycle++) {
		updateGrid(u0, u1, width);
		tptr = u0;
		u0 = u1;
		u1 = tptr;
	}

	for (i=0; i<width; i++) {
		for (j=0; j<width; j++) {
			diff = dataAt(u1, i, j, width) - dataAt(uall, i, j, width);
			if (fabs(diff) > THRESHOLD){
				ok == 0;
				if (DEBUG)
					printf("Error at %i, %i: difference of %.9lf\n", i, j, diff);
			}
		}
	}

	if (ok) {
		printf("All solutions within error threshold\n");
	} else {
		printf("All solutions not within error threshold\n");
	}
	return ok;
}

