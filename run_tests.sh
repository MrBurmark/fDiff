#!/bin/bash

mpicc mpFdiff.c Fdutils.c -O3 -o mpFd -lm

for file in G1202 G1602 G2402 G3202 G4802; do
	for p in 1 4 16; do
		for i in 1 2 3 4 5; do
			mpirun -n $p ./mpFd $file >> test.log
		done
	done
done
