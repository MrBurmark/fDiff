#!/bin/bash

mpicc mpFdiff.c Fdutils.c -O3 -o mpFd -lm

for file in G1202 G2402 G4802; do
	echo running test $file
	for p in 1 4 9 16; do
		echo running test with $p procs
		for i in 1 2 3 4 5; do
			echo test $i
			mpirun -n $p ./mpFd $file >> $file-$p.log
		done
	done
done

for file in G1602 G3202; do
	echo running test $file
	for p in 1 4 16; do
		echo running test with $p procs
		for i in 1 2 3 4 5; do
			echo test $i
			mpirun -n $p ./mpFd $file >> $file-$p.log
		done
	done
done

