#ifndef PRT_TIME_H
#define PRT_TIME_H

#include <stdio.h>
#include <time.h>

void prt_time(void){

	time_t timer;
    char buffer[100];

    time(&timer);

    strftime(buffer, 100, "%Y-%m-%d %H:%M:%S UTC", gmtime(&timer));
    puts(buffer);
}

#endif