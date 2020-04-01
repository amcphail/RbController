#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include "fifo.h"

int main(int argc, char* argv[])
{
  int samples = 2097152;

  long* buffer = (long *)malloc(BANKS*samples*sizeof(long));

  double inc = 2*M_PI/FREQUENCY;

  int i, j;

  int sine = LARGEST_CHUNK_SIZE;
  
  for (i = 0; i < sine; i++) {
    for (j = 0; j < BANKS; j++) {
      buffer[i*BANKS+j] = 5000000/(j+1);
    }
  }

  for (i = sine; i < samples; i++) {
    for (j = 0; j < BANKS; j++) {
      buffer[i*BANKS+j] = round(sin(i*inc/(j+1))*5000000.0);
    }
  }

  long result = stop(0);

  printf("Stop result: %ld\n",result);

  result = writeChannel(0,0,10000000);

  printf("Write result: %ld\n",result);

  sleep(1);

  result = writeChannel(0,0,0);

  printf("Write result: %ld\n",result);

  sleep(1);

  result = writeChannel(0,0,-10000000);

  printf("Write result: %ld\n",result);

  if (result) {
    char err[256];
    errorString(result,err);
    printf("Write failed: %s\n",err);
  }

  result = query(0,200,0);
  printf("Number of channels: %ld\n",result);
  result = query(0,201,0);
  printf("Maximum output frequency: %ld\n",result);
  result = query(0,202,0);
  printf("D/A Subsystem busy: %ld\n",result);
  result = query(0,203,0);
  printf("FIFO Size: %ld\n",result);
  result = query(0,204,0);
  printf("Buffer Size: %ld\n",result);
  result = query(0,205,0);
  printf("Buffer space available: %lu\n",result);
  result = query(0,206,0);
  printf("Buffer underrun: %ld\n",result);

  result = run(0,samples,buffer);
  
  if (result) {
    char err[256];
    errorString(result,err);
    printf("Run failed: %s\n",err);
  }

  return result;
}
