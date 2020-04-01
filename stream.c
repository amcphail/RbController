#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <edre/edrapi.h>

#define FREQUENCY 10000
#define LARGEST_CHUNK_SIZE 524288

#define BANKS 24

#define SnDAC0 (1000034286)

int main(int argc, char* argv[])
{
  char filename[256];
  strcpy(filename,argv[1]);

  char* err_str;

  unsigned long channels;
  unsigned long samples;

  long sample;

  long* buffer;

  int i, j;


  FILE* file = fopen(filename,"r");

  if (!file) {
    err_str = strerror(errno);
    printf("Error opening file: %s.\n",err_str);
    return errno;
  }

  fscanf(file,"%lu\t%lu\n",&channels,&samples);

  if (channels < 24) {
    printf("Insufficient number of channels: %lu, need 24.\n",channels);
  }

  buffer = (long*)malloc(channels*samples*sizeof(long));

  for (i = 0; i < samples; i++) {
    for (j = 0; j < channels; j++) {
      fscanf(file,"%ld",&sample);
      buffer[i*channels+j] = sample;
      if (j + 1 == channels) {
	fscanf(file,"\n");
      }
      else {
	fscanf(file,"\t");
      }
    }
  }
    
  fclose(file);

  int left = samples;

  long result = EDRE_DAControl(SnDAC0,0,2);
  if (result) {
    char err[256];
    EDRE_StrError(result,err);
    printf("Stop failed: %ld, %s\n",result,err);
  }

  struct timespec tim_req, tim_res;

  int cursor = 0;

  int size;
  int space;

  if ((samples*BANKS) < LARGEST_CHUNK_SIZE) {
    size = samples;
  }
  else {
    size = LARGEST_CHUNK_SIZE/BANKS;
  }

  result = EDRE_DAConfig(SnDAC0,63,FREQUENCY,0,1,0,size*BANKS,buffer);
  if (result) {
    char err[256];
    EDRE_StrError(result,err);
    printf("Update failed: %ld, %s\n",result,err);
  }

  cursor += size*BANKS;
  left -= size;
  
  result = EDRE_DAControl(SnDAC0,0,1);

  tim_req.tv_sec = 1;
  tim_req.tv_nsec = 0;

  while (cursor < samples*BANKS) {
    
    nanosleep(&tim_req,&tim_res);
  
    space = EDRE_Query(SnDAC0,205,0);

    space /= 2;

    if (space > 0) {
      
      if ((left*BANKS) < space) {
	size = left;
      }
      else {
	size = space/BANKS;
      }

      if (size) {
	result = EDRE_DAUpdateData(SnDAC0,63,size*BANKS,&buffer[cursor]);
	if (result) {
	  char err[256];
	  EDRE_StrError(result,err);
	  printf("Update failed: %ld, %s\n",result,err);
	}
	else {
	  cursor += size*BANKS;
	  left -= size;
	}
      }
      else {
	size = 1;
	result = EDRE_DAUpdateData(SnDAC0,63,space,&buffer[cursor]);
	if (result) {
	  char err[256];
	  EDRE_StrError(result,err);
	  printf("Update failed: %ld, %s\n",result,err);
	}
	else {
	  cursor += space*BANKS;
	  left -= space;
      }
      }
    }
  }

  result = EDRE_WaitOnInterrupt(SnDAC0);

  result = EDRE_DAControl(SnDAC0,0,2);
  
  free(buffer);

  return result;
}
