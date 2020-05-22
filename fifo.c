#include <edre/edrapi.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "fifo.h"

#define SnDAC0 (1000034286)

long config(int card, unsigned long samples, long* data);
long update(int card, unsigned long samples, long* data);
long waitOnInterrupt(int card);

struct arg_struct {
  int card;
  unsigned long samples;
  long* data;
};


void errorString(long error, char* buffer)
{
  EDRE_StrError(error,buffer);
}


long query(int card,unsigned long queryCode,unsigned long parameter)
{
  long result;

  if (card != 0) return -1;

  result = EDRE_Query(SnDAC0,queryCode,parameter);

  return result;
}

long writeChannel(int card,unsigned long channel, long value)
{
  if (card != 0) return -1;

  long result = EDRE_DAWrite(SnDAC0,channel,value);

  return result;
}

long config(int card, unsigned long samples, long* data)
{
  if (card != 0) return -1;

  long result = EDRE_DAConfig(SnDAC0,63,FREQUENCY,0,1,0,samples*BANKS,data);

  /*
  int i, j;

  for (i = 0; i < samples; i++) {
    for (j = 0; j < BANKS; j++) {
      printf("%d\t",data[i*BANKS+j]);
    }
    printf("\n");
  }
  */

  return result;
}

long start(int card)
{
  if (card != 0) return -1;

  long result = EDRE_DAControl(SnDAC0,0,1);

  return result;
}

long stop(int card)
{
  if (card != 0) return -1;

  long result = EDRE_DAControl(SnDAC0,0,2);

  return result;
}

long update(int card, unsigned long samples, long* data)
{
  if (card != 0) return -1;

  long result = EDRE_DAUpdateData(SnDAC0,63,samples*BANKS,data);

  return result;
}

void* doRun(void* arguments)
{
  struct arg_struct *args = (struct arg_struct *)arguments;

  long* data = args->data;

  int left = args->samples;

  struct timespec tim_req, tim_res;
  tim_req.tv_sec = 1;
  tim_req.tv_nsec = 0;

  int cursor = 0;

  long size;
  long space;

  printf("fifo.so: doRun(), entered.\n");

  printf("fifo.so: doRun(), card: %d, samples: %d.\n",args->card, args->samples);

  //  if ((samples*BANKS) < LARGEST_CHUNK_SIZE/2) {
  if ((args->samples*BANKS) < LARGEST_CHUNK_SIZE) {
    size = args->samples;
  }
  else {
    //    size = LARGEST_CHUNK_SIZE/BANKS/2;
    size = LARGEST_CHUNK_SIZE/BANKS;
  }
  
  size = (size / BANKS) * BANKS;
  
  long result = stop(args->card);

  if (result) {
    char err[256];
    errorString(result,err);
    printf("Stop failed: %s.\n",err);
  }

  result = query(args->card,205,0);

  printf("Buffer space: %ld\n",result);

  result = config(args->card,size,data);
  cursor += size*BANKS;
  left -= size;

  if (result) {
    char err[256];
    errorString(result,err);
    printf("Configuration failed: %s.\n",err);
  }

  printf("fifo.so: doRun(), configured.\n");

  result = start(args->card);

  if (result) {
    char err[256];
    errorString(result,err);
    printf("Start failed: %s.\n",err);
  }

  printf("fifo.so: doRun(), started.\n");

  printf("fifo.so: doRun(), cursor: %d, samples: %d.\n",cursor,args->samples);


  while (cursor < args->samples*BANKS) {
    
    printf("fifo.so: doRun(), while loop.\n");


    nanosleep(&tim_req,&tim_res);
  
    space = query(args->card,205,0);

//    space /= 2;
    if (space > LARGEST_CHUNK_SIZE) {
      space = LARGEST_CHUNK_SIZE;
    }

    space = (space / BANKS) * BANKS;

    if (space > 0) {

      if (left*BANKS < space) {
	size = left;
      }
      else {
	size = space/BANKS;
      }

      if (size) {
	
	result = update(args->card,size,&data[cursor]);
      
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

	result = update(args->card,size,&data[cursor]);

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
    }
    else {
      printf("No space available yet.\n");
    }
  }
   
  printf("About to wait for interrupt.\n");

  result = waitOnInterrupt(args->card);

  printf("Received interrupt and continuing.\n");

  result = stop(args->card);

  printf("Stopped.\n");

  pthread_exit((void*)result);
}

long run(int card, unsigned long samples, long* data)
{
  pthread_t thread;
  struct arg_struct args;

  args.card = card;
  args.samples = samples;
  args.data = data;
  
  int i;

  printf("fifo.so: run(): samples: %d.\n",samples);

  int rc;

  rc = pthread_create(&thread,NULL,&doRun,(void *)(&args));
  rc = pthread_join(thread,NULL);

  printf("fifo.so: run(): pthread exited.\n");

  return 0;
}

long waitOnInterrupt(int card)
{
  long result;

  if (card != 0) return -1;

  result = EDRE_WaitOnInterrupt(SnDAC0);

  return result;
}
