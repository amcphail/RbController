#ifndef __AVHM_FIFO_H__
#define __AVHM_FIFO_H__

//#define FREQUENCY 75
#define FREQUENCY 1000
#define LARGEST_CHUNK_SIZE 524288
//#define LARGEST_CHUNK_SIZE 240000

#define BANKS 24

void errorString(long error, char* str);
long query(int card,unsigned long queryCode,unsigned long parameter); 
long writeChannel(int card,unsigned long channel, long value);
long start(int card);
long stop(int card);

long run(int card, unsigned long samples, long* data);

#endif
