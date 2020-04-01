# Application Makefile

#CFLAGS = -DLINUX -Wall -I/usr/include/edre
CFLAGS = -g -DLINUX -Wall -D_REENTRANT

TGT = main
LIBS = fifo.so 
OBJS = main.o 
CC = gcc

all: $(TGT) fifo.so

minimal: minimal.o
	$(CC) -o minimal minimal.o -l edreapi -l m

stream: stream.o
	$(CC) -o stream stream.o -l edreapi -l m

$(TGT): $(LIBS) $(OBJS)
	$(CC) -o $(TGT) $(OBJS) $(LIBS) -l edreapi -l m -l pthread

fifo.so: fifo.c
	$(CC) $(CFLAGS) fifo.c --shared -fPIC -o $@ -l edreapi -l m -l pthread

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@




clean:
	rm -f $(TGT) minimal *.o *.so
