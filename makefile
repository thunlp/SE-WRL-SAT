CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: SAT_new
	
SAT_new :
	$(CC) SAT_new.c -o SAT_new $(CFLAGS)
clean:
	rm -rf SAT