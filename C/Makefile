CC = gcc 

all: gf16_gauss.o gf16.o algorithm1.o algorithm2.o 
	gcc -pthread algorithm2.o gf16_gauss.o gf16.o -o algorithm2
	gcc -pthread algorithm1.o gf16_gauss.o gf16.o -o algorithm1

gf16_gauss.o: gf16_gauss.c
	gcc -c gf16_gauss.c 

gf16.o:gf16.c 
	gcc -c gf16.c 
	
algorithm1.o: algorithm1.c
	gcc -c algorithm1.c
	
algorithm2.o: algorithm2.c
	gcc -c algorithm2.c

clean: 
	rm -f gf16.o gf16_gauss.o algorithm1.o algorithm2.o algorithm1 algorithm2

