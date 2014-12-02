CC	= g++
ORIGINALCFLAGS	= -Wall -O3
CFLAGS = -Wall -O3

all:
	make clean
	make prog

clean:
	rm -f libRF.a
	rm -f *.o
	rm -f prog
	
ClassifierRF.o: libRF/ClassifierRF.cpp libRF/ClassifierRF.h libRF/Classifier.h libRF/Features.h
	$(CC) libRF/ClassifierRF.cpp -c $(CFLAGS)

FeaturesTable.o: libRF/FeaturesTable.cpp libRF/FeaturesTable.h libRF/Features.h
	$(CC) libRF/FeaturesTable.cpp -c $(CFLAGS)

Instantiation.o: libRF/Instantiation.cpp libRF/FeaturesTable.h libRF/Features.h libRF/ClassifierRF.h libRF/Classifier.h libRF/Features.h
	$(CC) libRF/Instantiation.cpp -c $(CFLAGS)

libRF.a: ClassifierRF.o FeaturesTable.o Instantiation.o
	ar rcs libRF.a ClassifierRF.o FeaturesTable.o Instantiation.o

prog: Test_libRF/Test_libRF.cpp libRF.a
	$(CC) Test_libRF/Test_libRF.cpp -o prog $(CFLAGS) -L. -lRF
