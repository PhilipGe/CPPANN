CC = g++
CFILES = SFMLTest.o
LIBS = -lsfml-graphics -lsfml-window -lsfml-system 
EXEC_FILE = SFMLTest.out

MNISTImageReader.out: MNISTImageReader.o
	$(CC) -o MNISTImageReader.out MNISTImageReader.o $(LIBS)

MNISTImageReader.o: MNISTImageReader.cpp
	$(CC) -c MNISTImageReader.cpp 

MNISTLabelReader.out: MNISTLabelReader.o
	$(CC) -o MNISTLabelReader.out MNISTLabelReader.o $(LIBS)

MNISTLabelReader.o: MNISTLabelReader.cpp
	$(CC) -c MNISTLabelReader.cpp 

clean: MNISTImageReader.o
	rm -f MNISTImageReader.o MNISTImageReader.out MNISTLabelReader.o MNISTLabelReader.out