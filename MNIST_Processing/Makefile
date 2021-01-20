CC = g++
CFILES = SFMLTest.o
LIBS = -lsfml-graphics -lsfml-window -lsfml-system 
EXEC_FILE = SFMLTest.out
NETWORK_FILES_C = ../src/Network.cpp ../src/Layer.cpp
NETWORK_FILES_O = Network.o Layer.o 

MNISTImageReader.out: MNISTImageReader.o
	$(CC) -o MNISTImageReader.out MNISTImageReader.o $(LIBS)

MNISTImageReader.o: MNISTImageReader.cpp
	$(CC) -c MNISTImageReader.cpp 

MNISTLabelReader.out: MNISTLabelReader.o
	$(CC) -o MNISTLabelReader.out MNISTLabelReader.o $(LIBS)

MNISTLabelReader.o: MNISTLabelReader.cpp
	$(CC) -c MNISTLabelReader.cpp 

SQLV1: SQLTests.cpp
	$(CC) -c SQLTests.cpp
	$(CC) -o SQL.out SQLTests.o -lsqlite3

SQL: NetworkSaver.cpp
	$(CC) -c Driver.cpp NetworkSaver.cpp ../src/Network.cpp ../src/Layer.cpp
	$(CC) -o NetworkSaver.out Driver.o NetworkSaver.o Network.o Layer.o -lsqlite3

MNISTTrainer:MNISTTrainer.cpp
	$(CC) -c  Driver.cpp MNISTTrainer.cpp $(NETWORK_FILES_C) ../FileStorage/NetworkSaver.cpp
	$(CC) -o MNISTTrainer.out MNISTTrainer.o $(NETWORK_FILES_O)  Driver.o NetworkSaver.o -lsqlite3

TestTrainer: TestTrainer.cpp ../src/Layer.cpp ../src/Network.cpp ../FileStorage/NetworkSaver.cpp
	$(CC) -c  TestTrainer.cpp -o bin/TestTrainer.o 
	$(CC) -c  ../FileStorage/NetworkSaver.cpp -o bin/NetworkSaver.o
	$(CC) -c  ../src/Network.cpp -o bin/Network.o
	$(CC) -c  ../src/Layer.cpp -o bin/Layer.o
	$(CC) -c  MNISTTrainerV2.cpp -o bin/MNISTTrainerV2.o
	$(CC) -o TestTrainer bin/MNISTTrainerV2.o bin/Network.o bin/Layer.o  bin/TestTrainer.o bin/NetworkSaver.o -lsqlite3	-lpthread

TestDriver: MNISTTrainerV2.cpp TestDriver.cpp ../src/Layer.cpp ../src/Network.cpp ../FileStorage/NetworkSaver.cpp
	$(CC) -c  TestDriver.cpp -o bin/TestDriver.o 
	$(CC) -c  MNISTTester.cpp -o bin/MNISTTester.o 
	$(CC) -c  MNISTTrainerV2.cpp -o bin/MNISTTrainerV2.o 
	$(CC) -c  ../FileStorage/NetworkSaver.cpp -o bin/NetworkSaver.o
	$(CC) -c  ../src/Network.cpp -o bin/Network.o
	$(CC) -c  ../src/Layer.cpp -o bin/Layer.o
	$(CC) -o TestDriver bin/MNISTTester.o bin/MNISTTrainerV2.o bin/Network.o bin/Layer.o  bin/TestDriver.o bin/NetworkSaver.o -lsqlite3

OptimizedNetwork:
	$(CC) -c  Driver.cpp   $(NETWORK_FILES_C) 
	$(CC) -o Optimized.out $(NETWORK_FILES_O) Driver.o 

TableExistsTest:Driver.cpp
	$(CC) -c  Driver.cpp
	$(CC) -o TableExists.out  Driver.o -lsqlite3

clean:
	rm -f MNISTImageReader.o MNISTImageReader.out MNISTLabelReader.o MNISTLabelReader.out a.out new.db SQL.out SQLTests.o Network.o NetworkSaver.o 
	rm -f MNISTTrainer.out $(NETWORK_FILES_O) MNISTTrainer.o Driver.o Trainer.o Optimized.out MNISTTrainerV2.o MNISTTrainerV2.out
	rm -f TestDriver TestTrainer bin/MNISTTrainerV2.o bin/Network.o bin/Layer.o  bin/TestDriver.o bin/NetworkSaver.o TestTrainer.o TestTrainer