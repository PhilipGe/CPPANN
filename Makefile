CC = g++
CFLAGS = -c -Wall -o
CFILES = bin/main.o bin/Layer.o
EXEC_FILE = output/main

all:output/main

output/main: $(CFILES)
	$(CC) -o $(EXEC_FILE) $(CFILES)

bin/main.o:src/main.cpp
	$(CC) $(CFLAGS) bin/main.o src/main.cpp 

bin/Layer.o: src/Layer.cpp
	$(CC) $(CFLAGS) bin/Layer.o src/Layer.cpp 

run: $(EXEC_FILE)
	./$(EXEC_FILE)

clean:
	rm -f $(CFILES) $(EXEC_FILE)
