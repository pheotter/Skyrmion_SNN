.PHONY: all
CC = g++
SRC = ./src/*.cpp
BIN = ./bin
TEST = ./test/ut_all.cpp
CFLAG = -Wfatal-errors
LIB = -lgtest -lpthread

#all: target
all:clean what $(BIN)/ut_all

#target: $(SRC) what
#	$(CC) $(SRC) -o $(BIN)/stringExample

$(BIN)/ut_all: $(TEST) ./src/sky.cpp
	$(CC) $(TEST) ./src/sky.cpp -o $@ $(LIB)

what:
	mkdir -p bin obj

.PHONY: clean
clean:
	rm -f bin/*
	rm -f obj/*
