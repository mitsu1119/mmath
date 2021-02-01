CC = g++
CXXFLAGS = -c -Wall -Wextra -O2 -std=c++17 -I ../include
BIN = ../bin
BINNAME = bin
DIST = ../dist

SRCS = test.cpp 
OBJS = $(DIST)/$(SRCS:.cpp=.o)

.PHONY: all
all: $(SRCS) $(BIN)/$(BINNAME)

$(BIN)/$(BINNAME): $(OBJS)
	$(CC) $(OBJS) -o $(BIN)/$(BINNAME)

$(DIST)/%.o: %.cpp $(DIST)
	$(CC) $(CXXFLAGS) $*.cpp -o $(DIST)/$*.o
