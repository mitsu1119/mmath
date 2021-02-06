CC = nvcc
CXXFLAGS = -c --compiler-options -Wall --compiler-options -Wextra -O2 -std=c++17 -I ../include
BIN = ../bin
BINNAME = bin
DIST = ../dist

SRCS = Int.cu test.cu
OBJS = $(subst .o ,.o $(DIST)/,$(DIST)/$(SRCS:%.cu=%.o))

.PHONY: all
all: $(SRCS) $(BIN)/$(BINNAME)

$(BIN)/$(BINNAME): $(OBJS)
	$(CC) $(OBJS) -o $(BIN)/$(BINNAME)

$(DIST)/%.o: %.cu $(DIST)
	$(CC) $(CXXFLAGS) $*.cu -o $(DIST)/$*.o
