CC = nvcc
CXXFLAGS = --compiler-options -Wall --compiler-options -Wextra -O2 --expt-extended-lambda -std=c++17 -I ../include -dc
BIN = ../bin
BINNAME = libmmath.a
DIST = ../dist

SRCS = Int.cu Digits.cu
OBJS = $(subst .o ,.o $(DIST)/,$(DIST)/$(SRCS:%.cu=%.o))

.PHONY: all
all: $(SRCS) $(BIN)/$(BINNAME)

$(BIN)/$(BINNAME): $(OBJS)
	$(CC) -o $@ -lib $^

$(DIST)/%.o: %.cu
	$(CC) -c $(CXXFLAGS) $*.cu -o $(DIST)/$*.o
	$(CC) -dlink $(CXXFLAGS) $(DIST)/$*.o -o $(DIST)/$*.o
