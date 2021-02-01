SRC = src
INCLUDE = include
BIN = ./bin
BINNAME = bin
DIST = ./dist

.PHONY: all
all: $(INCLUDE) $(BIN) $(DIST)
	cd $(SRC) && make

$(BIN):
	mkdir $(BIN)

$(DIST):
	mkdir $(DIST)

.PHONY: run
run:
	make
	$(BIN)/$(BINNAME)

.PHONY: clean
clean:
	rm -rf $(BIN)
	rm -rf $(DIST)
