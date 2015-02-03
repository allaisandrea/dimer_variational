local: CXX = g++
local: CXXFLAGS = -g
local: LIBS = -llapack -lblas -lgsl -lcblas
local: OUTPUT = debug/run

cluster: CXX = g++
cluster: LIBDIR = /n/home06/allais/lib
cluster: INCLUDEDIR = /n/home06/allais/include
cluster: CXXFLAGS = -O3 -D ARMA_NO_DEBUG -I $(INCLUDEDIR)
cluster: LIBS = -llapack -lgsl -lgslcblas
cluster: OUTPUT = ./run

OBJECTS = main.o data_structures.o states.o

local: $(OBJECTS) $(OBJECTS1)
	g++ $(CXXFLAGS) $(OBJECTS) -o $(OUTPUT) $(LIBS)
main.o: main.cpp tests.cpp
	g++ $(CXXFLAGS) -c main.cpp -o main.o
cluster:$(OBJECTS)
	g++ $(CXXFLAGS) -L $(LIBDIR) $(OBJECTS) -o $(OUTPUT) $(LIBS)
clean:
	rm *.o $(OUTPUT)
