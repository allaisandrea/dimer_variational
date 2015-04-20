local: CXX = mpiCC
local: CXXFLAGS = -g 
local: LIBS = -llapack -lblas -lgsl -lcblas
local: OUTPUT = ../debug/run

cluster: CXX = mpiCC
cluster: LIBDIR = /n/home06/allais/lib
cluster: INCLUDEDIR = /n/home06/allais/include
cluster: CXXFLAGS = -D ARMA_NO_DEBUG -I $(INCLUDEDIR)
cluster: LIBS = -llapack -lgsl -lgslcblas
cluster: OUTPUT = ./run

OBJECTS = main.o data_structures.o states.o monte_carlo.o \
	linear_algebra.o observables.o measure_drivers.o \
	single_point_drivers.o minimization_gradient_drivers.o

local: $(OBJECTS) $(OBJECTS1)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(OUTPUT) $(LIBS)
main.o: main.cpp test_minimization_gradient.cpp minimization_gradient.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o
cluster:$(OBJECTS)
	$(CXX) $(CXXFLAGS) -L $(LIBDIR) $(OBJECTS) -o $(OUTPUT) $(LIBS)
clean:
	rm *.o $(OUTPUT)
