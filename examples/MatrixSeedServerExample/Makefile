CXX = g++
CXXFLAGS = -O2 -g -Wall -std=c++17
LIBS = -pthread -lzmq

all: main.cpp
	$(CXX) $(CXXFLAGS) -o MatrixSeedServerExample main.cpp $(LIBS)

clean:
	$(RM) MatrixSeedServerExample
