CXX = g++
CXXFLAGS = -O2 -g -Wall -std=c++17
LIBS = -pthread -lzmq

all: main.cpp
	$(CXX) $(CXXFLAGS) -o LargeBlocksizeExample main.cpp $(LIBS)

clean:
	$(RM) LargeBlocksizeExample
