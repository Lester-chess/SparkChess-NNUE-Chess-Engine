CXX ?= g++
CXXFLAGS = -std=c++20 -O3 -mavx2 -mbmi2 -mpopcnt -flto -DNDEBUG -DUSE_AVX2 -pthread
TARGET = sparkchess

all: $(TARGET)

$(TARGET): main.cpp position.hpp movegen.hpp nnue.hpp search.hpp uci.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
