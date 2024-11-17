#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <random>
#include <cmath>

#include "Value.h"
#include "tests.cpp"
#include "examples.cpp"
#include "MLP.h"
#include "utils.h"


int main() {
    binary_classification();
    return 0;
};


/* 


g++ -std=c++14 -c Value.cpp -o Value.o && g++ -std=c++14 -c main.cpp -o main.o && g++ -std=c++14 Value.o main.o -o main && ./main

g++ -std=c++14 -c Value.cpp -o Value.o && \
g++ -std=c++14 -c MLP.cpp -o MLP.o && \
g++ -std=c++14 -c utils.cpp -o utils.o && \
g++ -std=c++14 -c main.cpp -o main.o && \
g++ -std=c++14 -c tests.cpp -o tests.o && \
g++ -std=c++14 Value.o MLP.o utils.o main.o tests.o -o main && ./main
*/


