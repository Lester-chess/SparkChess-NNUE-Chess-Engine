#!/usr/bin/env bash
set -e
echo "Compiling SparkChess with GCC/Clang C++20 AVX2..."
g++ -std=c++20 -O3 -mavx2 -mbmi2 -mpopcnt -flto -DNDEBUG -DUSE_AVX2 -pthread main.cpp -o sparkchess
echo "[SUCCESS] Compiled sparkchess successfully!"
