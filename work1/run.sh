#!/bin/bash

./build/bin/tests/work1_tests
cd ./build
make run_benchmarks_json
cd ..
python3 ./plots/plots.py