#!/bin/bash

./build/bin/tests/work1_tests
make ./build/run_benchmarks_json
python3 ./plots/plots.py