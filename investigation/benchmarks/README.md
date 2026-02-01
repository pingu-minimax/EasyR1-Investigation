# Protocol Performance Benchmarks

## Purpose

This directory contains benchmarks for measuring the performance impact of protocol changes.

## Benchmark Suite

### 1. Data Transfer Benchmark

Measures the impact of `non_blocking` parameter on data transfer speeds.

### 2. Batch Processing Benchmark

Compares throughput with different batch size configurations.

### 3. Serialization Benchmark

Evaluates overhead from `TensorDict.consolidate()` calls.

## Running Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks
python run_benchmarks.py

# Run specific benchmark
python run_benchmarks.py --benchmark data_transfer
```

## Results

Benchmark results will be stored in `results/` directory.
