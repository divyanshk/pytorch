#!/bin/bash

# Dataloader Benchmark Comparison Script
# Compares threading vs multiprocessing across different batch sizes

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/dataloader_benchmark.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Default parameters (can be overridden via command line)
DATASET_TYPE="${DATASET_TYPE:-local}"
DATA_PATH="${DATA_PATH:-train/}"
HF_DATASET_NAME="${HF_DATASET_NAME:-}"
NUM_WORKERS_LIST="${NUM_WORKERS_LIST:-4 8 16}"
MAX_BATCHES="${MAX_BATCHES:-100}"
DATASET_COPIES="${DATASET_COPIES:-1}"
BATCH_SIZES="${BATCH_SIZES:-8 64 128 256 512 1024 2048}"
WORKER_METHODS="${WORKER_METHODS:-multiprocessing thread}"

# Print configuration
echo "=========================================="
echo "Dataloader Benchmark Comparison"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Results directory: ${RESULTS_DIR}"
echo "Dataset type: ${DATASET_TYPE}"
if [ "${DATASET_TYPE}" = "local" ]; then
    echo "Data path: ${DATA_PATH}"
else
    echo "HuggingFace dataset: ${HF_DATASET_NAME}"
fi
echo "Number of workers to test: ${NUM_WORKERS_LIST}"
echo "Max batches per run: ${MAX_BATCHES}"
echo "Batch sizes to test: ${BATCH_SIZES}"
echo "Worker methods: ${WORKER_METHODS}"
echo "=========================================="
echo ""

# Validate dataset configuration
if [ "${DATASET_TYPE}" = "local" ] && [ -z "${DATA_PATH}" ]; then
    echo "ERROR: DATA_PATH must be set for local dataset type"
    echo "Usage: DATA_PATH=/path/to/dataset ./run_comparison.sh"
    exit 1
fi

if [ "${DATASET_TYPE}" = "huggingface" ] && [ -z "${HF_DATASET_NAME}" ]; then
    echo "ERROR: HF_DATASET_NAME must be set for huggingface dataset type"
    echo "Usage: HF_DATASET_NAME=imagenet-1k DATASET_TYPE=huggingface ./run_comparison.sh"
    exit 1
fi

# Summary file
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
echo "Dataloader Benchmark Summary - ${TIMESTAMP}" > "${SUMMARY_FILE}"
echo "==========================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Run benchmarks
for worker_method in ${WORKER_METHODS}; do
    echo ""
    echo "=========================================="
    echo "Testing worker_method: ${worker_method}"
    echo "=========================================="

    for num_workers in ${NUM_WORKERS_LIST}; do
        echo ""
        echo "Testing with ${num_workers} workers"
        echo "------------------------------------------"

        for batch_size in ${BATCH_SIZES}; do
            echo ""
            echo "Running: worker_method=${worker_method}, num_workers=${num_workers}, batch_size=${batch_size}"
            echo "------------------------------------------"

            LOG_FILE="${RESULTS_DIR}/${worker_method}_workers${num_workers}_batch${batch_size}.log"

            factor=$(( ((batch_size * MAX_BATCHES) / 150000) + 1))
            # Build command based on dataset type
            CMD="python ${BENCHMARK_SCRIPT} \
                --dataset_type ${DATASET_TYPE} \
                --batch_size ${batch_size} \
                --num_workers ${num_workers} \
                --max_batches ${MAX_BATCHES} \
                --worker_method ${worker_method} \
                --dataset_copies $((DATASET_COPIES * factor))"

            if [ "${DATASET_TYPE}" = "local" ]; then
                CMD="${CMD} --data_path ${DATA_PATH}"
            elif [ "${DATASET_TYPE}" = "huggingface" ]; then
                CMD="${CMD} --hf_dataset_name ${HF_DATASET_NAME}"
            fi

            # Run benchmark and save output
            ${CMD} 2>&1 | tee "${LOG_FILE}"

            # Extract key metrics from log file
            echo "" >> "${SUMMARY_FILE}"
            echo "worker_method=${worker_method}, num_workers=${num_workers}, batch_size=${batch_size}" >> "${SUMMARY_FILE}"
            echo "----------------------------------------" >> "${SUMMARY_FILE}"
            grep -E "(Memory increase:|Samples per second \(overall throughput\):|Average data loading time:)" "${LOG_FILE}" >> "${SUMMARY_FILE}" || true
            echo "" >> "${SUMMARY_FILE}"

            echo "Results saved to: ${LOG_FILE}"
            echo ""

            # Small delay between runs
            sleep 2
        done
    done
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo ""
echo "Summary:"
echo "----------------------------------------"
cat "${SUMMARY_FILE}"
echo ""
echo "Compare memory usage:"
echo "  grep 'Memory increase' ${RESULTS_DIR}/*.log"
echo ""
echo "Compare speed:"
echo "  grep 'Samples per second (overall throughput)' ${RESULTS_DIR}/*.log"
