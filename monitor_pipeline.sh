#!/bin/bash
# Monitor the benchmark pipeline progress

echo "Monitoring benchmark pipeline..."
echo "================================"
echo ""

# Check if log file exists
if [ ! -f "benchmark_pipeline.log" ]; then
    echo "Pipeline not started yet or log file not created."
    exit 1
fi

# Show last 50 lines
echo "Last 50 lines of output:"
echo "------------------------"
tail -n 50 benchmark_pipeline.log

echo ""
echo "================================"
echo "Pipeline statistics:"
echo "------------------------"

# Count completed benchmarks
completed=$(grep -c "Status: completed" benchmark_pipeline.log 2>/dev/null || echo "0")
errors=$(grep -c "Status: error" benchmark_pipeline.log 2>/dev/null || echo "0")
echo "Completed: $completed"
echo "Errors: $errors"

# Check if results file exists
if [ -f "benchmark_results.json" ]; then
    echo ""
    echo "Results file exists: benchmark_results.json"
    echo "File size: $(ls -lh benchmark_results.json | awk '{print $5}')"
fi
