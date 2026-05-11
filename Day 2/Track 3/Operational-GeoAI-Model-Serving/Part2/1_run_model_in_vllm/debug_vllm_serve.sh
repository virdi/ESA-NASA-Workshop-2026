#!/bin/bash
# Debug script for vLLM serve with detailed logging

# Set debug environment variables
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1
export PYTHONUNBUFFERED=1

# Enable Python warnings
export PYTHONWARNINGS="default"

# Log file
LOG_FILE="vllm_serve_debug.log"

echo "Starting vLLM serve with debug logging..."
echo "Log file: $LOG_FILE"
echo "Model directory: ./ttexample/"
echo ""

# Run vLLM with debug output
vllm serve \
  --model ./ttexample/ \
  --enforce-eager \
  --skip-tokenizer-init \
  --enable-mm-embeds \
  --io-processor-plugin terratorch_segmentation \
  2>&1 | tee "$LOG_FILE"

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ vLLM serve failed with exit code: $EXIT_CODE"
    echo "Check $LOG_FILE for details"
    echo ""
    echo "Last 50 lines of log:"
    tail -n 50 "$LOG_FILE"
else
    echo ""
    echo "✅ vLLM serve started successfully"
fi

exit $EXIT_CODE

# Made with Bob
