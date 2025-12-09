#!/bin/bash

# Script to run autonomous latency analysis using the unified latency_analyzer agent
# This uses a single comprehensive query that lets the agent make intelligent decisions


# Suppress experimental feature warnings
export PYTHONWARNINGS="ignore::UserWarning"
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables
if [ -f "${SCRIPT_DIR}"/.env ]; then
    echo "Loading .env file..."
    set -a
    source "${SCRIPT_DIR}"/.env
    set +a
    export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=$REGION
    echo "PROJECT_ID: $PROJECT_ID"
    echo "REGION: $REGION"
    echo "DATASET: $DATASET"
    echo "LOG_TABLE: $GEMINI_LOG_TABLE"
fi

echo "=========================================="
echo "Autonomous Latency Analysis "
echo "=========================================="

cd "$SCRIPT_DIR/agents"
adk run --replay ../autonomous_analysis_90d.json latency_analyzer

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Check the reports/ directory for the saved report."
echo "=========================================="
