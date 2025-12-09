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
fi

echo "=========================================="
echo "Autonomous Latency Analysis (90 days)"
echo "=========================================="
echo ""
echo "This will run an autonomous analysis where the agent:"
echo "  - Makes intelligent decisions about which tools to use"
echo "  - Adapts based on findings (e.g., uses alternative tools if one fails)"
echo "  - Generates and tests hypotheses systematically"
echo "  - Analyzes correlations, clusters, costs, and individual queries"
echo "  - Creates a comprehensive final report"
echo "  - Saves the report to reports/ directory with timestamp"
echo ""
echo "Starting autonomous analysis..."
echo ""



cd "$SCRIPT_DIR/agents"
adk run --replay ../autonomous_analysis_90d.json latency_analyzer

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Check the reports/ directory for the saved report."
echo "=========================================="
