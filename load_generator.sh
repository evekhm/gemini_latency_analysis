#!/bin/bash

# Wrapper script for running load_generator.py with different model configurations
# Usage: ./load_generator.sh <model_identifier> [scenario] [additional_args]
# Example: ./load_generator.sh 2.5pro
# Example: ./load_generator.sh 2.5pro all
# Example: ./load_generator.sh 2.0flash thinking_vs_baseline --count 5

set -e

if [ -z "$1" ]; then
    MODEL_IDENTIFIER="2.5pro"
    SCENARIO="all"
    echo "No arguments provided. Defaulting to model: $MODEL_IDENTIFIER"
else
    MODEL_IDENTIFIER="$1"
    
    # Determine scenario and shift arguments accordingly
    # If the second argument is present and doesn't start with a dash, use it as scenario
    if [ -n "$2" ] && [[ "$2" != -* ]]; then
        SCENARIO="$2"
        shift 2
    else
        SCENARIO="all"
        shift 1
    fi
fi

ENV_FILE=".env-${MODEL_IDENTIFIER}"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file '$ENV_FILE' not found"
    echo "Please create $ENV_FILE with your model configuration"
    exit 1
fi

echo "========================================="
echo "Load Generator Wrapper"
echo "========================================="
echo "Model Identifier: $MODEL_IDENTIFIER"
echo "Environment File: $ENV_FILE"
echo "Scenario: $SCENARIO"
if [ $# -gt 0 ]; then
    echo "Additional Args: $@"
fi
echo "========================================="
echo ""

# Run the load generator with the specified environment file
python load_generator.py "$SCENARIO" --env-file "$ENV_FILE" "$@"
