#!/bin/bash
# Enhanced autonomous latency analysis with colors and pre-flight checks

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Suppress experimental feature warnings
export PYTHONWARNINGS="ignore::UserWarning"
# Suppress gRPC fork warnings
export GRPC_ENABLE_FORK_SUPPORT=0
export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=none

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Parallel Autonomous Latency Analysis${NC}"
echo -e "${BLUE}  With Automatic Deep Research Triggers${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Pre-flight checks
REPLAY_FILE="autonomous_analysis_90d.json"

# Check if .env file exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo -e "${GREEN}✓ Loading .env file...${NC}"
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=$AGENT_REGION
    echo -e "${GREEN}  Project: $PROJECT_ID${NC}"
    echo -e "${GREEN}  Region: $AGENT_REGION${NC}"
    echo -e "${GREEN}  Dataset: $DATASET_ID${NC}"
    echo -e "${GREEN}  Table(s): $AGENT_TABLE_ID${NC}"
else
    echo -e "${YELLOW}⚠ Warning: .env file not found. Make sure environment variables are set.${NC}"
fi

echo ""

# Check if replay file exists
if [ ! -f "${SCRIPT_DIR}/$REPLAY_FILE" ]; then
    echo -e "${YELLOW}✗ Error: Replay file '$REPLAY_FILE' not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Using replay file: $REPLAY_FILE${NC}"

# Check if virtual environment exists
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo -e "${YELLOW}✗ Error: Virtual environment not found. Run 'python -m venv .venv' first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment found${NC}"

echo ""
echo -e "${GREEN}Starting autonomous analysis...${NC}"
echo ""

# Create logs directory if it doesn't exist
    mkdir -p "${SCRIPT_DIR}/logs"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SCRIPT_LOG="${SCRIPT_DIR}/logs/script_${TIMESTAMP}.log"
    
    # Run the agent with caching enabled (replaces adk run) and pipe to tee
    echo -e "${GREEN}Running with ADK Context Caching enabled...${NC}"
    echo -e "${GREEN}Script output being saved to: ${SCRIPT_LOG}${NC}"
    
    # Create latest_script.log symlink
    ln -sf "${SCRIPT_LOG}" "${SCRIPT_DIR}/latest_script.log"
    echo -e "${GREEN}Created symlink: latest_script.log -> ${SCRIPT_LOG}${NC}"
    echo -e "${BLUE}  Tip: Run 'tail -f latest_script.log' to monitor full script output${NC}"
    
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/run_with_caching.py" "$SCRIPT_DIR/$REPLAY_FILE" 2>&1 | tee "${SCRIPT_LOG}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Analysis Complete!${NC}"
echo -e "${GREEN}  Check reports/ for the analysis report${NC}"
echo -e "${GREEN}========================================${NC}"
