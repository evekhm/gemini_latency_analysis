#!/bin/bash
# run_latency_research.sh - Run deep latency analysis using replay mode

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Deep Latency Research Analysis${NC}"
echo -e "${BLUE}  Time Range: Last 90 Days${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Make sure environment variables are set.${NC}"
fi

# Check if replay file exists
REPLAY_FILE="deep_latency_research_90d.json"
if [ ! -f "$REPLAY_FILE" ]; then
    echo -e "${YELLOW}Error: Replay file '$REPLAY_FILE' not found.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Error: Virtual environment not found. Run 'python -m venv .venv' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Starting latency analyzer agent...${NC}"
echo -e "${GREEN}✓ Using replay file: $REPLAY_FILE${NC}"
echo ""

# Change to agents directory and run
cd agents

# Run the agent with replay
../.venv/bin/adk run --replay ../$REPLAY_FILE latency_analyzer

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
