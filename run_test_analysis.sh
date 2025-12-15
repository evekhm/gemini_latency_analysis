#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPLAY_FILE="autonomous_analysis_test.json"

bash "${SCRIPT_DIR}/run_analysis.sh" "$REPLAY_FILE"
