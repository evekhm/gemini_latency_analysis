#!/bin/bash

# Script to run comprehensive latency analysis using the unified latency_analyzer agent
# This demonstrates all 16 tools including statistical analysis, clustering, and individual query deep-dives

set -e

echo "=========================================="
echo "Comprehensive Latency Analysis (90 days)"
echo "=========================================="
echo ""
echo "This will run a complete analysis including:"
echo "  - Overall statistics and health check"
echo "  - Token correlation analysis (output+thought tokens)"
echo "  - Clustering of slow queries"
echo "  - Agent performance comparison"
echo "  - Performance degradation trends"
echo "  - Cost analysis"
echo "  - Individual slow query deep-dives"
echo "  - Prioritized recommendations"
echo ""
echo "Starting analysis..."
echo ""

cd agents
adk run --replay ../comprehensive_analysis_90d.json latency_analyzer

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
