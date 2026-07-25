#!/bin/bash
# Interactive Demo Execution Script for AI-Native Security Platform
set -e

echo "=========================================================="
echo "RUNNING INTERACTIVE PROTOTYPE DEMO (ICAI-2026)"
echo "=========================================================="

PYTHONPATH=. python prototype/edge/detector/main.py

echo "Demo execution finished successfully."
