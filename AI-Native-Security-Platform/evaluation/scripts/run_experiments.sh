#!/bin/bash
# Reproduction Script for ICAI-2026 Paper Experiments & Figures
set -e

echo "=========================================================="
echo "REPRODUCING ALL BENCHMARK EXPERIMENTS & GENERATING FIGURES"
echo "=========================================================="

PYTHONPATH=. python evaluation/scripts/generate_figures.py

echo "Experiment reproduction complete. Figures generated in paper/figures/"
