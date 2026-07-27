"""
quantize.py - PyTorch Dynamic INT8 Quantization Utility for Edge Deployments

Ported from sdn-its-resilience-ai (models/quantize.py).
Quantizes PyTorch FP32 TCN-GRU linear and GRU layers to qint8 for edge execution.
"""

import os
import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel

def quantize_tcn_gru_model(model: nn.Module) -> nn.Module:
    """Applies dynamic PyTorch quantization (float32 -> qint8) to Linear and GRU layers."""
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    return quantized_model

def benchmark_model_latency(model: nn.Module, sample_input: torch.Tensor, num_runs: int = 500) -> Dict[str, float]:
    """Measures single-sample inference latency (ms) and throughput (samples/sec)."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(sample_input)
            
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input)
    end_time = time.perf_counter()
    
    total_time_s = end_time - start_time
    avg_latency_ms = (total_time_s / num_runs) * 1000.0
    throughput = num_runs / total_time_s
    
    return {
        "avg_latency_ms": avg_latency_ms,
        "throughput_fps": throughput
    }
