"""Generate architecture diagram for the ICAI-2026 paper."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PNG = os.path.join(BASE_DIR, "paper", "figures", "fig1_architecture.png")
OUT_RESULTS = os.path.join(BASE_DIR, "results", "fig1_architecture.png")


def _box(ax, x, y, w, h, text, facecolor, edgecolor, linestyle="-", linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=linewidth,
        linestyle=linestyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, fontweight="bold", wrap=True)
    return patch


def generate_architecture_figure():
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("AI-Native Event-Driven Edge–Cloud Security Architecture", fontsize=11, pad=8)

    # Lane backgrounds
    _box(ax, 0.3, 0.6, 3.2, 3.8, "", "#E3F2FD", "#1565C0")
    _box(ax, 4.0, 0.6, 2.4, 3.8, "", "#FFF3E0", "#EF6C00")
    _box(ax, 6.9, 0.6, 4.7, 3.8, "", "#F3E5F5", "#6A1B9A")

    # Trust Boundary (Red Dashed Box around Edge Ingest)
    _box(ax, 0.15, 0.45, 3.5, 4.15, "", "none", "#D32F2F", linestyle="--", linewidth=1.8)
    ax.text(1.9, 4.7, "[!] TRUST BOUNDARY / ATTACK SURFACE", ha="center", fontsize=7.5, fontweight="bold", color="#D32F2F")

    ax.text(1.9, 4.15, "Campus Edge (Untrusted Ingest)", ha="center", fontsize=8.5, fontweight="bold", color="#0D47A1")
    ax.text(5.2, 4.15, "Event Bus (TLS Stream)", ha="center", fontsize=8.5, fontweight="bold", color="#E65100")
    ax.text(9.25, 4.15, "Cloud Control Plane (Trusted Domain)", ha="center", fontsize=8.5, fontweight="bold", color="#4A148C")

    _box(ax, 0.55, 2.7, 2.7, 0.9, "Telemetry Vectorization\nΔt = 1.0 s (10-D flows)", "#FFFFFF", "#1565C0")
    _box(ax, 0.55, 1.5, 2.7, 0.9, "INT8 TCN-GRU\nDetector", "#FFFFFF", "#1565C0")
    _box(ax, 0.55, 0.8, 2.7, 0.5, "Identity Fusion (IP/MAC→Role)", "#FFFFFF", "#1565C0")

    _box(ax, 4.25, 1.7, 1.9, 1.6, "Redis Stream\nsecurity:\ntelemetry:stream", "#FFFFFF", "#EF6C00")

    _box(ax, 7.2, 2.7, 2.0, 0.9, "Dynamic Risk\nResolver R", "#FFFFFF", "#6A1B9A")
    _box(ax, 9.4, 2.7, 1.95, 0.9, "DQN SDN\nController", "#FFFFFF", "#6A1B9A")
    _box(ax, 7.2, 1.2, 4.15, 1.1, "SOAR Action Descriptors\nForward / Rate-Limit / Reroute / Isolate", "#FFFFFF", "#6A1B9A")

    arrows = [
        ((3.25, 1.95), (4.25, 2.3)),
        ((6.15, 2.5), (7.2, 3.1)),
        ((9.2, 3.15), (9.4, 3.15)),
        ((8.2, 2.7), (8.2, 2.3)),
        ((10.35, 2.7), (9.6, 2.3)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#333333",
            )
        )

    ax.text(
        6.0,
        0.2,
        "Software orchestration path; dataplane enforcement not measured",
        ha="center",
        fontsize=8,
        style="italic",
    )

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_RESULTS, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved architecture figure -> {OUT_PNG}")


if __name__ == "__main__":
    generate_architecture_figure()
