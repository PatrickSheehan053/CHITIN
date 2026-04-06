"""
CHITIN · src/utils.py
─────────────────────
Config loading, logging, memory monitoring, and plotting theme.
"""

import yaml
import logging
import os
import gc
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def load_config(config_path: str = "chitin_config.yaml") -> dict:
    """Load and validate the CHITIN YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["project_root"])
    cfg["paths"]["_root"] = root
    cfg["paths"]["_input"] = root / cfg["paths"]["input_dir"]
    cfg["paths"]["_output"] = root / cfg["paths"]["output_dir"]
    cfg["paths"]["_figures"] = root / cfg["paths"]["figures_dir"]
    cfg["paths"]["_logs"] = root / cfg["paths"]["log_dir"]

    for key in ["_output", "_figures", "_logs"]:
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)

    return cfg


def setup_logger(cfg: dict, name: str = "CHITIN") -> logging.Logger:
    """Configure CHITIN logger with console + file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s",
                            datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = cfg["paths"]["_logs"] / f"chitin_run_{timestamp}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s"))
    logger.addHandler(fh)

    logger.info(f"CHITIN log initialized → {log_path}")
    return logger


def log_phase(logger, title: str):
    """Print a clean phase header."""
    bar = "═" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def get_memory_usage() -> str:
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    if mem_bytes >= 1e9:
        return f"{mem_bytes / 1e9:.1f} GB"
    return f"{mem_bytes / 1e6:.0f} MB"


def log_memory(logger, label: str = ""):
    mem = get_memory_usage()
    tag = f" ({label})" if label else ""
    logger.info(f"  💾 Memory{tag}: {mem}")


def force_gc(logger=None):
    collected = gc.collect()
    if logger:
        logger.info(f"  🗑️  GC collected {collected} objects  |  RAM: {get_memory_usage()}")


def snapshot(adata, label: str, logger):
    """Log shape snapshot."""
    n, g = adata.shape
    logger.info(f"  [{label}] → {n:,} metacells  ×  {g:,} genes  |  RAM: {get_memory_usage()}")


# ── SPORE Dark Theme ────────────────────────────────────────────────────────

_DARK = {
    "bg": "#0D1117", "panel": "#161B22", "text": "#E6EDF3",
    "grid": "#21262D", "accent": "#58A6FF", "warn": "#F85149",
    "good": "#3FB950", "muted": "#8B949E", "highlight": "#D2A8FF",
    "ctrl": "#79C0FF", "pert": "#F0883E", "delta": "#3FB950",
}

_LIGHT = {
    "bg": "#FFFFFF", "panel": "#F6F8FA", "text": "#1F2328",
    "grid": "#D1D9E0", "accent": "#0969DA", "warn": "#CF222E",
    "good": "#1A7F37", "muted": "#656D76", "highlight": "#8250DF",
    "ctrl": "#0550AE", "pert": "#BC4C00", "delta": "#1A7F37",
}


def get_theme(cfg: dict) -> dict:
    return _DARK if cfg["plotting"]["style"] == "dark" else _LIGHT


def apply_chitin_style(cfg: dict):
    """Apply the CHITIN/SPORE matplotlib style globally."""
    theme = get_theme(cfg)
    plt.rcParams.update({
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["panel"],
        "axes.edgecolor": theme["grid"],
        "axes.labelcolor": theme["text"],
        "text.color": theme["text"],
        "xtick.color": theme["text"],
        "ytick.color": theme["text"],
        "grid.color": theme["grid"],
        "grid.alpha": 0.5,
        "figure.dpi": cfg["plotting"]["dpi"],
        "font.family": "monospace",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": theme["panel"],
        "legend.edgecolor": theme["grid"],
        "savefig.facecolor": theme["bg"],
        "savefig.bbox": "tight",
        "savefig.dpi": cfg["plotting"]["dpi"],
    })


def save_fig(fig, cfg: dict, filename: str):
    """Save figure if configured."""
    if cfg["plotting"]["save_figures"]:
        fmt = cfg["plotting"]["figure_format"]
        path = cfg["paths"]["_figures"] / f"{filename}.{fmt}"
        fig.savefig(path, facecolor=fig.get_facecolor())
        return path
    return None


def format_ax(ax, theme: dict, title: str, xlabel: str = "", ylabel: str = ""):
    """Apply consistent formatting to an axis."""
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
