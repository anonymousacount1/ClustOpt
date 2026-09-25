"""Paths, naming and figure style for rebuilding the paper's tables and figures.

Presentation only. Every value is read from a released result table (``results/``)
through ``analysis/release_paths.py``; outputs go to ``analysis/output/paper``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from release_paths import ROOT as _ROOT, LegacyPath  # noqa: E402

PROJ = _ROOT / "analysis" / "output" / "paper"
FIG = PROJ / "figures"
TAB = PROJ / "tables"
MAC = PROJ / "macros"
EVI = PROJ / "evidence"
MAN = PROJ / "manifests"
for _d in (FIG, TAB, MAC, EVI, MAN):
    _d.mkdir(parents=True, exist_ok=True)

REPO = LegacyPath("")
CR = REPO / "results_analysis" / "clustering_repository"

S1 = CR / "stage1_2x3_aggregation"
S2 = (REPO / "results_analysis" / "clustopt_policy_predictor"
      / "stage2c_final_v2_split1_online_replay")
S2B5 = CR / "candidate_reranker_split1_online_final"
AC = CR / "autoclust_extended_result_provenance_audit"
S3 = CR / "metric_analysis_existing_domain" / "family_subfamily_specialization"
S3P = CR / "metric_analysis_existing_domain"
DATA = CR / "analyzed_data"
UNI = None  # the full training table is not released; see build_evidence.repository()
CATALOGUE = _ROOT / "data" / "synthetic_generator" / "dataset_catalogue.csv"

RUN = (REPO / "results_analysis" / "independent_domain_benchmark"
       / "stage4c_v2_323194df26")
SR = RUN / "scientific_review"

IDBCFG = REPO / "models" / "Independent_Domain_Benchmark" / "configs"
CORPUS = (REPO / "results_analysis" / "independent_domain_benchmark"
          / "corpus_manifest" / "stage4_external_corpus_publication_manifest.json")
MLPRUN = (REPO / "results_analysis" / "mlp"
          / "20260615_231715__metric_utility_mlp_split1_holdout")
KNNRUN = REPO / "results_analysis" / "metric_utility_knn" / "phase_e1"
RRPREP = (REPO / "results_analysis" / "clustopt_candidate_reranker"
          / "stage2b5a_final_models_and_split1_preparation")
STAGE1REP = REPO / "results_analysis" / "mlp"

# --------------------------------------------------------------- naming
# Human-readable arm names. Internal codes never appear as a primary label.
NAME = {
    "C5": "ClustOpt PPv2 + rerank",
    "C4": "ClustOpt KNN + rerank",
    "C3": "ClustOpt MLP + rerank",
    "C2": "ClustOpt PPv1",
    "C1": "ClustOpt KNN",
    "C0": "ClustOpt MLP",
    "A0": "AutoClust (original)",
    "A1": "AutoClust (+established)",
    "A2": "AutoClust (+new)",
    "A3": "AutoClust (extended)",
    "M0": "ML2DAC (original)",
    "M1": "ML2DAC (+established)",
    "M2": "ML2DAC (+new)",
    "M3": "ML2DAC (extended)",
}
SHORT = {k: v.replace(" (", "\n(") for k, v in NAME.items()}

TEXNAME = {
    "C5": r"\clustopt{} \textsc{ppv2}+R", "C4": r"\clustopt{} \textsc{knn}+R",
    "C3": r"\clustopt{} \textsc{mlp}+R", "C2": r"\clustopt{} \textsc{ppv1}",
    "C1": r"\clustopt{} \textsc{knn}", "C0": r"\clustopt{} \textsc{mlp}",
    "A0": "AutoClust original", "A1": "AutoClust +established",
    "A2": "AutoClust +new", "A3": "AutoClust extended",
    "M0": "ML2DAC original", "M1": "ML2DAC +established",
    "M2": "ML2DAC +new", "M3": "ML2DAC extended",
}

# 12 structural families: directory key -> short display label
FAM_SHORT = {
    "arcs_circles_rings": "Arcs \& rings",
    "deinterleaving_pri_toa_amp_like": "De-interleaving",
    "hollow_topological_components": "Hollow shapes",
    "hybrid_mixed_geometry": "Hybrid mixed",
    "ladder_grid_lattice": "Ladder \& grid",
    "linear_bands_parallel_stripes": "Bands \& stripes",
    "parabolic_curved_chains": "Curved chains",
    "piecewise_polylines_corners": "Polylines",
    "polygons_rectangles_frames": "Polygons",
    "radial_spiral_meander": "Radial \& spiral",
    "standard_clustering_blobs": "Isotropic blobs",
    "texture_density_fields": "Texture fields",
}
FAM_PLAIN = {k: v.replace("\\&", "&") for k, v in FAM_SHORT.items()}

# ------------------------------------------------------- colour system
# Colour-blind-safe (Okabe-Ito derived), semantic and consistent everywhere.
C_CLUSTOPT = "#0072B2"      # blue family
C_CLUSTOPT_L = "#56B4E9"
C_AUTOCLUST = "#D55E00"     # orange
C_ML2DAC = "#009E73"        # green
C_ORACLE = "#6E6E6E"        # neutral grey
C_ACCENT = "#CC79A7"
C_NEUTRAL = "#3A3A3A"
C_GRID = "#D9D9D9"

SRC_COLOR = {"fcps": "#0072B2", "sklearn_shapes": "#D55E00",
             "real_pca": "#009E73"}
SRC_LABEL = {"fcps": "FCPS", "sklearn_shapes": "scikit-learn",
             "real_pca": "real (PCA)"}


def arm_color(short: str) -> str:
    if short.startswith("C"):
        return C_CLUSTOPT if short in ("C5", "C4") else C_CLUSTOPT_L
    if short.startswith("A"):
        return C_AUTOCLUST
    if short.startswith("M"):
        return C_ML2DAC
    return C_ORACLE


def apply_style():
    """One style module for every figure (brief section 30)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#4A4A4A",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.85,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "patch.linewidth": 0.5,
    })
    return plt


PROV: list = []


def record(asset: str, sources, note: str):
    PROV.append({"asset": asset, "sources": list(sources), "note": note})


def write_prov(name: str = "asset_provenance_v12.json"):
    (MAN / name).write_text(
        json.dumps({"note": "presentation-only; no scientific recomputation",
                    "assets": PROV}, indent=2), encoding="utf-8")
