"""
Steerable Filter Pipeline for TEM Replication Fork Analysis.

Modular pipeline split into independent steps:
  1. Ridge filter detection
  2. Mask prediction & cleanup
  3. Skeletonization & pruning
  4. Graph extraction & pruning
  5. Fork detection & classification
  6. Visualization & metrics
"""

from pipeline.config import PipelineConfig
from pipeline.io_utils import (
    PredictionResult,
    SkeletonGraph,
    Fork,
    load_image,
    load_mask,
    match_images_masks,
)
from pipeline.step1_ridge import run_ridge_filter
from pipeline.step2_mask import run_mask_prediction
from pipeline.step3_skeleton import run_skeletonize
from pipeline.step4_graph import run_graph_extraction
from pipeline.step5_forks import run_fork_detection
from pipeline.step6_visualize import run_visualization, compute_metrics
