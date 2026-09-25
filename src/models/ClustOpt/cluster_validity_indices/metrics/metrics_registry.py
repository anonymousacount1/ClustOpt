# metrics/registry.py
from __future__ import annotations
from typing import Dict
from .image_base.utils.image_utils import RasterParams
from .metrics_base import BaseMetric
from .core_cvi.Silhouette import SilhouetteMetric
from .core_cvi.calinski_harabasz import CalinskiHarabaszMetric
from .core_cvi.davies_bouldin import DaviesBouldinMetric
from .core_cvi.Density_Based_Clustering_Validation_Index import DBCVMetric
from .core_cvi.S_Dbw_Index import SDbwMetric
from .core_cvi.Noise_Aware_Silhouette import NoiseAwareSilhouetteMetric
from .structure_base.Average_cluster_compactness import AvgPCAIsotropyMetric
from .structure_base.Entropy_cluster_sizes import ClusterSizeBalanceEntropyMetric, ClusterSizeImbalanceEntropyMetric
from .structure_base.Local_Neighborhood_Purity import NeighborhoodPurityMetric, NeighborhoodPurityMultiKMetric
from .structure_base.min_intercluster_distance import MinInterclusterDistanceMetric
from .structure_base.avg_within_cluster_dispersion import AvgWithinClusterDispersionMetric
from .structure_base.min_centroid_distance import MinCentroidDistanceMetric
from .pattern_base.arc_circle_fit import ArcCircleFitMetric
from .pattern_base.ladderness import LaddernessMetric
from .pattern_base.parallel_bands import ParallelBandsMetric
from .pattern_base.parabolicity import ParabolicityMetric
from .pattern_base.periodicity_1d import Periodicity1DMetric
from .pattern_base.line_straightness import LineStraightnessMetric
from .pattern_base.piecewise_linearity import PiecewiseLinearityMetric
from .pattern_base.alpha_shape_compactness import AlphaShapeCompactnessMetric
from .pattern_base.bimodality_anti_thickness import BimodalityAntiThicknessMetric
from .pattern_base.convexity_ratio import ConvexityRatioMetric
from .pattern_base.curvature_consistency import CurvatureConsistencyMetric
from .pattern_base.turning_points_count import TurningPointsCountMetric
from .pattern_base.direction_entropy import DirectionEntropyMetric
from .pattern_base.gridness_2d import Gridness2DMetric
from .pattern_base.mst_smoothness import MSTSmoothnessMetric
from .pattern_base.ribbon_thickness import RibbonThicknessMetric
from .pattern_base.axial_symmetry import AxialSymmetryMetric
from .pattern_base.connectivity_components import ConnectivityComponentsMetric
from .pattern_base.corner_sharpness import CornerSharpnessMetric
from .pattern_base.eccentricity_ratio import EccentricityRatioMetric
from .pattern_base.hollow_score import HollowScoreMetric
from .pattern_base.radial_uniformity import RadialUniformityMetric
from .image_base.hough_line_strength import HoughLineStrengthMetric
from .image_base.peak_to_background_line_ratio import PeakToBackgroundHoughMetric
from .image_base.orientation_histogram_entropy import OrientationHistogramEntropyMetric
from .image_base.edge_coherence import EdgeCoherenceStructureTensorMetric
from .image_base.hough_arc_circle_strength import HoughCircleArcStrengthMetric
from .image_base.convexity_ratio_image_based import ConvexitySolidityImageMetric
from .image_base.ellipse_fit_score import EllipseFitScoreMetric
from .image_base.circularity_compactness import CircularityCompactnessMetric
from .image_base.polygonality import PolygonalityMetric
from .image_base.rectangularity import RectangularityMetric
from .image_base.morphological_band_count import MorphologicalBandCountMetric
from .image_base.skeleton_connectivity import SkeletonConnectivityMetric
from .image_base.contour_graph_connectivity import ContourGraphConnectivityMetric
from .image_base.ridge_strength import RidgeStrengthMetric
from .image_base.thickness_uniformity import ThicknessUniformityMetric
from .image_base.glcm_haralick import GLCMHaralickMetric
from .image_base.lbp_stationarity import LBPStationarityMetric
from .image_base.fractal_dimension import FractalDimensionMetric
from .image_base.gridness_fft_acf import GridnessFFTACFMetric
from .image_base.symmetry_score import SymmetryScoreMetric
from .image_base.euler_holes_count import EulerHolesCountMetric
from .image_base.connected_components_count import ConnectedComponentsCountMetric
from .image_base.blobness_log_dog import BlobnessLogDogMetric
from .image_base.corner_junction_density import CornerJunctionDensityMetric



METRIC_REGISTRY: Dict[str, BaseMetric] = {
    "silhouette": SilhouetteMetric(),
    "calinski_harabasz": CalinskiHarabaszMetric(),
    "davies_bouldin": DaviesBouldinMetric(),
    "dbcv": DBCVMetric(),
    "s_dbw": SDbwMetric(),
    "noise_aware_silhouette": NoiseAwareSilhouetteMetric(alpha=1.0),
    "avg_pca_isotropy": AvgPCAIsotropyMetric(eps=1e-8, agg="mean"),
    "cluster_size_balance_entropy": ClusterSizeBalanceEntropyMetric(eps=1e-12),
    "cluster_size_imbalance_entropy": ClusterSizeImbalanceEntropyMetric(eps=1e-12),
    "neighborhood_purity": NeighborhoodPurityMetric(k=10),
    "neighborhood_purity_multi_k": NeighborhoodPurityMultiKMetric(ks=(5, 10, 20), agg="mean", weights=None),
    "min_intercluster_distance": MinInterclusterDistanceMetric(eps=1e-12, scale=1.0),
    "avg_within_cluster_dispersion": AvgWithinClusterDispersionMetric(eps=1e-12, scale=1.0, agg="mean"),
    "min_centroid_distance": MinCentroidDistanceMetric(eps=1e-12, scale=1.0),
    "arc_circle_fit": ArcCircleFitMetric(radial_clip=2.0, min_cluster_points=5, use_ellipse=True, ellipse_fallback=1.0, eps=1e-12),
    "ladderness": LaddernessMetric(bins=64, min_rel_height=0.05, peak_distance_bins=2, tau_peaks=3.0, elong_r0=2.0, fallback_weight=0.25, min_cluster_points=8, eps=1e-12, adaptive_bins=True),
    "parallel_bands": ParallelBandsMetric(straightness_floor=0.6, w_angle=0.5, w_spacing=0.5, min_cluster_points=8, eps=1e-12),
    "parabolicity": ParabolicityMetric(a_scale=0.1, a_clip=5.0, r2_min=0.2, min_cluster_points=5, eps=1e-12),
    "periodicity_1d": Periodicity1DMetric(bins=128, adaptive_bins=True, min_bins_occupied=6, ignore_low_freq_frac=0.05, w_elon=0.0, elong_r0=2.0, min_cluster_points=8, eps=1e-12),
    "line_straightness": LineStraightnessMetric(min_cluster_points=3, eps=1e-8, w_energy=0.5, w_thickness=0.5),
    "piecewise_linearity": PiecewiseLinearityMetric(min_cluster_points=5, eps=1e-8),
    "alpha_shape_compactness": AlphaShapeCompactnessMetric(alpha=None, alpha_scale=3.5, min_cluster_points=10, eps=1e-12),
    "bimodality_anti_thickness": BimodalityAntiThicknessMetric(min_cluster_points=20, bins=80, adaptive_bins=True, min_bins_occupied=10, smooth_kernel=(1, 2, 3, 2, 1), min_peak_rel_height=0.15, min_peak_distance_bins=6, sep_r0=1.0, w_valley=0.55, w_sep=0.45, w_elon=0.30, elong_r0=2.0, eps=1e-12),
    "convexity_ratio": ConvexityRatioMetric(alpha=None, alpha_scale=3.5, min_cluster_points=10, eps=1e-12),
    "curvature_consistency": CurvatureConsistencyMetric(min_cluster_points=18, min_triplets=8, eps=1e-12, w_sign=0.6, w_var=0.4, var_scale=1.0, w_elon=0.3, elong_r0=2.0),
    "turning_points_count": TurningPointsCountMetric(min_cluster_points=5, eps=1e-12, near_zero_percentile=60.0, w_elon=0.0, elong_r0=2.0),
    "direction_entropy": DirectionEntropyMetric(bins=24, min_cluster_points=18, min_steps=12, step_eps=1e-8, eps=1e-12, w_elon=0.0, elong_r0=2.0),
    "gridness_2d": Gridness2DMetric(grid_size=32, min_cluster_points=40, min_occupied_rows=3, min_occupied_cols=3, rowcol_presence_thresh=2, eps=1e-12, w_elon=0.0, elong_r0=2.0),
    "mst_smoothness": MSTSmoothnessMetric(min_cluster_points=20, knn_k=10, w_angle=0.45, branch_scale=0.25, eps=1e-12),
    "ribbon_thickness": RibbonThicknessMetric(min_cluster_points=30, n_bins=20, min_bin_points=8, q_low=10.0, q_high=90.0, eps=1e-12, w_thin=0.5, w_cons=0.5, elong_r0=2.0),
    "axial_symmetry": AxialSymmetryMetric(min_cluster_points=20, n_bins=36, min_bin_points=10),
    "connectivity_components": ConnectivityComponentsMetric(min_cluster_points=10, eps=1e-12),
    "corner_sharpness": CornerSharpnessMetric(min_cluster_points=20, eps=1e-12),
    "eccentricity_ratio": EccentricityRatioMetric(min_cluster_points=5, eps=1e-12),
    "hollow_score": HollowScoreMetric(min_cluster_points=20),
    "radial_uniformity": RadialUniformityMetric(min_cluster_points=20, n_bins=36),
    "hough_line_strength": HoughLineStrengthMetric(min_cluster_points=30, raster=RasterParams(grid_size=256, mode="binary", blur_ksize=3, dilate_iters=1), canny_low=50, canny_high=150, eps=1e-12),
    "peak_to_background_hough": PeakToBackgroundHoughMetric(min_cluster_points=30, canny_low=50, canny_high=150, eps=1e-12),
    "orientation_histogram_entropy": OrientationHistogramEntropyMetric(min_cluster_points=20, eps=1e-12),
    "edge_coherence": EdgeCoherenceStructureTensorMetric(min_cluster_points=30, eps=1e-12),
    "hough_arc_circle_strength": HoughCircleArcStrengthMetric(min_cluster_points=20, canny_low=50, canny_high=150, dp=1.0, param1=50, param2=30),
    "convexity_ratio_image_based": ConvexitySolidityImageMetric(min_cluster_points=20, eps=1e-12),
    "ellipse_fit_score": EllipseFitScoreMetric(min_cluster_points=5),
    "circularity_compactness": CircularityCompactnessMetric(min_cluster_points=5, eps=1e-12),
    "polygonality": PolygonalityMetric(min_cluster_points=5, eps=1e-12),
    "rectangularity": RectangularityMetric(min_cluster_points=5, eps=1e-12),
    "morphological_band_count": MorphologicalBandCountMetric(min_cluster_points=80, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=2, line_kernel_frac=0.12, profile_smooth_k=11, peak_min_dist=10, peak_min_prom=0.06),
    "skeleton_connectivity": SkeletonConnectivityMetric(min_cluster_points=30),
    "contour_graph_connectivity": ContourGraphConnectivityMetric(min_cluster_points=80, grid_size=256, raster_mode="density", blur_ksize=3, dilate_iters=1, canny_low=50, canny_high=150, a_comp=0.9, min_edges=40),
    "ridge_strength": RidgeStrengthMetric(min_cluster_points=80, grid_size=256, raster_mode="density", blur_ksize=3, dilate_iters=1, sigmas=[1.0, 1.6, 2.4, 3.4], top_quantile=0.95, beta=0.5, c=15.0),
    "thickness_uniformity": ThicknessUniformityMetric(min_cluster_points=30, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=1),
    "glcm_haralick": GLCMHaralickMetric(min_cluster_points=120, grid_size=256, raster_mode="density", blur_ksize=3, dilate_iters=1, levels=16, offsets=[(1, 0), (0, 1), (2, 0), (0, 2)]),
    "lbp_stationarity": LBPStationarityMetric(min_cluster_points=150, grid_size=256, raster_mode="density", blur_ksize=3, dilate_iters=1, tiles=4, min_foreground_frac=0.02),
    "fractal_dimension": FractalDimensionMetric(min_cluster_points=120, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=2, canny_low=50, canny_high=150),
    "gridness_fft_acf": GridnessFFTACFMetric(min_cluster_points=40, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=1),
    "symmetry_score": SymmetryScoreMetric(min_cluster_points=120, grid_size=256, raster_mode="density", blur_ksize=3, dilate_iters=1),
    "euler_holes_count": EulerHolesCountMetric(min_cluster_points=150, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=2, close_iters=1),
    "connected_components_count": ConnectedComponentsCountMetric(min_cluster_points=120, grid_size=256, raster_mode="density", blur_ksize=5, dilate_iters=3, a_comp=0.6, min_foreground=40),
    "blobness_log_dog": BlobnessLogDogMetric(min_cluster_points=80, grid_size=256, raster_mode="density", blur_ksize=0, dilate_iters=0, sigmas=(1.2, 2.0, 3.2), dog_ratio=1.6, top_q=0.03, eps=1e-12),
    "corner_junction_density": CornerJunctionDensityMetric(min_cluster_points=80, grid_size=256, raster_mode="binary", blur_ksize=3, dilate_iters=1, canny_low=50, canny_high=150, max_corners=120, quality_level=0.01, min_distance=6, block_size=3, use_harris=False, sat_c=10.0, eps=1e-12)
}
