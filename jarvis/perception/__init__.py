"""Local visual perception for the JARVIS joint-attention prototype."""

from .geometry import PointingRay, pointing_geometry, ray_to_border
from .pipeline import PerceptionPipeline
from .servo import (
    FaceServoConfig,
    FaceServoPlanner,
    HttpPointingServoBridge,
    PointingGazeConfig,
    PointingGazePlanner,
    PointingGazeRecommendation,
    ServoRecommendation,
)
from .tracker import FaceTracker, select_primary_face
from .types import (
    BoundingBox,
    FaceCandidate,
    HandObservation,
    PerceptionFrame,
    TrackedFace,
    UvPoint,
)

__all__ = [
    "BoundingBox",
    "FaceCandidate",
    "FaceServoConfig",
    "FaceServoPlanner",
    "FaceTracker",
    "HandObservation",
    "HttpPointingServoBridge",
    "PerceptionFrame",
    "PerceptionPipeline",
    "PointingGazeConfig",
    "PointingGazePlanner",
    "PointingGazeRecommendation",
    "PointingRay",
    "ServoRecommendation",
    "TrackedFace",
    "UvPoint",
    "pointing_geometry",
    "ray_to_border",
    "select_primary_face",
]

