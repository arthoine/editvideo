"""
EditVideo - Éditeur Automatique de Streams Gaming avec IA.

Package principal contenant les modules d'analyse et d'édition vidéo.
"""

__version__ = "1.0.0"
__author__ = "EditVideo Team"
__license__ = "MIT"

from src.analyzer import VideoAnalyzer, Segment
from src.editor import VideoEditor
from src.utils import (
    ConfigManager,
    GPUManager,
    CacheManager,
    LoggerSetup,
    SystemUtils,
)

__all__ = [
    "VideoAnalyzer",
    "Segment",
    "VideoEditor",
    "ConfigManager",
    "GPUManager",
    "CacheManager",
    "LoggerSetup",
    "SystemUtils",
]
