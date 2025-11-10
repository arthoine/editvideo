"""
Module d'analyse vidéo pour la détection de moments clés.

Ce module implémente :
- Analyse audio (détection de pics sonores)
- Détection de changements de scène
- Analyse LLaVA (IA multimodale pour détection intelligente)
- Scoring des segments
- Sélection intelligente des meilleurs moments
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import librosa
import soundfile as sf
from tqdm import tqdm
from loguru import logger

from src.utils import CacheManager, SystemUtils

# Import conditionnel de LLaVA
try:
    from src.llava_analyzer import LLaVAAnalyzer, LLaVAAnalysis
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    logger.warning("LLaVA non disponible (installer ollama)")


@dataclass
class Segment:
    """Représente un segment vidéo avec ses métadonnées."""

    start_time: float  # Temps de début en secondes
    end_time: float  # Temps de fin en secondes
    duration: float  # Durée en secondes
    audio_score: float  # Score audio (0-100)
    visual_score: float  # Score visuel (0-100)
    llava_score: float  # Score LLaVA IA (0-100)
    combined_score: float  # Score combiné (0-100)
    peak_timestamp: float  # Timestamp du pic d'intensité
    llava_description: str = ""  # Description LLaVA (optionnel)

    def __repr__(self):
        return (
            f"Segment({self.start_time:.1f}s-{self.end_time:.1f}s, "
            f"score={self.combined_score:.1f}, llava={self.llava_score:.1f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le segment en dictionnaire."""
        return asdict(self)


class VideoAnalyzer:
    """Analyseur vidéo pour détecter les moments clés."""

    def __init__(
        self,
        video_path: str,
        config: Dict[str, Any],
        cache_manager: Optional[CacheManager] = None,
    ):
        """
        Initialise l'analyseur vidéo.

        Args:
            video_path: Chemin vers le fichier vidéo.
            config: Configuration de l'analyse.
            cache_manager: Gestionnaire de cache (optionnel).
        """
        self.video_path = Path(video_path)
        self.config = config
        self.cache_manager = cache_manager

        if not self.video_path.exists():
            raise FileNotFoundError(f"Fichier vidéo non trouvé : {video_path}")

        self.duration = SystemUtils.get_video_duration(str(video_path))
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.segments: List[Segment] = []

        # Initialiser LLaVA si activé et disponible
        self.llava_analyzer: Optional[LLaVAAnalyzer] = None
        self.use_llava = self.config.get("analysis", {}).get("use_llava", False)

        if self.use_llava and LLAVA_AVAILABLE:
            try:
                model_name = self.config.get("analysis", {}).get("llava_model", "llava:7b")
                self.llava_analyzer = LLaVAAnalyzer(model_name=model_name, config=config)
                logger.info("✓ Analyseur LLaVA activé")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser LLaVA : {e}")
                self.use_llava = False
        elif self.use_llava and not LLAVA_AVAILABLE:
            logger.warning("LLaVA activé dans config mais non disponible")
            self.use_llava = False

        logger.info(f"Analyseur initialisé pour : {self.video_path.name}")
        logger.info(f"Durée vidéo : {SystemUtils.format_duration(self.duration)}")

    def analyze(self, use_cache: bool = True) -> List[Segment]:
        """
        Analyse complète de la vidéo.

        Args:
            use_cache: Utiliser le cache si disponible.

        Returns:
            Liste des segments détectés avec leurs scores.
        """
        # Vérifier le cache
        cache_key = None
        if use_cache and self.cache_manager:
            cache_key = self.cache_manager.generate_cache_key(
                str(self.video_path), self.config
            )
            cached_segments = self.cache_manager.get(cache_key)
            if cached_segments:
                logger.info("Analyse récupérée depuis le cache")
                self.segments = [Segment(**s) for s in cached_segments]
                return self.segments

        logger.info("Début de l'analyse vidéo...")

        # Étape 1 : Extraction et analyse audio
        num_steps = 4 if self.use_llava else 3
        logger.info(f"[1/{num_steps}] Analyse audio...")
        audio_peaks = self._analyze_audio()

        # Étape 2 : Détection de scènes
        logger.info(f"[2/{num_steps}] Détection de scènes...")
        scene_changes = self._detect_scenes()

        # Étape 3 : Analyse LLaVA (si activée)
        llava_highlights = []
        if self.use_llava and self.llava_analyzer:
            logger.info(f"[3/{num_steps}] Analyse LLaVA (IA multimodale)...")
            try:
                sample_rate = self.config.get("analysis", {}).get("llava_frame_sampling", 60)
                prompt_type = self.config.get("analysis", {}).get("llava_prompt_type", "fps")
                max_frames = self.config.get("analysis", {}).get("llava_max_frames", None)

                llava_analyses = self.llava_analyzer.analyze_video_segments(
                    video_path=str(self.video_path),
                    sample_rate=sample_rate,
                    prompt_type=prompt_type,
                    max_frames=max_frames,
                    show_progress=True,
                )

                # Extraire les timestamps importants
                intensity_threshold = self.config.get("analysis", {}).get("llava_intensity_threshold", 60.0)
                llava_highlights = self.llava_analyzer.get_highlight_timestamps(
                    llava_analyses, intensity_threshold=intensity_threshold
                )

                # Stocker les analyses pour usage ultérieur
                self.llava_analyses = llava_analyses

            except Exception as e:
                logger.error(f"Erreur lors de l'analyse LLaVA : {e}")
                logger.warning("Poursuite sans analyse LLaVA")

        # Étape 4 : Création et scoring des segments
        step_num = 4 if self.use_llava else 3
        logger.info(f"[{step_num}/{num_steps}] Création des segments...")
        self.segments = self._create_segments(audio_peaks, scene_changes, llava_highlights)

        # Sauvegarder dans le cache
        if cache_key and self.cache_manager:
            segments_dict = [s.to_dict() for s in self.segments]
            self.cache_manager.set(cache_key, segments_dict)
            logger.info("Résultats d'analyse sauvegardés dans le cache")

        logger.info(f"Analyse terminée : {len(self.segments)} segments détectés")
        return self.segments

    def _analyze_audio(self) -> List[Dict[str, float]]:
        """
        Analyse l'audio pour détecter les pics sonores.

        Returns:
            Liste des pics avec timestamp et intensité.
        """
        import av

        # Charger l'audio depuis la vidéo
        logger.debug("Extraction de l'audio...")
        try:
            container = av.open(str(self.video_path))
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )

            if not audio_stream:
                logger.warning("Aucune piste audio trouvée")
                return []

            # Extraire l'audio
            audio_frames = []
            for frame in container.decode(audio_stream):
                audio_frames.append(frame.to_ndarray())

            container.close()

            if not audio_frames:
                logger.warning("Aucune donnée audio extraite")
                return []

            # Concaténer les frames
            audio_data = np.concatenate(audio_frames, axis=1)
            # Convertir en mono si stéréo
            if audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            else:
                audio_data = audio_data[0]

            self.audio_data = audio_data.astype(np.float32)
            self.sample_rate = audio_stream.sample_rate

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction audio : {e}")
            return []

        # Normaliser l'audio
        self.audio_data = librosa.util.normalize(self.audio_data)

        # Calculer l'énergie RMS (Root Mean Square)
        window_size = int(
            self.config.get("analysis", {}).get("audio_window_size", 0.5)
            * self.sample_rate
        )
        hop_length = window_size // 2

        rms_energy = librosa.feature.rms(
            y=self.audio_data, frame_length=window_size, hop_length=hop_length
        )[0]

        # Détecter les pics d'énergie
        threshold = self.config.get("analysis", {}).get("audio_threshold", 0.7)
        mean_energy = np.mean(rms_energy)
        std_energy = np.std(rms_energy)
        peak_threshold = mean_energy + (threshold * std_energy)

        # Trouver les pics
        peaks = []
        for i, energy in enumerate(rms_energy):
            if energy > peak_threshold:
                timestamp = librosa.frames_to_time(
                    i, sr=self.sample_rate, hop_length=hop_length
                )
                # Normaliser le score (0-100)
                normalized_score = min(100, (energy / peak_threshold) * 50)
                peaks.append({"timestamp": timestamp, "intensity": normalized_score})

        logger.debug(f"Détection de {len(peaks)} pics audio")
        return peaks

    def _detect_scenes(self) -> List[float]:
        """
        Détecte les changements de scène dans la vidéo.

        Returns:
            Liste des timestamps de changements de scène.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0:
            logger.error("Impossible de lire le FPS de la vidéo")
            cap.release()
            return []

        scene_threshold = self.config.get("analysis", {}).get("scene_threshold", 30.0)
        scene_changes = []

        # Lire la vidéo par frames
        prev_frame = None
        frame_count = 0

        # Échantillonner toutes les 5 frames pour performance
        sample_rate = 5

        with tqdm(total=total_frames, desc="Analyse scènes", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    # Redimensionner pour performance
                    small_frame = cv2.resize(frame, (320, 180))
                    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                    if prev_frame is not None:
                        # Calculer la différence entre frames
                        frame_diff = cv2.absdiff(gray_frame, prev_frame)
                        diff_mean = np.mean(frame_diff)

                        # Si différence significative, c'est un changement de scène
                        if diff_mean > scene_threshold:
                            timestamp = frame_count / fps
                            scene_changes.append(timestamp)

                    prev_frame = gray_frame

                frame_count += 1
                pbar.update(1)

        cap.release()

        # Fusionner les changements de scène proches (dans 1 seconde)
        merged_changes = []
        last_change = -999
        for change in scene_changes:
            if change - last_change > 1.0:
                merged_changes.append(change)
                last_change = change

        logger.debug(f"Détection de {len(merged_changes)} changements de scène")
        return merged_changes

    def _create_segments(
        self,
        audio_peaks: List[Dict[str, float]],
        scene_changes: List[float],
        llava_highlights: List[float] = None,
    ) -> List[Segment]:
        """
        Crée des segments basés sur les pics audio, changements de scène et analyse LLaVA.

        Args:
            audio_peaks: Liste des pics audio détectés.
            scene_changes: Liste des changements de scène.
            llava_highlights: Liste des timestamps détectés par LLaVA (optionnel).

        Returns:
            Liste de segments scorés.
        """
        if llava_highlights is None:
            llava_highlights = []

        min_duration = self.config.get("analysis", {}).get("min_segment_duration", 5)
        max_duration = self.config.get("analysis", {}).get("max_segment_duration", 45)
        context_before = self.config.get("analysis", {}).get("context_before", 2.5)
        context_after = self.config.get("analysis", {}).get("context_after", 3.0)

        # Poids des scores (ajustés si LLaVA est activé)
        if self.use_llava and llava_highlights:
            audio_weight = self.config.get("analysis", {}).get("audio_weight", 0.3)
            visual_weight = self.config.get("analysis", {}).get("visual_weight", 0.2)
            llava_weight = self.config.get("analysis", {}).get("llava_weight", 0.5)
        else:
            audio_weight = self.config.get("analysis", {}).get("audio_weight", 0.6)
            visual_weight = self.config.get("analysis", {}).get("visual_weight", 0.4)
            llava_weight = 0.0

        segments = []

        # Créer des segments autour des pics audio
        for peak in audio_peaks:
            peak_time = peak["timestamp"]
            audio_intensity = peak["intensity"]

            # Définir les bornes du segment
            start_time = max(0, peak_time - context_before)
            end_time = min(self.duration, peak_time + context_after)
            duration = end_time - start_time

            # Vérifier la durée
            if duration < min_duration or duration > max_duration:
                continue

            # Calculer le score visuel (nombre de changements de scène dans le segment)
            scene_count = sum(
                1 for sc in scene_changes if start_time <= sc <= end_time
            )
            # Normaliser (plus de changements = plus d'action)
            visual_score = min(100, scene_count * 20)

            # Calculer le score LLaVA (si disponible)
            llava_score = 0.0
            llava_description = ""

            if self.use_llava and hasattr(self, 'llava_analyses'):
                # Trouver les analyses LLaVA dans ce segment
                segment_llava = [
                    a for a in self.llava_analyses
                    if start_time <= a.timestamp <= end_time
                ]

                if segment_llava:
                    # Utiliser le score max d'intensité dans le segment
                    llava_score = max(a.intensity_score for a in segment_llava)

                    # Récupérer la description la plus pertinente
                    best_analysis = max(segment_llava, key=lambda a: a.intensity_score)
                    llava_description = best_analysis.description

            # Calculer si LLaVA a détecté ce segment
            llava_detected = any(
                abs(hl - peak_time) < 2.0 for hl in llava_highlights
            )
            if llava_detected and llava_score == 0.0:
                llava_score = 70.0  # Score par défaut si détecté mais pas analysé

            # Score combiné (avec ou sans LLaVA)
            combined_score = (
                (audio_intensity * audio_weight) +
                (visual_score * visual_weight) +
                (llava_score * llava_weight)
            )

            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                audio_score=audio_intensity,
                visual_score=visual_score,
                llava_score=llava_score,
                combined_score=combined_score,
                peak_timestamp=peak_time,
                llava_description=llava_description,
            )

            segments.append(segment)

        # Trier par score décroissant
        segments.sort(key=lambda s: s.combined_score, reverse=True)

        logger.debug(
            f"Segments créés avec scoring : "
            f"audio={audio_weight:.1%}, visual={visual_weight:.1%}, "
            f"llava={llava_weight:.1%}"
        )

        return segments

    def select_best_segments(
        self, target_duration: int, segments: Optional[List[Segment]] = None
    ) -> List[Segment]:
        """
        Sélectionne les meilleurs segments pour atteindre la durée cible.

        Args:
            target_duration: Durée cible en minutes.
            segments: Liste de segments (utilise self.segments par défaut).

        Returns:
            Liste des segments sélectionnés, triés chronologiquement.
        """
        if segments is None:
            segments = self.segments

        if not segments:
            logger.warning("Aucun segment disponible pour la sélection")
            return []

        target_seconds = target_duration * 60
        overlap_margin = self.config.get("analysis", {}).get(
            "overlap_prevention_margin", 1.0
        )

        selected_segments = []
        total_duration = 0.0

        # Si target_duration = 0, prendre TOUS les segments (pas de limite)
        no_limit = (target_duration <= 0)

        if no_limit:
            logger.info("Sélection de TOUS les meilleurs segments (pas de limite de durée)...")
        else:
            logger.info(
                f"Sélection des meilleurs segments pour {target_duration} min ({SystemUtils.format_duration(target_seconds)})..."
            )

        for segment in segments:
            # Vérifier si on a atteint la durée cible (sauf si no_limit)
            if not no_limit and total_duration >= target_seconds:
                break

            # Vérifier le chevauchement avec les segments déjà sélectionnés
            overlaps = False
            for selected in selected_segments:
                # Vérifier si les segments se chevauchent (avec marge)
                if not (
                    segment.end_time + overlap_margin < selected.start_time
                    or segment.start_time - overlap_margin > selected.end_time
                ):
                    overlaps = True
                    break

            if not overlaps:
                # Vérifier si l'ajout de ce segment ne dépasse pas trop la cible (sauf si no_limit)
                if no_limit or total_duration + segment.duration <= target_seconds * 1.1:
                    selected_segments.append(segment)
                    total_duration += segment.duration

        # Trier chronologiquement
        selected_segments.sort(key=lambda s: s.start_time)

        logger.info(
            f"{len(selected_segments)} segments sélectionnés "
            f"(durée totale: {SystemUtils.format_duration(total_duration)})"
        )

        return selected_segments

    def export_metadata(self, output_path: str, segments: List[Segment]) -> None:
        """
        Exporte les métadonnées d'analyse en JSON.

        Args:
            output_path: Chemin du fichier de sortie.
            segments: Liste des segments à exporter.
        """
        import json
        import numpy as np

        def convert_to_python_types(obj):
            """Convertit les types numpy en types Python natifs pour JSON."""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        metadata = {
            "video_path": str(self.video_path),
            "video_duration": float(self.duration),
            "total_segments": len(self.segments),
            "selected_segments": len(segments),
            "segments": [s.to_dict() for s in segments],
            "config": self.config,
        }

        # Convertir tous les types numpy en types Python
        metadata = convert_to_python_types(metadata)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Métadonnées exportées vers {output_path}")


def print_segments_summary(segments: List[Segment]) -> None:
    """
    Affiche un résumé des segments.

    Args:
        segments: Liste des segments à afficher.
    """
    if not segments:
        print("Aucun segment détecté.")
        return

    print(f"\n{'='*80}")
    print(f"{'RÉSUMÉ DES SEGMENTS SÉLECTIONNÉS':^80}")
    print(f"{'='*80}\n")

    total_duration = sum(s.duration for s in segments)

    print(f"{'#':<4} {'Début':<12} {'Fin':<12} {'Durée':<10} {'Score':<10}")
    print(f"{'-'*80}")

    for i, segment in enumerate(segments, 1):
        start = SystemUtils.format_duration(segment.start_time)
        end = SystemUtils.format_duration(segment.end_time)
        duration = f"{segment.duration:.1f}s"
        score = f"{segment.combined_score:.1f}"

        print(f"{i:<4} {start:<12} {end:<12} {duration:<10} {score:<10}")

    print(f"{'-'*80}")
    print(f"Total: {len(segments)} segments | Durée: {SystemUtils.format_duration(total_duration)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Test de l'analyseur
    from src.utils import ConfigManager

    print("=== Test du VideoAnalyzer ===\n")

    # Charger la config
    config_manager = ConfigManager("config.yaml")
    config = config_manager.config

    # Créer un analyseur (nécessite une vraie vidéo pour tester)
    test_video = "test_video.mp4"
    if Path(test_video).exists():
        analyzer = VideoAnalyzer(test_video, config)
        segments = analyzer.analyze()

        print(f"\nSegments détectés : {len(segments)}")
        if segments:
            print("\nTop 5 des meilleurs segments :")
            for i, seg in enumerate(segments[:5], 1):
                print(f"{i}. {seg}")

            # Sélectionner pour 15 min
            selected = analyzer.select_best_segments(target_duration=15)
            print_segments_summary(selected)
    else:
        print(f"Fichier de test {test_video} non trouvé.")
        print("Pour tester, placez une vidéo nommée 'test_video.mp4' dans le dossier.")
