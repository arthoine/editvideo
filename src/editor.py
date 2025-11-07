"""
Module de montage vidéo pour EditVideo.

Ce module gère :
- Extraction de segments vidéo
- Assemblage avec transitions
- Ajout d'intro/outro
- Encodage final optimisé (GPU)
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import shutil

from loguru import logger
from tqdm import tqdm
import ffmpeg

from src.analyzer import Segment
from src.utils import SystemUtils, GPUManager


class VideoEditor:
    """Éditeur vidéo pour assembler les segments."""

    def __init__(
        self,
        config: Dict[str, Any],
        gpu_manager: Optional[GPUManager] = None,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialise l'éditeur vidéo.

        Args:
            config: Configuration de l'édition.
            gpu_manager: Gestionnaire GPU (optionnel).
            temp_dir: Répertoire temporaire (optionnel).
        """
        self.config = config
        self.gpu_manager = gpu_manager
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "editvideo"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.use_gpu = config.get("performance", {}).get("use_gpu", True)
        if self.gpu_manager and not self.gpu_manager.is_available():
            self.use_gpu = False
            logger.warning("GPU non disponible, utilisation du CPU")

        logger.info(f"Éditeur vidéo initialisé (GPU: {self.use_gpu})")

    def extract_segment(
        self, video_path: str, segment: Segment, output_path: str
    ) -> bool:
        """
        Extrait un segment d'une vidéo.

        Args:
            video_path: Chemin de la vidéo source.
            segment: Segment à extraire.
            output_path: Chemin de sortie.

        Returns:
            True si succès, False sinon.
        """
        try:
            # Construire la commande FFmpeg
            input_stream = ffmpeg.input(
                video_path, ss=segment.start_time, t=segment.duration
            )

            # Configuration de l'encodage
            output_args = {
                "vcodec": "libx264",
                "acodec": "aac",
                "preset": "ultrafast",  # Rapide pour extraction
            }

            # Utiliser GPU si disponible
            if self.use_gpu:
                output_args["vcodec"] = "h264_nvenc"
                output_args["preset"] = "fast"

            output = ffmpeg.output(input_stream, output_path, **output_args)
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            logger.debug(f"Segment extrait : {output_path}")
            return True

        except ffmpeg.Error as e:
            logger.error(f"Erreur FFmpeg lors de l'extraction : {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du segment : {e}")
            return False

    def create_transition(
        self,
        clip1_path: str,
        clip2_path: str,
        output_path: str,
        transition_type: str = "fade",
        duration: float = 0.5,
    ) -> bool:
        """
        Crée une transition entre deux clips.

        Args:
            clip1_path: Premier clip.
            clip2_path: Second clip.
            output_path: Chemin de sortie.
            transition_type: Type de transition (fade, dissolve, etc.).
            duration: Durée de la transition en secondes.

        Returns:
            True si succès, False sinon.
        """
        try:
            if transition_type == "cut":
                # Pas de transition, juste concaténer
                return True

            # Pour fade/dissolve, utiliser xfade filter
            input1 = ffmpeg.input(clip1_path)
            input2 = ffmpeg.input(clip2_path)

            # Obtenir la durée du premier clip
            probe = ffmpeg.probe(clip1_path)
            clip1_duration = float(probe["format"]["duration"])

            # Offset pour la transition
            offset = clip1_duration - duration

            # Appliquer le filtre xfade
            transition_map = {
                "fade": "fade",
                "dissolve": "dissolve",
                "wipeleft": "wipeleft",
                "wiperight": "wiperight",
            }

            xfade_type = transition_map.get(transition_type, "fade")

            output = ffmpeg.filter(
                [input1, input2],
                "xfade",
                transition=xfade_type,
                duration=duration,
                offset=offset,
            ).output(output_path, vcodec="libx264", acodec="aac")

            ffmpeg.run(output, overwrite_output=True, quiet=True)
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la création de transition : {e}")
            return False

    def concatenate_segments(
        self, segment_paths: List[str], output_path: str, use_transitions: bool = True
    ) -> bool:
        """
        Concatène plusieurs segments vidéo.

        Args:
            segment_paths: Liste des chemins des segments.
            output_path: Chemin de sortie.
            use_transitions: Utiliser des transitions entre segments.

        Returns:
            True si succès, False sinon.
        """
        if not segment_paths:
            logger.error("Aucun segment à concaténer")
            return False

        try:
            # Créer un fichier de liste pour FFmpeg
            concat_file = self.temp_dir / "concat_list.txt"
            with open(concat_file, "w", encoding="utf-8") as f:
                for segment_path in segment_paths:
                    # FFmpeg concat demande des chemmin absolus
                    abs_path = Path(segment_path).resolve()
                    # Échapper les apostrophes pour FFmpeg
                    escaped_path = str(abs_path).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Résolution et FPS
            resolution = self.config.get("editing", {}).get(
                "output_resolution", "1920x1080"
            )
            fps = self.config.get("editing", {}).get("output_fps", 60)
            codec = self.config.get("editing", {}).get("video_codec", "h264")
            preset = self.config.get("editing", {}).get("encoding_preset", "medium")
            crf = self.config.get("editing", {}).get("crf", 20)
            audio_bitrate = self.config.get("editing", {}).get("audio_bitrate", 192)

            # Construire la commande FFmpeg
            cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file)]

            # Filtres vidéo
            vf_filters = [f"scale={resolution}", f"fps={fps}"]
            cmd.extend(["-vf", ",".join(vf_filters)])

            # Encodage
            if self.use_gpu and codec == "h264":
                cmd.extend(["-c:v", "h264_nvenc", "-preset", preset])
            elif codec == "h264":
                cmd.extend(["-c:v", "libx264", "-preset", preset])
            elif codec == "h265":
                if self.use_gpu:
                    cmd.extend(["-c:v", "hevc_nvenc", "-preset", preset])
                else:
                    cmd.extend(["-c:v", "libx265", "-preset", preset])

            cmd.extend(["-crf", str(crf)])

            # Audio
            cmd.extend(["-c:a", "aac", "-b:a", f"{audio_bitrate}k"])

            # Autres options
            cmd.extend(["-movflags", "+faststart"])  # Optimisation web
            cmd.extend(["-y", output_path])  # Écraser si existe

            # Exécuter FFmpeg
            logger.info("Assemblage des segments...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8"
            )

            if result.returncode != 0:
                logger.error(f"Erreur FFmpeg : {result.stderr}")
                return False

            logger.info(f"Vidéo assemblée : {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la concaténation : {e}")
            return False

    def add_intro_outro(
        self,
        video_path: str,
        output_path: str,
        intro_path: Optional[str] = None,
        outro_path: Optional[str] = None,
    ) -> bool:
        """
        Ajoute une intro et/ou outro à la vidéo.

        Args:
            video_path: Vidéo principale.
            output_path: Chemin de sortie.
            intro_path: Chemin de l'intro (optionnel).
            outro_path: Chemin de l'outro (optionnel).

        Returns:
            True si succès, False sinon.
        """
        parts = []

        if intro_path and Path(intro_path).exists():
            parts.append(intro_path)
            logger.info(f"Ajout de l'intro : {intro_path}")

        parts.append(video_path)

        if outro_path and Path(outro_path).exists():
            parts.append(outro_path)
            logger.info(f"Ajout de l'outro : {outro_path}")

        if len(parts) == 1:
            # Pas d'intro/outro, juste copier
            shutil.copy2(video_path, output_path)
            return True

        # Concaténer
        return self.concatenate_segments(parts, output_path, use_transitions=False)

    def edit_video(
        self,
        video_path: str,
        segments: List[Segment],
        output_path: str,
        intro_path: Optional[str] = None,
        outro_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> bool:
        """
        Processus complet d'édition vidéo.

        Args:
            video_path: Vidéo source.
            segments: Liste des segments à extraire.
            output_path: Chemin de sortie final.
            intro_path: Intro (optionnel).
            outro_path: Outro (optionnel).
            show_progress: Afficher la progression.

        Returns:
            True si succès, False sinon.
        """
        logger.info("Début du processus d'édition...")

        if not segments:
            logger.error("Aucun segment fourni pour l'édition")
            return False

        try:
            # Étape 1 : Extraire les segments
            logger.info(f"[1/3] Extraction de {len(segments)} segments...")
            segment_paths = []

            iterator = (
                tqdm(segments, desc="Extraction", unit="segment")
                if show_progress
                else segments
            )

            for i, segment in enumerate(iterator):
                segment_output = self.temp_dir / f"segment_{i:04d}.mp4"
                success = self.extract_segment(
                    video_path, segment, str(segment_output)
                )

                if success:
                    segment_paths.append(str(segment_output))
                else:
                    logger.warning(f"Échec de l'extraction du segment {i}")

            if not segment_paths:
                logger.error("Aucun segment extrait avec succès")
                return False

            # Étape 2 : Assembler les segments
            logger.info("[2/3] Assemblage des segments...")
            main_video = self.temp_dir / "main_assembled.mp4"
            success = self.concatenate_segments(segment_paths, str(main_video))

            if not success:
                logger.error("Échec de l'assemblage")
                return False

            # Étape 3 : Ajouter intro/outro
            logger.info("[3/3] Ajout intro/outro et finalisation...")
            success = self.add_intro_outro(
                str(main_video), output_path, intro_path, outro_path
            )

            if not success:
                logger.error("Échec de l'ajout intro/outro")
                return False

            # Nettoyer les fichiers temporaires
            if not self.config.get("output", {}).get("keep_temp_files", False):
                logger.info("Nettoyage des fichiers temporaires...")
                for segment_path in segment_paths:
                    Path(segment_path).unlink(missing_ok=True)
                main_video.unlink(missing_ok=True)

            logger.info(f"✓ Édition terminée : {output_path}")

            # Afficher les statistiques
            output_size = Path(output_path).stat().st_size
            logger.info(f"Taille du fichier : {SystemUtils.format_filesize(output_size)}")

            return True

        except Exception as e:
            logger.error(f"Erreur lors du processus d'édition : {e}")
            return False

    def cleanup(self) -> None:
        """Nettoie le répertoire temporaire."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug("Répertoire temporaire nettoyé")
        except Exception as e:
            logger.warning(f"Impossible de nettoyer le répertoire temporaire : {e}")


def create_preview(
    video_path: str,
    output_path: str,
    resolution: str = "854x480",
    fps: int = 30,
) -> bool:
    """
    Crée une prévisualisation basse résolution de la vidéo.

    Args:
        video_path: Vidéo source.
        output_path: Chemin de sortie.
        resolution: Résolution de prévisualisation.
        fps: FPS de prévisualisation.

    Returns:
        True si succès, False sinon.
    """
    try:
        logger.info(f"Création de la prévisualisation : {output_path}")

        input_stream = ffmpeg.input(video_path)
        output = ffmpeg.output(
            input_stream,
            output_path,
            vf=f"scale={resolution},fps={fps}",
            vcodec="libx264",
            crf=28,
            preset="veryfast",
            acodec="aac",
            audio_bitrate="96k",
        )

        ffmpeg.run(output, overwrite_output=True, quiet=True)
        logger.info("✓ Prévisualisation créée")
        return True

    except Exception as e:
        logger.error(f"Erreur lors de la création de la prévisualisation : {e}")
        return False


if __name__ == "__main__":
    # Test de l'éditeur
    from src.utils import ConfigManager, GPUManager
    from src.analyzer import Segment

    print("=== Test du VideoEditor ===\n")

    # Charger la config
    config_manager = ConfigManager("config.yaml")
    config = config_manager.config
    gpu_manager = GPUManager()

    # Créer un éditeur
    editor = VideoEditor(config, gpu_manager)

    # Créer des segments de test
    test_segments = [
        Segment(
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            audio_score=85.0,
            visual_score=75.0,
            combined_score=81.0,
            peak_timestamp=15.0,
        ),
        Segment(
            start_time=45.0,
            end_time=60.0,
            duration=15.0,
            audio_score=90.0,
            visual_score=80.0,
            combined_score=86.0,
            peak_timestamp=52.5,
        ),
    ]

    print(f"Segments de test : {len(test_segments)}")
    for seg in test_segments:
        print(f"  - {seg}")

    # Test d'extraction (nécessite une vraie vidéo)
    test_video = "test_video.mp4"
    if Path(test_video).exists():
        output_video = "test_output.mp4"
        success = editor.edit_video(test_video, test_segments, output_video)

        if success:
            print(f"\n✓ Test réussi ! Vidéo créée : {output_video}")
        else:
            print("\n✗ Test échoué")

        editor.cleanup()
    else:
        print(f"Fichier de test {test_video} non trouvé.")
        print("Pour tester, placez une vidéo nommée 'test_video.mp4' dans le dossier.")
