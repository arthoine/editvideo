#!/usr/bin/env python3
"""
EditVideo - Éditeur Automatique de Streams Gaming avec IA

Point d'entrée CLI pour l'application.

Usage:
    python main.py --input stream.mp4 --duration 15 --output highlights.mp4

Auteur: EditVideo Team
Licence: MIT
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time

import click
from loguru import logger

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    ConfigManager,
    GPUManager,
    CacheManager,
    LoggerSetup,
    SystemUtils,
    print_banner,
    print_system_info,
)
from src.analyzer import VideoAnalyzer, print_segments_summary
from src.editor import VideoEditor, create_preview


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Chemin vers le fichier vidéo source (VOD).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Chemin du fichier de sortie (défaut: highlights.mp4).",
)
@click.option(
    "--duration",
    "-d",
    default=15,
    type=int,
    help="Durée cible de la vidéo finale en minutes (défaut: 15).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config.yaml",
    type=click.Path(exists=True),
    help="Chemin vers le fichier de configuration (défaut: config.yaml).",
)
@click.option(
    "--intro",
    type=click.Path(exists=True),
    default=None,
    help="Vidéo d'introduction à ajouter au début (optionnel).",
)
@click.option(
    "--outro",
    type=click.Path(exists=True),
    default=None,
    help="Vidéo de fin à ajouter à la fin (optionnel).",
)
@click.option(
    "--preset",
    type=click.Choice([
        "fps_intense",
        "fps_tactique",
        "battle_royale",
        "moba",
        "extract_shooter",
        "quick_extract",
        "quick_fps",
        "debug_permissive"
    ]),
    default=None,
    help="Preset de configuration par type de jeu.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Affichage détaillé des logs (mode debug).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Désactiver le cache d'analyse.",
)
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Désactiver l'utilisation du GPU (forcer CPU).",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Générer également une prévisualisation basse résolution.",
)
@click.option(
    "--export-metadata",
    is_flag=True,
    help="Exporter les métadonnées d'analyse en JSON.",
)
@click.option(
    "--show-system-info",
    is_flag=True,
    help="Afficher les informations système et quitter.",
)
def main(
    input_path: str,
    output_path: str,
    duration: int,
    config_path: str,
    intro: str,
    outro: str,
    preset: str,
    verbose: bool,
    no_cache: bool,
    no_gpu: bool,
    preview: bool,
    export_metadata: bool,
    show_system_info: bool,
):
    """
    EditVideo - Éditeur Automatique de Streams Gaming avec IA.

    Analyse automatiquement un VOD de stream et génère une vidéo montée
    avec les meilleurs moments, optimisée pour YouTube.
    """
    # Afficher le banner
    print_banner()

    # Charger la configuration
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.config
    except Exception as e:
        click.echo(f"❌ Erreur lors du chargement de la configuration : {e}", err=True)
        sys.exit(1)

    # Appliquer le preset si spécifié
    if preset:
        if preset in config.get("presets", {}):
            preset_config = config["presets"][preset]
            # Fusionner le preset avec la config
            for section, values in preset_config.items():
                if section in config:
                    config[section].update(values)
            logger.info(f"Preset '{preset}' appliqué")
        else:
            click.echo(f"⚠️  Preset '{preset}' non trouvé dans la config", err=True)

    # Configurer le logger
    log_level = "DEBUG" if verbose else config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("log_file") if config.get("logging", {}).get("log_to_file", True) else None
    colored = config.get("logging", {}).get("colored_output", True)

    LoggerSetup.setup_logger(level=log_level, log_file=log_file, colored=colored)

    # Initialiser le GPU manager
    gpu_manager = GPUManager()

    # Afficher les infos système et quitter si demandé
    if show_system_info:
        print_system_info(gpu_manager)
        sys.exit(0)

    # Désactiver GPU si demandé
    if no_gpu:
        config["performance"]["use_gpu"] = False
        logger.info("GPU désactivé par l'utilisateur")

    # Vérifier FFmpeg
    if not SystemUtils.check_ffmpeg():
        click.echo("❌ FFmpeg n'est pas installé ou non trouvé dans le PATH.", err=True)
        click.echo("   Téléchargez FFmpeg depuis : https://ffmpeg.org/download.html", err=True)
        sys.exit(1)

    # Afficher les infos système
    print_system_info(gpu_manager)

    # Définir le chemin de sortie
    if output_path is None:
        output_name = config.get("output", {}).get("default_filename", "highlights.mp4")
        if config.get("output", {}).get("include_timestamp", False):
            timestamp = datetime.now().strftime(
                config.get("output", {}).get("timestamp_format", "%Y%m%d_%H%M%S")
            )
            name_parts = output_name.rsplit(".", 1)
            output_name = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        output_path = output_name

    # Créer le dossier de sortie si nécessaire
    output_dir = Path(output_path).parent
    if output_dir != Path(".") and config.get("output", {}).get("create_output_dir", True):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialiser le cache
    cache_manager = None
    if config.get("performance", {}).get("cache_enabled", True) and not no_cache:
        cache_dir = config.get("performance", {}).get("cache_dir", ".cache")
        cache_ttl = config.get("performance", {}).get("cache_ttl", 168) * 3600  # Heures → secondes
        cache_manager = CacheManager(cache_dir, cache_ttl)
        logger.info("Cache activé")
    else:
        logger.info("Cache désactivé")

    # Début du traitement
    start_time = time.time()

    try:
        # ===== PHASE 1: ANALYSE =====
        logger.info("=" * 80)
        logger.info("PHASE 1 : ANALYSE VIDÉO")
        logger.info("=" * 80)

        analyzer = VideoAnalyzer(
            video_path=input_path,
            config=config,
            cache_manager=cache_manager,
        )

        # Analyser la vidéo
        segments = analyzer.analyze(use_cache=not no_cache)

        if not segments:
            click.echo("❌ Aucun segment détecté dans la vidéo.", err=True)
            sys.exit(1)

        # Sélectionner les meilleurs segments
        selected_segments = analyzer.select_best_segments(
            target_duration=duration,
            segments=segments,
        )

        if not selected_segments:
            click.echo("❌ Aucun segment sélectionné pour la durée cible.", err=True)
            sys.exit(1)

        # Afficher le résumé
        print_segments_summary(selected_segments)

        # Exporter les métadonnées si demandé
        if export_metadata:
            metadata_file = config.get("advanced", {}).get(
                "metadata_file", "analysis_metadata.json"
            )
            analyzer.export_metadata(metadata_file, selected_segments)

        # ===== PHASE 2: MONTAGE =====
        logger.info("=" * 80)
        logger.info("PHASE 2 : MONTAGE VIDÉO")
        logger.info("=" * 80)

        temp_dir = config.get("output", {}).get("temp_dir", "temp")
        editor = VideoEditor(
            config=config,
            gpu_manager=gpu_manager,
            temp_dir=temp_dir,
        )

        # Éditer la vidéo
        show_progress = config.get("logging", {}).get("show_progress_bar", True)
        success = editor.edit_video(
            video_path=input_path,
            segments=selected_segments,
            output_path=output_path,
            intro_path=intro,
            outro_path=outro,
            show_progress=show_progress,
        )

        if not success:
            click.echo("❌ Échec du montage vidéo.", err=True)
            sys.exit(1)

        # Générer une prévisualisation si demandé
        if preview or config.get("advanced", {}).get("generate_preview", False):
            preview_path = str(Path(output_path).with_suffix("")) + "_preview.mp4"
            preview_res = config.get("advanced", {}).get("preview_resolution", "854x480")
            preview_fps = config.get("advanced", {}).get("preview_fps", 30)
            create_preview(output_path, preview_path, preview_res, preview_fps)

        # Nettoyer
        editor.cleanup()

        # ===== PHASE 3: FINALISATION =====
        elapsed_time = time.time() - start_time

        logger.info("=" * 80)
        logger.info("✓ TRAITEMENT TERMINÉ")
        logger.info("=" * 80)

        # Statistiques finales
        if config.get("logging", {}).get("show_stats", True):
            output_file = Path(output_path)
            if output_file.exists():
                file_size = output_file.stat().st_size
                video_duration = SystemUtils.get_video_duration(str(output_file))

                click.echo("\n📊 Statistiques :")
                click.echo(f"  ├─ Fichier : {output_file.name}")
                click.echo(f"  ├─ Taille : {SystemUtils.format_filesize(file_size)}")
                click.echo(f"  ├─ Durée : {SystemUtils.format_duration(video_duration)}")
                click.echo(f"  ├─ Segments : {len(selected_segments)}")
                click.echo(f"  └─ Temps de traitement : {SystemUtils.format_duration(elapsed_time)}")
                click.echo(f"\n✅ Vidéo créée avec succès : {output_path}\n")
            else:
                click.echo("⚠️  Fichier de sortie non trouvé.", err=True)

    except KeyboardInterrupt:
        logger.warning("Traitement interrompu par l'utilisateur")
        click.echo("\n⚠️  Traitement annulé.", err=True)
        sys.exit(130)

    except Exception as e:
        logger.exception(f"Erreur fatale : {e}")
        click.echo(f"\n❌ Erreur : {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
