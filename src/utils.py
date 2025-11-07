"""
Fonctions utilitaires pour EditVideo.

Ce module contient des fonctions helper pour :
- Gestion de configuration
- Détection et utilisation du GPU
- Système de cache
- Logging
- Gestion de fichiers
- Utilitaires système
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime

import yaml
import psutil
from loguru import logger
from diskcache import Cache
from colorama import init, Fore, Style

# Initialiser colorama pour Windows
init(autoreset=True)


class ConfigManager:
    """Gestionnaire de configuration pour l'application."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le gestionnaire de configuration.

        Args:
            config_path: Chemin vers le fichier de configuration YAML.
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML.

        Returns:
            Dictionnaire contenant la configuration.

        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas.
            yaml.YAMLError: Si le fichier YAML est mal formé.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Fichier de configuration non trouvé : {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Configuration chargée depuis {self.config_path}")
        return self.config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration avec notation pointée.

        Args:
            key_path: Chemin de la clé (ex: "analysis.audio_threshold").
            default: Valeur par défaut si la clé n'existe pas.

        Returns:
            Valeur de configuration ou valeur par défaut.

        Example:
            >>> config.get("analysis.audio_threshold")
            0.7
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Définit une valeur de configuration.

        Args:
            key_path: Chemin de la clé (ex: "analysis.audio_threshold").
            value: Nouvelle valeur.
        """
        keys = key_path.split(".")
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, output_path: Optional[str] = None) -> None:
        """
        Sauvegarde la configuration dans un fichier YAML.

        Args:
            output_path: Chemin de sortie (par défaut, écrase le fichier source).
        """
        save_path = Path(output_path) if output_path else self.config_path

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration sauvegardée dans {save_path}")


class GPUManager:
    """Gestionnaire pour la détection et l'utilisation du GPU."""

    def __init__(self):
        """Initialise le gestionnaire GPU."""
        self.gpu_available = False
        self.gpu_name = None
        self.cuda_available = False
        self.detect_gpu()

    def detect_gpu(self) -> Tuple[bool, Optional[str]]:
        """
        Détecte la présence d'un GPU NVIDIA compatible CUDA.

        Returns:
            Tuple (GPU disponible, nom du GPU).
        """
        try:
            # Vérifier CUDA via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                self.gpu_name = result.stdout.strip().split("\n")[0]
                self.gpu_available = True
                self.cuda_available = True
                logger.info(f"GPU détecté : {self.gpu_name}")
            else:
                logger.warning("Aucun GPU NVIDIA détecté")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("nvidia-smi non disponible, GPU non détecté")

        return self.gpu_available, self.gpu_name

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Récupère les informations détaillées du GPU.

        Returns:
            Dictionnaire contenant les infos GPU.
        """
        if not self.gpu_available:
            return {"available": False}

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "available": True,
                    "name": values[0],
                    "memory_total_mb": int(values[1]),
                    "memory_free_mb": int(values[2]),
                    "memory_used_mb": int(values[3]),
                    "utilization_percent": int(values[4]),
                }

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos GPU : {e}")

        return {"available": False}

    def is_available(self) -> bool:
        """Retourne True si un GPU est disponible."""
        return self.gpu_available


class CacheManager:
    """Gestionnaire de cache disque pour les analyses vidéo."""

    def __init__(self, cache_dir: str = ".cache", ttl: int = 604800):
        """
        Initialise le gestionnaire de cache.

        Args:
            cache_dir: Répertoire de cache.
            ttl: Durée de vie du cache en secondes (défaut: 7 jours).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache = Cache(str(self.cache_dir))
        self.ttl = ttl

    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calcule le hash MD5 d'un fichier.

        Args:
            file_path: Chemin du fichier.

        Returns:
            Hash MD5 en hexadécimal.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Lire par chunks pour les gros fichiers
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def generate_cache_key(
        self, video_path: str, config_params: Dict[str, Any]
    ) -> str:
        """
        Génère une clé de cache unique basée sur le fichier et la config.

        Args:
            video_path: Chemin de la vidéo.
            config_params: Paramètres de configuration utilisés.

        Returns:
            Clé de cache unique.
        """
        # Hash du fichier vidéo
        file_hash = self.get_file_hash(video_path)

        # Hash des paramètres de config
        config_str = json.dumps(config_params, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Combiner les deux
        cache_key = f"{file_hash}_{config_hash}"
        return cache_key

    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé de cache.

        Returns:
            Valeur mise en cache ou None si non trouvée/expirée.
        """
        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT pour clé : {key[:16]}...")
            else:
                logger.debug(f"Cache MISS pour clé : {key[:16]}...")
            return value
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du cache : {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé de cache.
            value: Valeur à stocker.
            ttl: Durée de vie en secondes (utilise self.ttl par défaut).

        Returns:
            True si succès, False sinon.
        """
        try:
            expire_time = ttl if ttl is not None else self.ttl
            self.cache.set(key, value, expire=expire_time)
            logger.debug(f"Valeur mise en cache avec clé : {key[:16]}...")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'écriture du cache : {e}")
            return False

    def clear(self) -> None:
        """Vide tout le cache."""
        self.cache.clear()
        logger.info("Cache vidé")

    def get_cache_size(self) -> int:
        """
        Retourne la taille du cache en octets.

        Returns:
            Taille en octets.
        """
        total_size = 0
        for item in self.cache_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size


class LoggerSetup:
    """Configuration du système de logging."""

    @staticmethod
    def setup_logger(
        level: str = "INFO",
        log_file: Optional[str] = None,
        colored: bool = True,
    ) -> None:
        """
        Configure le logger de l'application.

        Args:
            level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Chemin du fichier de log (optionnel).
            colored: Activer les couleurs dans la console.
        """
        # Retirer les handlers par défaut
        logger.remove()

        # Format pour console
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Format pour fichier
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )

        # Handler console
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=colored,
        )

        # Handler fichier
        if log_file:
            logger.add(
                log_file,
                format=file_format,
                level=level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
            )

        logger.info(f"Logger configuré (niveau: {level})")


class SystemUtils:
    """Utilitaires système pour informations et ressources."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Récupère les informations système.

        Returns:
            Dictionnaire avec les infos système.
        """
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "python_version": platform.python_version(),
        }

    @staticmethod
    def check_ffmpeg() -> bool:
        """
        Vérifie si FFmpeg est installé et accessible.

        Returns:
            True si FFmpeg est disponible, False sinon.
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """
        Récupère la durée d'une vidéo en secondes.

        Args:
            video_path: Chemin de la vidéo.

        Returns:
            Durée en secondes.
        """
        try:
            import av

            with av.open(video_path) as container:
                duration = float(container.duration / av.time_base)
                return duration
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la durée : {e}")
            return 0.0

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Formate une durée en secondes vers HH:MM:SS.

        Args:
            seconds: Durée en secondes.

        Returns:
            Chaîne formatée.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def format_filesize(size_bytes: int) -> str:
        """
        Formate une taille de fichier en unités lisibles.

        Args:
            size_bytes: Taille en octets.

        Returns:
            Chaîne formatée (ex: "1.5 GB").
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Crée un répertoire s'il n'existe pas.

        Args:
            path: Chemin du répertoire.

        Returns:
            Objet Path du répertoire.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


def print_banner():
    """Affiche le banner de l'application."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  {Fore.YELLOW}███████╗██████╗ ██╗████████╗██╗   ██╗██╗██████╗ ███████╗{Fore.CYAN}║
║  {Fore.YELLOW}██╔════╝██╔══██╗██║╚══██╔══╝██║   ██║██║██╔══██╗██╔════╝{Fore.CYAN}║
║  {Fore.YELLOW}█████╗  ██║  ██║██║   ██║   ██║   ██║██║██║  ██║█████╗  {Fore.CYAN}║
║  {Fore.YELLOW}██╔══╝  ██║  ██║██║   ██║   ╚██╗ ██╔╝██║██║  ██║██╔══╝  {Fore.CYAN}║
║  {Fore.YELLOW}███████╗██████╔╝██║   ██║    ╚████╔╝ ██║██████╔╝███████╗{Fore.CYAN}║
║  {Fore.YELLOW}╚══════╝╚═════╝ ╚═╝   ╚═╝     ╚═══╝  ╚═╝╚═════╝ ╚══════╝{Fore.CYAN}║
║                                                          ║
║     {Fore.WHITE}Éditeur Automatique de Streams Gaming avec IA{Fore.CYAN}      ║
║                  {Fore.GREEN}Version 1.0.0{Fore.CYAN}                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_system_info(gpu_manager: GPUManager):
    """
    Affiche les informations système.

    Args:
        gpu_manager: Instance du GPUManager.
    """
    sys_info = SystemUtils.get_system_info()

    print(f"\n{Fore.CYAN}[Informations Système]{Style.RESET_ALL}")
    print(f"  OS: {sys_info['platform']} {sys_info['platform_version']}")
    print(f"  CPU: {sys_info['processor']}")
    print(f"  Cœurs: {sys_info['cpu_count']} physiques, {sys_info['cpu_count_logical']} logiques")
    print(f"  RAM: {sys_info['ram_available_gb']} GB / {sys_info['ram_total_gb']} GB disponible")

    gpu_info = gpu_manager.get_gpu_info()
    if gpu_info["available"]:
        print(f"\n{Fore.GREEN}[GPU Détecté]{Style.RESET_ALL}")
        print(f"  Modèle: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['memory_free_mb']} MB / {gpu_info['memory_total_mb']} MB disponible")
        print(f"  Utilisation: {gpu_info['utilization_percent']}%")
    else:
        print(f"\n{Fore.YELLOW}[GPU]{Style.RESET_ALL} Non détecté - Traitement CPU uniquement")

    ffmpeg_ok = SystemUtils.check_ffmpeg()
    ffmpeg_status = f"{Fore.GREEN}✓ Installé{Style.RESET_ALL}" if ffmpeg_ok else f"{Fore.RED}✗ Non trouvé{Style.RESET_ALL}"
    print(f"\n{Fore.CYAN}[FFmpeg]{Style.RESET_ALL} {ffmpeg_status}")
    print()


if __name__ == "__main__":
    # Tests unitaires basiques
    print("=== Test des utilitaires ===\n")

    # Test ConfigManager
    print("Test ConfigManager...")
    config = ConfigManager("config.yaml")
    print(f"  Audio threshold: {config.get('analysis.audio_threshold')}")

    # Test GPUManager
    print("\nTest GPUManager...")
    gpu = GPUManager()
    print(f"  GPU disponible: {gpu.is_available()}")
    print(f"  Info GPU: {gpu.get_gpu_info()}")

    # Test SystemUtils
    print("\nTest SystemUtils...")
    print(f"  FFmpeg disponible: {SystemUtils.check_ffmpeg()}")
    print(f"  Format durée: {SystemUtils.format_duration(3665)}")
    print(f"  Format taille: {SystemUtils.format_filesize(1536000000)}")

    # Test CacheManager
    print("\nTest CacheManager...")
    cache = CacheManager()
    cache.set("test_key", {"data": "test"})
    cached_value = cache.get("test_key")
    print(f"  Valeur cache: {cached_value}")

    print("\n=== Tests terminés ===")
