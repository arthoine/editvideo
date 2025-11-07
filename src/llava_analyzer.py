"""
Module d'analyse vidéo avec LLaVA (Large Language and Vision Assistant).

Ce module utilise le modèle LLaVA:7b via Ollama pour analyser intelligemment
les frames vidéo et détecter les moments clés dans les streams gaming.

LLaVA peut comprendre visuellement :
- Actions de gameplay (kills, clutches, aces)
- Éléments HUD (kill feed, score)
- Situations de jeu (1vX, rush, défense)
- Émotions et réactions du streamer
"""

import os
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama non installé. Installez avec : pip install ollama")


@dataclass
class LLaVAAnalysis:
    """Résultat d'analyse LLaVA pour une frame."""

    timestamp: float  # Timestamp de la frame
    description: str  # Description de la scène
    action_detected: bool  # Action détectée (kill, clutch, etc.)
    intensity_score: float  # Score d'intensité (0-100)
    keywords: List[str]  # Mots-clés extraits
    confidence: float  # Confiance du modèle (0-1)


class LLaVAAnalyzer:
    """Analyseur vidéo utilisant LLaVA pour la détection intelligente."""

    def __init__(
        self,
        model_name: str = "llava:7b",
        ollama_host: str = "http://localhost:11434",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise l'analyseur LLaVA.

        Args:
            model_name: Nom du modèle Ollama (défaut: llava:7b).
            ollama_host: URL du serveur Ollama.
            config: Configuration additionnelle.
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama n'est pas installé. Installez avec : pip install ollama"
            )

        self.model_name = model_name
        self.ollama_host = ollama_host
        self.config = config or {}
        self.client = None

        # Vérifier la disponibilité d'Ollama
        self._check_ollama()

        # Prompts pré-définis pour l'analyse gaming
        self.gaming_prompts = {
            "fps": (
                "Analyze this FPS gaming screenshot. Identify if there's an important moment like: "
                "kills, headshots, multi-kills, clutch situations, aces, or intense action. "
                "Rate the intensity from 0-100. Respond in JSON format: "
                '{"action": true/false, "intensity": 0-100, "description": "brief description", '
                '"keywords": ["keyword1", "keyword2"]}'
            ),
            "extract_shooter": (
                "Analyze this extraction shooter game screenshot (Tarkov, Hunt: Showdown, Arc Raiders style). "
                "Detect critical moments: kills, eliminations, successful extractions, rare loot, "
                "intense firefights, survival situations, player deaths, close calls, tactical plays, "
                "AI enemies (mechs/raiders), boss fights. "
                "Rate intensity 0-100 (extraction=90, boss/elite=85, kill=80, firefight=75, loot=50). "
                "JSON format: "
                '{"action": true/false, "intensity": 0-100, "description": "brief", '
                '"keywords": ["keyword1", "keyword2"]}'
            ),
            "general": (
                "Analyze this gaming screenshot. Is this an exciting or important moment? "
                "Look for kills, victories, impressive plays, or intense action. "
                "Rate intensity 0-100. Respond in JSON: "
                '{"action": true/false, "intensity": 0-100, "description": "what\'s happening", '
                '"keywords": ["key", "words"]}'
            ),
            "battle_royale": (
                "Analyze this battle royale screenshot. Detect: eliminations, squad wipes, "
                "final circles, victories, high-kill games, or clutch moments. "
                "Rate intensity 0-100. JSON format: "
                '{"action": true/false, "intensity": 0-100, "description": "brief", '
                '"keywords": ["tags"]}'
            ),
        }

        logger.info(f"Analyseur LLaVA initialisé (modèle: {model_name})")

    def _check_ollama(self) -> bool:
        """
        Vérifie qu'Ollama est accessible et que le modèle est disponible.

        Returns:
            True si Ollama est opérationnel, False sinon.
        """
        try:
            # Vérifier la connexion
            response = ollama.list()

            # Support de différentes structures de réponse
            if isinstance(response, dict):
                models = response.get('models', [])
            else:
                models = response if isinstance(response, list) else []

            # Extraire les noms de modèles de façon robuste
            available_models = []
            for model in models:
                try:
                    if isinstance(model, dict):
                        model_name = model.get('name') or model.get('model') or str(model)
                    else:
                        model_name = str(model)
                    available_models.append(model_name)
                except:
                    pass

            if self.model_name not in available_models:
                logger.warning(
                    f"Modèle {self.model_name} non trouvé. "
                    f"Disponibles: {', '.join(available_models) if available_models else 'aucun'}"
                )
                logger.info(f"Téléchargez avec: ollama pull {self.model_name}")
                return False

            logger.info(f"✓ Modèle {self.model_name} disponible")
            return True

        except Exception as e:
            logger.error(f"Impossible de se connecter à Ollama: {e}")
            logger.info("Assurez-vous qu'Ollama est démarré: ollama serve")
            return False

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode une image numpy en base64 pour LLaVA.

        Args:
            image: Image numpy array (BGR).

        Returns:
            Image encodée en base64.
        """
        # Convertir BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convertir en PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Encoder en base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        prompt_type: str = "fps",
    ) -> Optional[LLaVAAnalysis]:
        """
        Analyse une frame vidéo avec LLaVA.

        Args:
            frame: Frame vidéo (numpy array BGR).
            timestamp: Timestamp de la frame en secondes.
            prompt_type: Type de prompt ("fps", "general", "battle_royale").

        Returns:
            Résultat d'analyse ou None en cas d'erreur.
        """
        try:
            # Encoder l'image
            image_b64 = self._encode_image(frame)

            # Sélectionner le prompt
            prompt = self.gaming_prompts.get(prompt_type, self.gaming_prompts["general"])

            # Appeler LLaVA via Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False,
            )

            # Extraire la réponse
            response_text = response.get('response', '').strip()

            # Parser la réponse JSON
            analysis = self._parse_llava_response(response_text, timestamp)

            return analysis

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse LLaVA (t={timestamp:.1f}s): {e}")
            return None

    def _parse_llava_response(
        self, response: str, timestamp: float
    ) -> LLaVAAnalysis:
        """
        Parse la réponse de LLaVA et extrait les informations.

        Args:
            response: Réponse texte de LLaVA.
            timestamp: Timestamp de la frame.

        Returns:
            Objet LLaVAAnalysis.
        """
        import json
        import re

        # Essayer de parser comme JSON
        try:
            # Extraire le JSON de la réponse (peut contenir du texte avant/après)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Pas de JSON trouvé dans la réponse")

            return LLaVAAnalysis(
                timestamp=timestamp,
                description=data.get('description', response[:100]),
                action_detected=data.get('action', False),
                intensity_score=float(data.get('intensity', 0)),
                keywords=data.get('keywords', []),
                confidence=0.9,  # Assume haute confiance si JSON valide
            )

        except (json.JSONDecodeError, ValueError):
            # Fallback : analyse textuelle
            logger.debug(f"JSON invalide, analyse textuelle de: {response[:100]}")

            # Détecter les mots-clés d'action
            action_keywords = [
                'kill', 'death', 'headshot', 'ace', 'clutch', 'victory',
                'elimination', 'frag', 'multikill', 'teamwipe', 'win'
            ]

            response_lower = response.lower()
            detected_keywords = [kw for kw in action_keywords if kw in response_lower]
            action_detected = len(detected_keywords) > 0

            # Estimer l'intensité basée sur les mots-clés
            intensity = min(100, len(detected_keywords) * 25) if action_detected else 20

            return LLaVAAnalysis(
                timestamp=timestamp,
                description=response[:200],
                action_detected=action_detected,
                intensity_score=float(intensity),
                keywords=detected_keywords,
                confidence=0.6,  # Confiance plus faible pour fallback
            )

    def analyze_video_segments(
        self,
        video_path: str,
        sample_rate: int = 30,
        prompt_type: str = "fps",
        max_frames: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[LLaVAAnalysis]:
        """
        Analyse une vidéo complète en échantillonnant des frames.

        Args:
            video_path: Chemin de la vidéo.
            sample_rate: Analyser une frame toutes les N frames.
            prompt_type: Type de prompt pour l'analyse.
            max_frames: Nombre max de frames à analyser (None = toutes).
            show_progress: Afficher la barre de progression.

        Returns:
            Liste des analyses par frame.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0:
            logger.error("Impossible de lire le FPS de la vidéo")
            cap.release()
            return []

        analyses = []
        frame_count = 0
        analyzed_count = 0

        # Calculer le nombre de frames à analyser
        frames_to_analyze = (total_frames // sample_rate) + 1
        if max_frames:
            frames_to_analyze = min(frames_to_analyze, max_frames)

        logger.info(
            f"Analyse LLaVA : {frames_to_analyze} frames "
            f"(1 frame / {sample_rate}, FPS={fps:.1f})"
        )

        iterator = tqdm(
            total=frames_to_analyze,
            desc="Analyse LLaVA",
            unit="frame",
        ) if show_progress else None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Analyser seulement toutes les N frames
                if frame_count % sample_rate == 0:
                    timestamp = frame_count / fps

                    # Analyser la frame
                    analysis = self.analyze_frame(frame, timestamp, prompt_type)

                    if analysis:
                        analyses.append(analysis)

                    analyzed_count += 1
                    if iterator:
                        iterator.update(1)

                    # Limiter si max_frames spécifié
                    if max_frames and analyzed_count >= max_frames:
                        break

                frame_count += 1

        finally:
            cap.release()
            if iterator:
                iterator.close()

        logger.info(f"✓ {len(analyses)} frames analysées avec LLaVA")

        # Statistiques
        action_frames = sum(1 for a in analyses if a.action_detected)
        avg_intensity = np.mean([a.intensity_score for a in analyses]) if analyses else 0

        logger.info(f"  - Frames avec action: {action_frames}/{len(analyses)}")
        logger.info(f"  - Intensité moyenne: {avg_intensity:.1f}/100")

        return analyses

    def get_highlight_timestamps(
        self,
        analyses: List[LLaVAAnalysis],
        intensity_threshold: float = 60.0,
        min_confidence: float = 0.5,
    ) -> List[float]:
        """
        Extrait les timestamps des moments forts détectés par LLaVA.

        Args:
            analyses: Liste des analyses LLaVA.
            intensity_threshold: Seuil d'intensité (0-100).
            min_confidence: Confiance minimale requise.

        Returns:
            Liste des timestamps de moments forts.
        """
        highlights = []

        for analysis in analyses:
            if (
                analysis.action_detected
                and analysis.intensity_score >= intensity_threshold
                and analysis.confidence >= min_confidence
            ):
                highlights.append(analysis.timestamp)

        logger.info(
            f"✓ {len(highlights)} moments forts détectés par LLaVA "
            f"(seuil: {intensity_threshold})"
        )

        return highlights


def test_llava_connection():
    """Teste la connexion à Ollama et LLaVA."""
    print("=== Test de connexion LLaVA ===\n")

    try:
        # Lister les modèles
        print("1. Vérification des modèles disponibles...")
        response = ollama.list()

        # Support de différentes structures de réponse
        if isinstance(response, dict):
            models = response.get('models', [])
        else:
            models = response if isinstance(response, list) else []

        print(f"   Modèles installés: {len(models)}")

        # Afficher les modèles de façon robuste
        llava_found = False
        for model in models:
            try:
                # Gérer différents formats de modèle
                if isinstance(model, dict):
                    model_name = model.get('name') or model.get('model') or str(model)
                else:
                    model_name = str(model)

                print(f"   - {model_name}")

                # Vérifier si c'est llava
                if 'llava' in model_name.lower():
                    llava_found = True
                    llava_model_name = model_name
            except Exception as e:
                print(f"   - [erreur lecture modèle: {e}]")

        # Vérifier LLaVA
        if llava_found:
            print(f"\n✓ LLaVA détecté: {llava_model_name}")
        else:
            print("\n✗ LLaVA non installé")
            print("   Installez avec: ollama pull llava:7b")
            return False

        print("\n2. Test d'analyse d'image...")
        # Créer une image de test
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            test_image, "TEST FRAME", (200, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        analyzer = LLaVAAnalyzer(model_name="llava:7b")
        result = analyzer.analyze_frame(test_image, 0.0, "general")

        if result:
            print(f"✓ Analyse réussie!")
            print(f"   Description: {result.description}")
            print(f"   Intensité: {result.intensity_score}/100")
            return True
        else:
            print("✗ Échec de l'analyse")
            return False

    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        print("\nAssurez-vous qu'Ollama est démarré:")
        print("  Windows: ollama serve")
        return False


if __name__ == "__main__":
    # Test du module
    if OLLAMA_AVAILABLE:
        test_llava_connection()
    else:
        print("❌ Ollama non installé")
        print("Installez avec: pip install ollama")
        print("Puis téléchargez LLaVA: ollama pull llava:7b")
