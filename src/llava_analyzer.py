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
                "Analyze this competitive FPS game screenshot (Valorant, CS:GO, COD, Apex).\n\n"

                "KEY MOMENTS TO DETECT:\n"
                "1. ACE (intensity: 95-100): Single player eliminating entire enemy team (5+ kills)\n"
                "2. MULTI-KILLS (intensity: 85-95): Double kill, triple kill, quad kill notifications\n"
                "3. CLUTCH (intensity: 80-95): 1v2 or higher situations, last player alive, round-winning plays\n"
                "4. HEADSHOTS (intensity: 75-85): Headshot kills, headshot icons, precision eliminations\n"
                "5. KILLS (intensity: 70-80): Standard eliminations, kill feed showing +1, enemy eliminated\n"
                "6. INTENSE COMBAT (intensity: 60-75): Active gunfights, multiple enemies visible, explosions\n"
                "7. ABILITIES (intensity: 50-70): Ultimate abilities used, special moves, grenades\n\n"

                "VISUAL CUES:\n"
                "- Kill feed: Shows recent eliminations, player names, weapon icons\n"
                "- HUD: Health, ammo, ability cooldowns, score, round timer\n"
                "- Notifications: '+100', 'ELIMINATED', 'HEADSHOT', 'DOUBLE KILL'\n"
                "- Crosshair feedback: Hit markers, damage numbers, kill confirmations\n\n"

                "SCORING:\n"
                "- Multiple kills in view = higher score\n"
                "- Clutch situations (1vX) = +15 intensity\n"
                "- Low time/health + success = +10 intensity\n"
                "- Quiet moments = 0-40, Active fights = 40-80, Epic plays = 80-100\n\n"

                "OUTPUT RULES:\n"
                "Respond with ONLY valid JSON. No markdown, no text before/after.\n"
                "Do NOT use ```json markers. Output must start with { and end with }\n"
                '{"action": true, "intensity": 75, "description": "what is happening", "keywords": ["key", "words"]}'
            ),
            "extract_shooter": (
                "You are analyzing an extraction shooter game (Tarkov, Hunt: Showdown, Arc Raiders). "
                "These are tactical PvPvE games where players loot, fight AI/players, and must extract to keep items.\n\n"

                "CRITICAL MOMENTS TO DETECT:\n"
                "1. EXTRACTION (intensity: 90-100): Player reaching extraction point, escape helicopter/vehicle visible, "
                "extraction countdown, successful escape animation, 'extraction successful' message\n"
                "2. BOSS/ELITE KILLS (intensity: 85-95): Defeating large AI bosses, mechs, elite enemies, "
                "rare enemy death animations, boss health bar depleted\n"
                "3. PVP KILLS (intensity: 80-90): Eliminating other players, kill notifications, kill feed showing player names, "
                "multiple kills in quick succession, headshots, long-range sniper kills\n"
                "4. INTENSE FIREFIGHTS (intensity: 75-85): Active combat with gunfire effects, muzzle flashes, "
                "explosions, taking damage (red screen edges), low health warnings, healing under fire\n"
                "5. CLOSE CALLS (intensity: 70-80): Near-death survival, very low health bar, reviving teammates, "
                "narrow escape from danger, last-second extraction\n"
                "6. TACTICAL PLAYS (intensity: 65-75): Flanking enemies, perfect positioning, clutch moments, "
                "1vX situations, using environment strategically\n"
                "7. RARE LOOT (intensity: 50-70): Legendary/epic items, rare weapon pickups, valuable loot containers, "
                "full inventory of high-tier items, special item notifications\n"
                "8. AI COMBAT (intensity: 45-65): Fighting regular AI enemies (raiders, mobs), clearing areas\n\n"

                "VISUAL INDICATORS:\n"
                "- HUD elements: health bars, ammo counter, kill feed, notifications\n"
                "- Screen effects: damage vignettes, blood splatter, healing effects, death screens\n"
                "- Player actions: looting animations, extraction timers, weapon firing\n"
                "- Environmental: extraction zones, boss arenas, combat areas\n\n"

                "SCORING GUIDELINES:\n"
                "- Multiple simultaneous factors = higher score (e.g., low health + kill + extraction = 95)\n"
                "- Context matters: killing while extracting is more intense than safe kills\n"
                "- Quiet moments (looting, walking) = 0-30\n"
                "- Moderate action (single AI fights) = 30-60\n"
                "- High action (PvP, boss fights, extraction) = 60-100\n\n"

                "RESPONSE FORMAT - CRITICAL:\n"
                "Respond with ONLY valid JSON. No markdown, no explanation, no code blocks.\n"
                "Do NOT use ```json or ``` markers.\n"
                "Output MUST start with { and end with }\n\n"

                "Required format:\n"
                '{"action": true, "intensity": 85, "description": "detailed description", "keywords": ["keyword1", "keyword2"]}\n\n'

                "Set action=true for intensity >= 50. Be precise with intensity scoring."
            ),
            "general": (
                "Analyze this gaming screenshot. Is this an exciting or important moment? "
                "Look for kills, victories, impressive plays, or intense action. "
                "Rate intensity 0-100. Respond in JSON: "
                '{"action": true/false, "intensity": 0-100, "description": "what\'s happening", '
                '"keywords": ["key", "words"]}'
            ),
            "battle_royale": (
                "Analyze this battle royale game screenshot (Apex, Warzone, Fortnite, PUBG).\n\n"

                "CRITICAL MOMENTS:\n"
                "1. VICTORY ROYALE (intensity: 100): Win screen, 'Champion', '#1', victory animation\n"
                "2. FINAL CIRCLE (intensity: 85-95): Top 3-5 players, small circle, end-game situation\n"
                "3. SQUAD WIPES (intensity: 80-90): Eliminating entire enemy squad, 'Squad eliminated'\n"
                "4. HIGH-KILL STREAK (intensity: 75-85): Kill count 5+, rapid eliminations, kill leader\n"
                "5. HOT DROP COMBAT (intensity: 70-80): Early game intense fights, multiple squads nearby\n"
                "6. CLUTCH REVIVES (intensity: 65-75): Reviving teammates under fire, last player standing\n"
                "7. ELIMINATIONS (intensity: 60-75): Knocking/eliminating enemies, kill notifications\n"
                "8. SUPPLY DROPS (intensity: 50-65): Care packages, legendary loot, airdrops\n\n"

                "VISUAL INDICATORS:\n"
                "- Kill count displayed, players remaining counter\n"
                "- Circle/zone visible on map, storm closing in\n"
                "- Notifications: eliminations, damage dealt, revives\n"
                "- Rarity indicators: legendary/epic items (gold, purple)\n\n"

                "SCORING RULES:\n"
                "- Final circle + combat = maximum intensity\n"
                "- Kill count visible in HUD adds +10 per milestone (5, 10, 15 kills)\n"
                "- Early game = 40-70, Mid game = 50-80, End game = 70-100\n\n"

                "OUTPUT RULES:\n"
                "Respond with ONLY valid JSON. No markdown, no explanation.\n"
                "Do NOT use ```json markers. Start with { and end with }\n"
                '{"action": true, "intensity": 85, "description": "specific action", "keywords": ["key", "words"]}'
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

            # Appeler LLaVA via Ollama avec mode JSON forcé et optimisations
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False,  # Attendre réponse complète (requis pour JSON parsing)
                format="json",  # Force la sortie en JSON valide
                options={
                    "temperature": 0.2,      # Très déterministe pour JSON consistant
                    "top_p": 0.9,            # Limite aux réponses probables
                    "num_predict": 200,      # Max ~200 tokens (notre JSON fait ~80-120)
                    "repeat_penalty": 1.1,   # Évite keywords dupliqués
                },
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
        Parse la réponse de LLaVA avec parser ultra-robuste et validation.

        Args:
            response: Réponse texte de LLaVA.
            timestamp: Timestamp de la frame.

        Returns:
            Objet LLaVAAnalysis.
        """
        import json
        import re

        # Essayer de parser comme JSON avec multiples stratégies
        for strategy_num, strategy in enumerate([
            self._parse_strategy_1_clean,
            self._parse_strategy_2_extract,
            self._parse_strategy_3_repair,
        ], 1):
            try:
                data = strategy(response)

                # Validation du schéma
                if self._validate_json_schema(data):
                    return LLaVAAnalysis(
                        timestamp=timestamp,
                        description=data.get('description', '')[:500],  # Limit length
                        action_detected=bool(data.get('action', False)),
                        intensity_score=float(max(0, min(100, data.get('intensity', 0)))),  # Clamp 0-100
                        keywords=data.get('keywords', [])[:10],  # Limit keywords
                        confidence=0.95,  # Haute confiance avec validation
                    )
                else:
                    logger.debug(f"Strategy {strategy_num} returned invalid schema at t={timestamp:.1f}s")

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                logger.debug(f"Strategy {strategy_num} failed at t={timestamp:.1f}s: {str(e)[:50]}")
                continue

        # Fallback : analyse textuelle si toutes les stratégies échouent
        logger.debug(f"All parsing strategies failed at t={timestamp:.1f}s, using keyword fallback")
        return self._fallback_keyword_analysis(response, timestamp)

    def _parse_strategy_1_clean(self, response: str) -> dict:
        """Stratégie 1: Nettoyage standard et parsing direct."""
        import json
        import re

        cleaned = response.strip()

        # Retirer markdown
        cleaned = re.sub(r'^```(?:json|JSON)?\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

        # Extraire JSON entre accolades
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')

        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        # Corrections courantes
        cleaned = cleaned.replace('{{', '{').replace('}}', '}')
        cleaned = cleaned.replace('\n', ' ')  # Single line

        return json.loads(cleaned)

    def _parse_strategy_2_extract(self, response: str) -> dict:
        """Stratégie 2: Extraction regex aggressive."""
        import json
        import re

        # Regex pour extraire un objet JSON complet
        patterns = [
            r'\{[^{}]*"action"[^{}]*"intensity"[^{}]*"description"[^{}]*"keywords"[^{}]*\}',
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested braces
            r'\{.+?\}',  # Lazy match
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(0)
                # Clean up
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                return json.loads(json_str)

        raise ValueError("No JSON object found")

    def _parse_strategy_3_repair(self, response: str) -> dict:
        """Stratégie 3: Réparation JSON cassé."""
        import json
        import re

        # Extraire le contenu entre {}
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            raise ValueError("No braces found")

        json_str = match.group(0)

        # Réparations agressives
        repairs = [
            (r',\s*}', '}'),  # Trailing comma
            (r',\s*]', ']'),  # Trailing comma in array
            (r'}\s*{', '},{'),  # Missing comma between objects
            (r'"\s*:\s*([^",\[\]{}]+)\s*,', r'": "\1",'),  # Unquoted values
            (r'"\s*:\s*([^",\[\]{}]+)\s*}', r'": "\1"}'),  # Unquoted values at end
            (r"'", '"'),  # Single to double quotes
        ]

        for pattern, replacement in repairs:
            json_str = re.sub(pattern, replacement, json_str)

        return json.loads(json_str)

    def _validate_json_schema(self, data: dict) -> bool:
        """Valide que le JSON contient les champs requis avec les bons types."""
        try:
            # Champs requis
            if not isinstance(data, dict):
                return False

            # Vérifier action (bool)
            if 'action' not in data or not isinstance(data['action'], bool):
                return False

            # Vérifier intensity (number)
            if 'intensity' not in data:
                return False
            intensity = data['intensity']
            if not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 100:
                return False

            # Vérifier description (string)
            if 'description' not in data or not isinstance(data['description'], str):
                return False

            # Vérifier keywords (array)
            if 'keywords' not in data or not isinstance(data['keywords'], list):
                return False

            # Tous les keywords doivent être des strings
            if not all(isinstance(kw, str) for kw in data['keywords']):
                return False

            return True

        except Exception:
            return False

    def _fallback_keyword_analysis(self, response: str, timestamp: float) -> LLaVAAnalysis:
        """Fallback : analyse par mots-clés si le JSON parsing échoue."""
        action_keywords = [
            'kill', 'death', 'headshot', 'ace', 'clutch', 'victory',
            'elimination', 'frag', 'multikill', 'teamwipe', 'win',
            'extraction', 'boss', 'elite', 'firefight', 'combat',
            'mech', 'raid', 'loot', 'extract'
        ]

        response_lower = response.lower()
        detected_keywords = [kw for kw in action_keywords if kw in response_lower]
        action_detected = len(detected_keywords) > 0

        # Estimer l'intensité
        intensity = min(100, len(detected_keywords) * 20) if action_detected else 15

        return LLaVAAnalysis(
            timestamp=timestamp,
            description=response[:300],
            action_detected=action_detected,
            intensity_score=float(intensity),
            keywords=detected_keywords[:5],
            confidence=0.5,  # Faible confiance pour fallback
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
