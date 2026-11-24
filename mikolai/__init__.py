"""MikołAI — mały polski TTS (48 kHz, jeden głos). Wagi: huggingface.co/cpral/MikolAI-113M-1-Polski-TTS.

    from mikolai import MikolAI
    tts = MikolAI.from_pretrained()
    tts.zapisz("Dzień dobry, tu MikołAI.", "dzien_dobry.wav")
"""

from .tts import DOMYSLNE_REPO, MikolAI, podziel_na_zdania

__all__ = ["MikolAI", "DOMYSLNE_REPO", "podziel_na_zdania"]
__version__ = "0.1.0"
