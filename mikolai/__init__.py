"""MikołAI — mały polski TTS (48 kHz, jeden głos). Wagi: huggingface.co/cpral/MikolAI-109M-1-Polski-TTS.

    from mikolai import MikolAI
    tts = MikolAI.from_pretrained()
    tts.zapisz("Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?", "wizyta.wav")
"""

from .tts import DOMYSLNE_REPO, MAKS_ZNAKOW, PRZERWY, MikolAI, podziel, podziel_na_zdania, sklej

__all__ = ["MikolAI", "DOMYSLNE_REPO", "MAKS_ZNAKOW", "PRZERWY", "podziel", "podziel_na_zdania", "sklej"]
__version__ = "0.1.0"
