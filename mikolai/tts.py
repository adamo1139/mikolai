"""MikołAI: synteza polskiej mowy (48 kHz) z wag w repo HF albo w katalogu lokalnym."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from .budowa import zbuduj
from .slownik import normalize_text, text_to_indices, wrap_lang_tags

DOMYSLNE_REPO = "cpral/MikolAI-113M-1-Polski-TTS"


def _plik(repo: str, nazwa: str, revision: str | None) -> str:
    if os.path.isdir(repo):
        return os.path.join(repo, nazwa)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, nazwa, revision=revision)


def podziel_na_zdania(tekst: str, max_znakow: int = 250) -> list[str]:
    """Zdania (po . ! ? …) sklejone w kawałki do `max_znakow`; model uczył się na wypowiedziach do ok. 30 s."""
    zdania = [z.strip() for z in re.split(r"(?<=[.!?…])\s+", tekst.strip()) if z.strip()]
    kawalki: list[str] = []
    for z in zdania:
        while len(z) > max_znakow:   # bardzo długie zdanie: tniemy po przecinku albo spacji
            i = max(z.rfind(", ", 0, max_znakow), z.rfind(" ", 0, max_znakow))
            i = i if i > 0 else max_znakow
            kawalki.append(z[: i + 1].strip())
            z = z[i + 1:].strip()
        if kawalki and len(kawalki[-1]) + 1 + len(z) <= max_znakow and not kawalki[-1].endswith(("?", "!")):
            kawalki[-1] = kawalki[-1] + " " + z
        elif z:
            kawalki.append(z)
    return kawalki


class MikolAI:
    """Model TTS. `MikolAI.from_pretrained()` pobiera wagi z HF; `synteza(tekst)` zwraca dźwięk 48 kHz (numpy)."""

    def __init__(self, cfg: dict, wagi: dict, glosy: dict, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.sample_rate = int(cfg["sample_rate"])
        self.hop = int(cfg["hop"])
        self.ccf = int(cfg["ttl"]["chunk_compress_factor"])
        self.latent_dim = int(cfg["ttl"]["latent_dim"])
        self.scale = float(cfg["normalizer_scale"])   # skala latentów (ttl.normalizer.scale w treningu)
        self.lang_tags = bool(cfg.get("lang_tags", False))
        te, vf, uncond, dp, dek = zbuduj(cfg)
        for mod, prefiks in ((te, "text_encoder."), (vf, "vf."), (uncond, "uncond."), (dp, "dp."), (dek, "dekoder.")):
            sd = {k[len(prefiks):]: v for k, v in wagi.items() if k.startswith(prefiks)}
            brak, nadmiar = mod.load_state_dict(sd, strict=False)
            brak = [k for k in brak if not k.startswith("ref_encoder.")]
            if brak or nadmiar:
                raise RuntimeError(f"{prefiks.rstrip('.')}: brak {brak[:5]}, nadmiarowe {nadmiar[:5]}")
        self.te, self.vf, self.uncond, self.dp, self.dekoder = (m.to(self.device).eval().requires_grad_(False)
                                                               for m in (te, vf, uncond, dp, dek))
        self.mean = wagi["stats.mean"].to(self.device).view(1, -1, 1)
        self.std = wagi["stats.std"].to(self.device).view(1, -1, 1)
        self.glosy = {n: {k: v.to(self.device) for k, v in g.items()} for n, g in glosy.items()}

    @classmethod
    def from_pretrained(cls, repo: str = DOMYSLNE_REPO, device: str | None = None, revision: str | None = None):
        """`repo`: id repo na HF albo katalog z config.json, model.safetensors i glosy/*.safetensors."""
        from safetensors.torch import load_file
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(_plik(repo, "config.json", revision), encoding="utf-8"))
        wagi = load_file(_plik(repo, "model.safetensors", revision))
        glosy = {n: load_file(_plik(repo, f"glosy/{n}.safetensors", revision)) for n in cfg["glosy"]}
        return cls(cfg, wagi, glosy, device)

    def _ids(self, tekst: str) -> torch.Tensor:
        ids = text_to_indices(normalize_text(tekst))
        if self.lang_tags:
            ids = wrap_lang_tags(ids, "pl")
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _jedno(self, tekst: str, glos: str, cfg_scale: float, kroki: int, tempo: float, gen) -> torch.Tensor:
        g = self.glosy[glos]
        styl, styl_dp = g["styl"], g["styl_dp"]
        ids = self._ids(tekst)
        maska_t = torch.ones(1, 1, ids.shape[1], device=self.device)
        # długość wypowiedzi (s) -> liczba klatek latentu (porcje hop × ccf próbek)
        sek = torch.exp(self.dp(ids, text_mask=maska_t, style_dp=styl_dp, return_log=True)) / tempo
        porcja = self.hop * self.ccf
        T = int((sek.max() * self.sample_rate + porcja - 1) // porcja)
        dl = int((sek[0] * self.sample_rate).long())
        maska = (torch.arange(T, device=self.device) < (dl + porcja - 1) // porcja).float().view(1, 1, T)
        x = torch.randn(1, self.latent_dim * self.ccf, T, device=self.device, generator=gen) * maska
        h = self.te(ids, styl, text_mask=maska_t)
        u = self.uncond
        u_key = getattr(u, "u_key", None)
        for i in range(kroki):
            t = torch.full((1,), i / kroki, device=self.device)
            xi = x * maska
            v = self.vf(noisy_latent=xi, text_emb=h, style_ttl=styl, latent_mask=maska, text_mask=maska_t, current_step=t)
            if cfg_scale > 1.0:
                v0 = self.vf(noisy_latent=xi, text_emb=u.u_text.expand(1, -1, h.shape[2]), style_ttl=u.u_ref,
                             latent_mask=maska, text_mask=maska_t, current_step=t,
                             style_keys=u_key if u_key is not None else None)
                v = v0 + cfg_scale * (v - v0)
            x = (x + v * maska / kroki) * maska
        z = (x / self.scale if self.scale not in (0.0, 1.0) else x) * self.std + self.mean
        z = z.view(1, self.latent_dim, self.ccf, T).permute(0, 1, 3, 2).flatten(2, 3)   # rozpakowanie ×ccf
        wav = self.dekoder(z).reshape(-1)
        return wav[porcja:-porcja]   # jak w próbkach treningu: bez pierwszej i ostatniej porcji

    def synteza(self, tekst: str, glos: str = "poziomka", cfg: float = 4.0, kroki: int = 16, tempo: float = 1.0,
                ziarno: int | None = 0, przerwa: float = 0.25) -> np.ndarray:
        """Tekst -> dźwięk 48 kHz, float32 w [-1, 1]. Długi tekst jest dzielony na zdania.

        cfg: siła prowadzenia (1 = bez CFG, dwa razy szybciej); kroki: kroki całkowania (16 domyślnie, 8 szybciej);
        tempo: > 1 szybciej; ziarno: powtarzalny wynik (None = losowo); przerwa: cisza między kawałkami w sekundach."""
        if glos not in self.glosy:
            raise ValueError(f"nieznany głos {glos!r}; dostępne: {', '.join(self.glosy)}")
        gen = torch.Generator(device=self.device)
        if ziarno is not None:
            gen.manual_seed(ziarno)
        else:
            gen.seed()
        cisza = torch.zeros(int(przerwa * self.sample_rate), device=self.device)
        czesci = []
        for i, kawalek in enumerate(podziel_na_zdania(tekst)):
            if i:
                czesci.append(cisza)
            czesci.append(self._jedno(kawalek, glos, cfg, kroki, tempo, gen))
        if not czesci:
            return np.zeros(0, dtype=np.float32)
        return torch.cat(czesci).clamp(-1, 1).float().cpu().numpy()

    def zapisz(self, tekst: str, sciezka: str, **kw) -> str:
        """synteza() i zapis do pliku (WAV/FLAC, 48 kHz)."""
        import soundfile as sf
        sf.write(sciezka, self.synteza(tekst, **kw), self.sample_rate)
        return sciezka
