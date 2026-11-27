"""MikołAI: synteza polskiej mowy (48 kHz) z wag w repo HF albo w katalogu lokalnym."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from .budowa import zbuduj
from .slownik import normalize_text, text_to_indices, wrap_lang_tags

DOMYSLNE_REPO = "cpral/MikolAI-109M-v0.1-Polish-TTS-Preview"
MAKS_PARTIA = 8192


def _plik(repo: str, nazwa: str, revision: str | None) -> str:
    if os.path.isdir(repo):
        return os.path.join(repo, nazwa)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, nazwa, revision=revision)


MAKS_ZNAKOW = 350   # do ok. 18 s mowy; dłuższe kawałki model czyta szybciej (przy ok. 550 znakach o ok. 8%)
PRZERWY = {"przecinek": 0.08, "zdanie": 0.18, "akapit": 0.28}   # cisza wstawiana po kawałku (s), zależnie od końca;
# razem z marginesem po przycięciu (2 × 0,03 s) daje ok. 0,14 / 0,24 / 0,34 s — tyle co pauzy modelu między zdaniami


def podziel(tekst: str, max_znakow: int = MAKS_ZNAKOW) -> list[tuple[str, str]]:
    """Tekst -> [(kawałek, koniec)]. Zdania (po . ! ? …) są sklejane w kawałki do `max_znakow`, także przez granice
    akapitów (akapity rozdziela nowa linia; krótkie kwestie dialogu nie robią się osobnymi kawałkami). Koniec kawałka:
    "akapit" (kończy się razem z akapitem), "zdanie" (na . ! ? …) albo "przecinek" (zbyt długie zdanie przecięte po
    przecinku albo spacji). Do kawałka kończącego się na ? albo ! nic się nie dokleja."""
    zdania: list[tuple[str, str]] = []
    for akapit in (a.strip() for a in tekst.split("\n")):
        czesci = [z.strip() for z in re.split(r"(?<=[.!?…])\s+", akapit) if z.strip()]
        for n, z in enumerate(czesci):
            while len(z) > max_znakow:   # bardzo długie zdanie: najlepiej po przecinku w drugiej połowie, inaczej po spacji
                i = z.rfind(", ", max_znakow // 2, max_znakow)
                i = i + 1 if i > 0 else z.rfind(" ", 0, max_znakow)
                i = i if i > 0 else max_znakow
                zdania.append((z[:i].strip(), "przecinek"))
                z = z[i:].strip()
            if z:
                if n == len(czesci) - 1 and not re.search(r"[.!?…,;:]\W*$", z):
                    z += "."   # akapit bez kropki (np. „Następny dzień”): bez niej zlałby się z następnym zdaniem
                zdania.append((z, "akapit" if n == len(czesci) - 1 else "zdanie"))
    out: list[tuple[str, str]] = []
    for z, k in zdania:
        if out and out[-1][1] != "przecinek" and len(out[-1][0]) + 1 + len(z) <= max_znakow \
                and not out[-1][0].endswith(("?", "!")):
            out[-1] = (out[-1][0] + " " + z, k)
        else:
            out.append((z, k))
    return out


def podziel_na_zdania(tekst: str, max_znakow: int = MAKS_ZNAKOW) -> list[str]:
    """Same kawałki tekstu (bez rodzaju końca), patrz podziel()."""
    return [k for k, _ in podziel(tekst, max_znakow)]


def _przytnij(w: torch.Tensor, sr: int, prog_db: float = -45.0, margines: float = 0.03, zanik: float = 0.01) -> torch.Tensor:
    """Cisza modelu z początku i końca kawałka (ramki 10 ms ciszej niż `prog_db` względem najgłośniejszej ramki),
    z marginesem `margines` s i wygaszeniem `zanik` s na brzegach (bez trzasków)."""
    r = sr // 100
    n = w.shape[0] // r
    if n < 3:
        return w
    e = w[: n * r].reshape(n, r).pow(2).mean(1).add(1e-12).log10().mul(10)
    glos = (e > e.max() + prog_db).nonzero()
    if len(glos) == 0:
        return w
    a = max(0, int(glos[0]) * r - int(margines * sr))
    b = min(w.shape[0], (int(glos[-1]) + 1) * r + int(margines * sr))
    w = w[a:b].clone()
    z = min(int(zanik * sr), w.shape[0] // 4)
    if z > 0:
        rampa = torch.linspace(0, 1, z, device=w.device)
        w[:z] *= rampa
        w[-z:] *= rampa.flip(0)
    return w


def sklej(nagrania: list[torch.Tensor], konce: list[str], sr: int, przerwy: dict | None = None) -> torch.Tensor:
    """Kawałki -> jedno nagranie: cisza modelu z brzegów przycięta, przerwa zależna od końca kawałka (PRZERWY)."""
    przerwy = {**PRZERWY, **(przerwy or {})}
    nagrania = [_przytnij(w, sr) for w in nagrania]
    czesci = []
    for i, (w, k) in enumerate(zip(nagrania, konce)):
        czesci.append(w)
        if i < len(nagrania) - 1:
            czesci.append(torch.zeros(int(przerwy[k] * sr), device=w.device))
    return torch.cat(czesci) if czesci else torch.zeros(0)


def zapisz_audio(audio: np.ndarray, sciezka: str, sr: int = 48000, kbps: int = 64) -> str:
    """Zapis nagrania: .ogg / .opus jako Opus `kbps` kbit/s (6–256; domyślnie 64), .wav / .flac bez strat. Bez
    rozszerzenia dopisywane jest .ogg. Zwraca ścieżkę zapisanego pliku."""
    import soundfile as sf
    rozsz = os.path.splitext(sciezka)[1].lower()
    if not rozsz:
        sciezka, rozsz = sciezka + ".ogg", ".ogg"
    if rozsz in (".opus", ".ogg"):
        if sr not in (8000, 12000, 16000, 24000, 48000):
            raise ValueError(f"Opus nie obsługuje {sr} Hz")
        # libsndfile: compression_level 0..1 to liniowo ok. 256..6 kbit/s
        poziom = min(1.0, max(0.0, (256 - float(kbps)) / 250))
        sf.write(sciezka, audio, sr, format="OGG", subtype="OPUS", compression_level=poziom)
    else:
        sf.write(sciezka, audio, sr)
    return sciezka


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
    def _przygotuj(self, kawalki: list[str], glos: str, tempo: float, gen) -> list[dict]:
        """Dla każdego kawałka po kolei: znaki, długość z predyktora (0,35M, osobno na kawałek — w partii z dopełnieniem
        jego wynik różni się o kilka ms, co potrafi przestawić zaokrąglenie o klatkę), maska i szum startowy.
        Szum losowany w kolejności tekstu, więc wynik nie zależy od rozmiaru partii ani od kolejności liczenia."""
        styl_dp = self.glosy[glos]["styl_dp"]
        porcja = self.hop * self.ccf
        C = self.latent_dim * self.ccf
        out = []
        for t in kawalki:
            ids = self._ids(t)[0]
            sek = torch.exp(self.dp(ids[None], text_mask=torch.ones(1, 1, len(ids), device=self.device),
                                    style_dp=styl_dp, return_log=True)).reshape(()) / tempo
            T = int((sek.item() * self.sample_rate + porcja - 1) // porcja)
            n = min(T, (int((sek * self.sample_rate).long()) + porcja - 1) // porcja)
            out.append({"ids": ids, "T": T, "n": n, "szum": torch.randn(1, C, T, device=self.device, generator=gen)[0]})
        limit = self.te.context_chars
        if limit and len(kawalki) > 1:
            raw = [text_to_indices(normalize_text(t)) for t in kawalki]
            for i, e in enumerate(out):
                left = raw[i - 1][-limit:] if i else []
                right = raw[i + 1][:limit] if i + 1 < len(out) else []
                e["window"] = (e["ids"].new_tensor(left), e["ids"], e["ids"].new_tensor(right))
        return out

    @torch.no_grad()
    def _plyn(self, el: list[dict], glos: str, cfg_scale: float, kroki: int, history=None) -> list[torch.Tensor]:
        """Całkowanie przepływu dla partii kawałków (tekst i latenty dopełnione do najdłuższego, maski jak w treningu).
        Zwraca latenty dekodera [1, latent_dim, T · ccf] każdego kawałka (już odnormowane i rozpakowane)."""
        g = self.glosy[glos]
        B = len(el)
        styl = g["styl"].expand(B, -1, -1)
        Lt, T = max(len(e["ids"]) for e in el), max(e["T"] for e in el)
        ids = torch.zeros(B, Lt, dtype=torch.long, device=self.device)
        maska_t = torch.zeros(B, 1, Lt, device=self.device)
        maska = torch.zeros(B, 1, T, device=self.device)
        x = torch.zeros(B, self.latent_dim * self.ccf, T, device=self.device)
        for b, e in enumerate(el):
            ids[b, : len(e["ids"])] = e["ids"]
            maska_t[b, 0, : len(e["ids"])] = 1
            maska[b, 0, : e["n"]] = 1
            x[b, :, : e["T"]] = e["szum"]
        x = x * maska
        from .modele.text_encoder import pack_context
        context = pack_context([e["window"] for e in el]) if "window" in el[0] else None
        h = self.te(ids, styl, text_mask=maska_t, context=context, history=history)
        u = self.uncond
        u_key = getattr(u, "u_key", None)
        for i in range(kroki):
            t = torch.full((B,), i / kroki, device=self.device)
            xi = x * maska
            v = self.vf(noisy_latent=xi, text_emb=h, style_ttl=styl, latent_mask=maska, text_mask=maska_t, current_step=t)
            if cfg_scale > 1.0:
                v0 = self.vf(noisy_latent=xi, text_emb=u.u_text.expand(B, -1, h.shape[2]), style_ttl=u.u_ref.expand(B, -1, -1),
                             latent_mask=maska, text_mask=maska_t, current_step=t,
                             style_keys=u_key.expand(B, -1, -1) if u_key is not None else None)
                v = v0 + cfg_scale * (v - v0)
            x = (x + v * maska / kroki) * maska
        if self.te.history_frames:
            for b, e in enumerate(el):
                tail = x[b:b+1, :, :e["n"]][:, :, -self.te.history_frames:]
                e["next_history"] = torch.nn.functional.pad(tail, (self.te.history_frames - tail.size(2), 0))
        z = (x / self.scale if self.scale not in (0.0, 1.0) else x) * self.std + self.mean
        return [z[b : b + 1, :, : e["T"]].reshape(1, self.latent_dim, self.ccf, e["T"]).permute(0, 1, 3, 2).flatten(2, 3)
                for b, e in enumerate(el)]

    @torch.no_grad()
    def _dekoduj(self, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Dekoder dla partii, z tym samym wynikiem co osobno na każdy kawałek.

        Ciało dekodera, przednia sieć i głowica „patch” są przyczynowe (albo działają na klatkę), więc dopełnienie
        z prawej nie zmienia niczego przed końcem kawałka. Moduł poprawki widma patrzy w obie strony: STFT i odwrotne
        STFT liczone są na każdy kawałek osobno (własne odbicie na końcu sygnału), a po każdej warstwie ramki za
        końcem kawałka są zerowane — tak jak dopełnienie zerami w pojedynczym przebiegu."""
        from .modele.latent_decoder import VocoderHead
        dek = self.dekoder
        porcja = self.hop * self.ccf
        samples = slice(None) if self.te.context_chars else slice(porcja, -porcja)
        if not isinstance(dek.head, VocoderHead):   # inna głowica (np. istft) nie jest przyczynowa: po kolei
            return [dek(z).reshape(-1)[samples] for z in zs]
        B, L = len(zs), max(z.shape[-1] for z in zs)
        x = torch.zeros(B, zs[0].shape[1], L, device=self.device)
        for b, z in enumerate(zs):
            x[b, :, : z.shape[-1]] = z[0]
        x = x * dek.latent_std + dek.latent_mean
        if dek.prenet is not None:
            x = dek.prenet(x)
        x = dek.embed(x)
        for blok in dek.convnext:
            x = blok(x)
        x = dek.final_norm(x)
        wav = dek.head(x)                                        # [B, L · hop]
        dl = [z.shape[-1] * self.hop for z in zs]
        if dek.postnet is None:
            return [wav[b, : dl[b]][samples] for b in range(B)]
        pn = dek.postnet
        X, wejscia, F_ = [], [], []
        for b in range(B):
            Xb = torch.stft(wav[b : b + 1, : dl[b]].float(), pn.n_fft, pn.hop, window=pn.window, center=True,
                            return_complex=True)
            mag = Xb.abs()
            komp = mag.clamp_min(1e-8) ** 0.3 * Xb / mag.clamp_min(1e-8)
            Fb = Xb.shape[-1]
            f = x[b : b + 1, :, : zs[b].shape[-1]].float()
            if f.shape[-1] < Fb:
                f = torch.cat([f, f[..., -1:].expand(-1, -1, Fb - f.shape[-1])], dim=-1)
            X.append(Xb); F_.append(Fb)
            wejscia.append(torch.cat([komp.real, komp.imag, f[..., :Fb]], dim=1))
        Fm = max(F_)
        H = torch.zeros(B, wejscia[0].shape[1], Fm, device=self.device)
        maska = torch.zeros(B, 1, Fm, device=self.device)
        for b in range(B):
            H[b, :, : F_[b]] = wejscia[b][0]
            maska[b, 0, : F_[b]] = 1
        h = pn.proj_in(H) * maska
        for blok in pn.blocks:
            h = blok(h) * maska
        a, b_ = pn.proj_out(h).chunk(2, dim=1)
        out = []
        for b in range(B):
            m = torch.complex(1.0 + a[b : b + 1, :, : F_[b]], b_[b : b + 1, :, : F_[b]])
            modul = m.abs()
            m = m * (modul.clamp(max=pn.max_mask) / modul.clamp_min(1e-8))
            y = torch.istft(X[b] * m, pn.n_fft, pn.hop, window=pn.window, center=True, length=dl[b])
            out.append(y.reshape(-1)[samples])
        return out

    def synteza(self, tekst: str, glos: str = "poziomka", cfg: float = 4.0, kroki: int = 16, tempo: float = 1.0,
                ziarno: int | None = 0, partia: int = 64, max_znakow: int = MAKS_ZNAKOW,
                przerwy: dict | None = None) -> np.ndarray:
        """Tekst -> dźwięk 48 kHz, float32 w [-1, 1]. Długi tekst jest dzielony na zdania.

        cfg: siła prowadzenia (1 = bez CFG, dwa razy szybciej); kroki: kroki całkowania (16 domyślnie, 8 szybciej);
        tempo: > 1 szybciej; ziarno: powtarzalny wynik (None = losowo);
        przerwy: cisza po kawałku w s, np. {"przecinek": 0.08, "zdanie": 0.18, "akapit": 0.28} (model kontekstowy domyślnie zachowuje pauzy z syntezy);
        partia: ile kawałków liczyć naraz (1–8192, domyślnie 64; mniej, jeśli tekst ma mniej kawałków; 1 = po kolei);
        max_znakow: najdłuższy kawałek tekstu (zdania są sklejane do tej długości; model uczył się na wypowiedziach do ok. 30 s).
        Model z historią audio generuje po jednym kawałku w kolejności tekstu. Pozostałe modele grupują kawałki według długości."""
        if glos not in self.glosy:
            raise ValueError(f"nieznany głos {glos!r}; dostępne: {', '.join(self.glosy)}")
        if not 1 <= int(partia) <= MAKS_PARTIA:
            raise ValueError(f"partia musi być od 1 do {MAKS_PARTIA}, jest {partia}")
        partia = int(partia)
        gen = torch.Generator(device=self.device)
        if ziarno is not None:
            gen.manual_seed(ziarno)
        else:
            gen.seed()
        podzial = podziel(tekst, max_znakow=max_znakow)
        kawalki = [k for k, _ in podzial]
        if not kawalki:
            return np.zeros(0, dtype=np.float32)
        el = self._przygotuj(kawalki, glos, tempo, gen)
        kolejnosc = sorted(range(len(el)), key=lambda i: el[i]["T"])
        nagrania: list[torch.Tensor | None] = [None] * len(el)
        if self.te.history_frames:
            kolejnosc = range(len(el))
            partia = 1
        history = None
        for i in range(0, len(kolejnosc), partia):
            grupa = kolejnosc[i : i + partia]
            zs = self._plyn([el[k] for k in grupa], glos, cfg, kroki, history=history)
            if self.te.history_frames:
                history = el[grupa[0]].pop("next_history")
            for k, w in zip(grupa, self._dekoduj(zs)):
                nagrania[k] = w
        if self.te.context_chars and przerwy is None:
            # Aligned training targets own the inter-sentence pauses.
            audio = torch.cat(nagrania)
        else:
            audio = sklej(nagrania, [k for _, k in podzial], self.sample_rate, przerwy)
        return audio.clamp(-1, 1).float().cpu().numpy()

    def zapisz(self, tekst: str, sciezka: str, kbps: int = 64, **kw) -> str:
        """synteza() i zapis do pliku. Format z rozszerzenia: .ogg (Opus, domyślny, także bez rozszerzenia; `kbps`
        kbit/s, przy 64 ok. 12× mniej niż WAV), .opus (to samo), .wav albo .flac (bez strat). Zwraca ścieżkę pliku."""
        return zapisz_audio(self.synteza(tekst, **kw), sciezka, self.sample_rate, kbps)
