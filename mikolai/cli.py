"""mikolai "tekst" -o plik  (plik.ogg: Opus 64 kbit/s; albo -o plik.wav | .flac; tekst z pliku: --plik tekst.txt)"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(prog="mikolai", description="MikołAI: polska synteza mowy, 48 kHz")
    p.add_argument("tekst", nargs="?", help="tekst do przeczytania (bez niego: --plik albo standardowe wejście)")
    p.add_argument("--plik", help="plik z tekstem (UTF-8)")
    p.add_argument("-o", "--wyjscie", default="mikolai.ogg",
                   help="plik wynikowy: .ogg (Opus 64 kbit/s, domyślnie; także bez rozszerzenia), .wav albo .flac")
    p.add_argument("--repo", default=None, help="repo HF albo katalog z wagami")
    p.add_argument("--cfg", type=float, default=4.0, help="siła prowadzenia; 1 = bez CFG (2× szybciej)")
    p.add_argument("--kroki", type=int, default=16, help="kroki całkowania (8 = szybciej)")
    p.add_argument("--tempo", type=float, default=1.0, help="> 1 szybciej, < 1 wolniej")
    p.add_argument("--ziarno", type=int, default=0, help="ziarno losowania (-1 = losowo)")
    p.add_argument("--kbps", type=int, default=64, help="przepływność Opus dla .ogg (kbit/s, domyślnie 64)")
    p.add_argument("--partia", type=int, default=64, help="ile kawałków tekstu liczyć naraz (1–8192)")
    p.add_argument("--max_znakow", type=int, default=350, help="najdłuższy kawałek tekstu (domyślnie 350)")
    p.add_argument("--device", default=None, help="cpu / cuda (domyślnie cuda, jeśli jest)")
    p.add_argument("--watki", type=int, default=None, help="liczba wątków CPU")
    a = p.parse_args()

    tekst = a.tekst or (open(a.plik, encoding="utf-8").read() if a.plik else sys.stdin.read())
    if not tekst.strip():
        p.error("pusty tekst")
    import torch
    if a.watki:
        torch.set_num_threads(a.watki)
    from .tts import DOMYSLNE_REPO, MikolAI
    tts = MikolAI.from_pretrained(a.repo or DOMYSLNE_REPO, device=a.device)
    t0 = time.time()
    plik = tts.zapisz(tekst, a.wyjscie, cfg=a.cfg, kroki=a.kroki, tempo=a.tempo,
               ziarno=None if a.ziarno < 0 else a.ziarno, partia=a.partia, max_znakow=a.max_znakow, kbps=a.kbps)
    import soundfile as sf
    dl = sf.info(plik).duration
    print(f"{plik}: {dl:.1f} s audio w {time.time() - t0:.1f} s (RTF {(time.time() - t0) / max(dl, 1e-6):.2f})")


if __name__ == "__main__":
    main()
