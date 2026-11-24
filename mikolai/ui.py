"""Interfejs w przeglądarce (Gradio): mikolai-ui [--repo ...] [--port 7860] [--share]"""

from __future__ import annotations

import argparse
import time

import numpy as np

PRZYKLADY = [
    "Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?",
    "Dzień dobry, to jest próbka nowego modelu. Jak brzmi?",
    "Chrząszcz brzmi w trzcinie w Szczebrzeszynie, a źdźbło trawy drży na wietrze.",
    "Wypad na plażę. Na plaży są same parawany, bez ludzi. Morze jest bardzo niespokojne i fale są ogromne.",
]


def zbuduj_ui(tts):
    import gradio as gr

    def mow(tekst, cfg, kroki, tempo, ziarno, losowo, partia):
        tekst = (tekst or "").strip()
        if not tekst:
            raise gr.Error("Wpisz tekst.")
        t0 = time.time()
        audio = tts.synteza(tekst, cfg=float(cfg), kroki=int(kroki), tempo=float(tempo),
                            ziarno=None if losowo else int(ziarno), partia=int(partia))
        dl = len(audio) / tts.sample_rate
        dt = time.time() - t0
        # 1 s ciszy na końcu: odtwarzacz w przeglądarce potrafi uciąć ostatnie dziesiątki ms nagrania
        audio = np.concatenate([audio, np.zeros(tts.sample_rate, dtype=audio.dtype)])
        return (tts.sample_rate, audio), f"{dl:.1f} s mowy w {dt:.2f} s (RTF {dt / max(dl, 1e-6):.2f}, {tts.device})"

    with gr.Blocks(title="MikołAI") as demo:
        gr.Markdown("# MikołAI — polska synteza mowy\nMały model TTS (109M parametrów, 48 kHz, jeden głos).")
        with gr.Row():
            with gr.Column(scale=3):
                tekst = gr.Textbox(label="Tekst", lines=5, value=PRZYKLADY[0])
                przycisk = gr.Button("Czytaj", variant="primary")
                gr.Examples(PRZYKLADY, inputs=tekst)
            with gr.Column(scale=2):
                cfg = gr.Slider(1.0, 6.0, value=4.0, step=0.5, label="CFG (1 = bez prowadzenia, 2× szybciej)")
                kroki = gr.Slider(2, 32, value=16, step=1, label="Kroki")
                tempo = gr.Slider(0.6, 1.6, value=1.0, step=0.05, label="Tempo")
                partia = gr.Number(value=64, precision=0, minimum=1, maximum=8192, label="Partia (kawałków naraz, 1–8192)")
                with gr.Row():
                    ziarno = gr.Number(value=0, precision=0, label="Ziarno")
                    losowo = gr.Checkbox(value=False, label="Losowo")
        wyjscie = gr.Audio(label="Nagranie (48 kHz)", type="numpy")
        info = gr.Markdown()
        przycisk.click(mow, [tekst, cfg, kroki, tempo, ziarno, losowo, partia], [wyjscie, info])
        tekst.submit(mow, [tekst, cfg, kroki, tempo, ziarno, losowo, partia], [wyjscie, info])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(prog="mikolai-ui", description="MikołAI w przeglądarce (Gradio)")
    p.add_argument("--repo", default=None, help="repo HF albo katalog z wagami")
    p.add_argument("--device", default=None, help="cpu / cuda (domyślnie cuda, jeśli jest)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = dostęp z sieci lokalnej")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="publiczny link Gradio")
    a = p.parse_args()
    try:
        import gradio  # noqa: F401
    except ImportError as e:
        raise SystemExit("interfejs wymaga Gradio: pip install 'mikolai[ui]'") from e
    from .tts import DOMYSLNE_REPO, MikolAI
    tts = MikolAI.from_pretrained(a.repo or DOMYSLNE_REPO, device=a.device)
    tts.synteza("Rozgrzewka.", kroki=2)   # pierwsze wywołanie (alokacje, jądra) przed pierwszym użytkownikiem
    zbuduj_ui(tts).queue().launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
