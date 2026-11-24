# mikolai

Biblioteka do syntezy mowy modelem **MikołAI** — mały (ok. 113M parametrów) polski TTS, 48 kHz, jeden
syntetyczny głos („poziomka”). Wagi: [cpral/MikolAI-113M-1-Polski-TTS](https://huggingface.co/cpral/MikolAI-113M-1-Polski-TTS).
Kod treningowy: [adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe).

## Instalacja

```bash
pip install "git+https://github.com/adamo1139/mikolai"            # sam TTS
pip install "mikolai[liczby] @ git+https://github.com/adamo1139/mikolai"   # + liczby na słowa (Morfeusz 2)
```

PyTorch najlepiej zainstalować wcześniej w wersji pasującej do sprzętu (CPU albo CUDA).

## Użycie

```python
from mikolai import MikolAI

tts = MikolAI.from_pretrained()                 # pobiera wagi z HF (ok. 440 MB), GPU jeśli jest
audio = tts.synteza("Dzień dobry, tu MikołAI.")  # numpy float32, 48 kHz
tts.zapisz("Spotkanie jest 15 września o 9:30.", "spotkanie.wav", liczby=True)
```

Z wiersza poleceń:

```bash
mikolai "Dzień dobry, tu MikołAI." -o dzien_dobry.wav
mikolai --plik artykul.txt -o artykul.wav --liczby --device cpu --watki 8
```

### Ustawienia

| argument | domyślnie | znaczenie |
|---|---|---|
| `cfg` | 4.0 | siła prowadzenia (classifier-free guidance); `1` wyłącza CFG i daje ok. 2× szybciej |
| `kroki` | 16 | kroki całkowania przepływu; `8` szybciej, odrobinę gorzej |
| `tempo` | 1.0 | > 1 szybciej, < 1 wolniej |
| `ziarno` | 0 | ten sam tekst i ziarno = ten sam dźwięk; `None` = za każdym razem inaczej |
| `liczby` | False | liczby, daty, godziny, kwoty i jednostki na słowa z odmianą (wymaga `mikolai[liczby]`) |

Model najlepiej czyta liczby zapisane słownie — z cyframi radzi sobie, ale gorzej (dane z cyframi pochodziły
od nauczyciela, który sam je często przekręcał). Długi tekst jest dzielony na zdania i czytany po kawałku
(do ok. 250 znaków), z krótką przerwą między nimi.

### Szybkość (orientacyjnie)

| sprzęt | ustawienia | RTF |
|---|---|---|
| i7-11700K, 8 wątków | CFG 4, 16 kroków | ok. 0,35 |
| i7-11700K, 1 wątek | CFG 4, 16 kroków | ok. 0,8 |
| i7-11700K, 1 wątek | CFG 1, 8 kroków | ok. 0,25 |

RTF = czas liczenia / długość nagrania (poniżej 1 = szybciej niż czas rzeczywisty).

## Licencje

Kod biblioteki: MIT (`LICENSE`); fragmenty skopiowane z BlueTTS i blue-codec — MIT (`LICENSE-BlueTTS`,
`LICENSE-blue-codec`, opis w `ZRODLO.md`). Wagi modelu w repo HF: OpenMDW 1.1.
