# mikolai

Biblioteka do syntezy mowy modelem **MikołAI** — mały (ok. 109M parametrów) polski TTS, 48 kHz, jeden
syntetyczny głos („poziomka”). Wagi: [cpral/MikolAI-109M-v0.1-Polish-TTS-Preview](https://huggingface.co/cpral/MikolAI-109M-v0.1-Polish-TTS-Preview).
Kod treningowy: [adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe).

## Instalacja

```bash
pip install "git+https://github.com/adamo1139/mikolai"
```

PyTorch najlepiej zainstalować wcześniej w wersji pasującej do sprzętu (CPU albo CUDA).

## Użycie

```python
from mikolai import MikolAI

tts = MikolAI.from_pretrained()                 # pobiera wagi z HF (ok. 440 MB), GPU jeśli jest
audio = tts.synteza("Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?")  # numpy float32, 48 kHz
tts.zapisz("Miło mi cię poznać.", "powitanie.ogg")
tts.zapisz("Dłuższy tekst do odsłuchu.", "tekst.ogg", kbps=48)   # Opus: ok. 12× mniejszy niż WAV
```

Z wiersza poleceń:

```bash
mikolai "Dzień Dobry, umawiała się Pani na strzyżenie włosów u nas na jutro, czy Pani wizyta jest aktualna?" -o wizyta.ogg
mikolai --plik artykul.txt -o artykul.ogg --device cpu --watki 8
mikolai --plik ksiazka.txt -o ksiazka.ogg --kbps 48  # Opus (domyślnie 64 kbit/s); bez strat: -o ksiazka.flac
```

### Interfejs w przeglądarce

```bash
pip install "mikolai[ui] @ git+https://github.com/adamo1139/mikolai"
mikolai-ui                      # http://127.0.0.1:7860
mikolai-ui --host 0.0.0.0       # dostęp z innych komputerów w sieci lokalnej
```

### Ustawienia

| argument | domyślnie | znaczenie |
|---|---|---|
| `cfg` | 4.0 | siła prowadzenia (classifier-free guidance); `1` wyłącza CFG i daje ok. 2× szybciej |
| `kroki` | 16 | kroki całkowania przepływu; `8` szybciej, odrobinę gorzej |
| `tempo` | 1.0 | > 1 szybciej, < 1 wolniej |
| `max_znakow` | 350 | najdłuższy kawałek długiego tekstu (ok. 18 s mowy); dłuższe model czyta szybciej |
| `przerwy` | `PRZERWY` | cisza po kawałku w s: `{"przecinek": 0.08, "zdanie": 0.18, "akapit": 0.28}` |
| `partia` | 64 | ile kawałków długiego tekstu liczyć naraz (1–8192); więcej = szybciej na GPU, więcej pamięci |
| `ziarno` | 0 | ten sam tekst i ziarno = ten sam dźwięk; `None` = za każdym razem inaczej |

Długi tekst (np. rozdział książki) jest dzielony na kawałki do ok. 350 znaków: zdania są sklejane także przez
granice akapitów (nowa linia), a zbyt długie zdanie jest cięte po przecinku. Każdy kawałek jest czytany osobno,
cisza modelu na jego brzegach jest przycinana, a między kawałkami wstawiana jest przerwa zależna od miejsca
(przecięte zdanie, koniec zdania, koniec akapitu), dobrana do naturalnych pauz modelu między zdaniami.

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
