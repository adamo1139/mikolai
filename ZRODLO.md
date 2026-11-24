# Skąd jest ten kod

Kod sieci w `mikolai/modele/` to kopie plików z repozytorium treningowego
[adamo1139/syntezujemy_polska_mowe](https://github.com/adamo1139/syntezujemy_polska_mowe) (katalog `trening/`),
które z kolei pochodzą z dwóch projektów na licencji MIT (pliki `LICENSE-BlueTTS` i `LICENSE-blue-codec`):

| plik | skąd |
|---|---|
| `modele/text_encoder.py`, `modele/vf_estimator.py` | `trening/training/t2l/models/` ← [BlueTTS](https://github.com/maxmelichov/BlueTTS) `training/` |
| `modele/duration_predictor.py`, `modele/dp_network.py` | `trening/training/dp/models/` ← BlueTTS |
| `modele/uncond.py` | klasa `UncondParams` z `trening/training/t2l/cfg_utils.py` ← BlueTTS |
| `modele/latent_decoder.py`, `modele/modules.py` | `trening/bluecodec/autoencoder/` ← [blue-codec](https://github.com/maxmelichov/blue-codec) (z naszymi zmianami: moduł poprawki STFT, przednia sieć, aktywacja okresowa) |
| `slownik.py` | `trening/training/data/text_vocab.py` (słownik znaków, nasz) |

Zmiany przy kopiowaniu: tylko importy (moduły względne), usunięte komunikaty przy imporcie i ładowanie
dekodera Supertonic-3 (nieużywane). `budowa.py` i `tts.py` są napisane od nowa, ale liczą to samo co
`trening/training/t2l/builders.py` i `sample_audio()` (sprawdzone: ten sam wynik co `trening/syntezuj.py`
przy tym samym ziarnie).
