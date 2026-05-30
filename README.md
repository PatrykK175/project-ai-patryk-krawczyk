# Projekt: Klasyfikacja obrazów MNIST z użyciem CNN (PyTorch)

Autor: **Patryk Krawczyk**

Projekt spełnia wymagania z `instrukcja.txt`:
- obszar: **klasyfikacja obrazów**,
- implementacja: **PyTorch**,
- uczenie: **wsteczna propagacja + optymalizatory (Adam/SGD)**,
- minimum 3 eksperymenty porównawcze,
- ewaluacja: accuracy, loss, błędne predykcje, mapy cech,
- dokumentacja: notebook `ipynb`.

## 1) Wymagania

- Python 3.10+
- `pip`

## 2) Instalacja krok po kroku (lokalnie)

1. Wejdź do katalogu projektu:
```bash
cd /home/patryk/Projects/Studia/mgr/ai/project
```

2. Utwórz i aktywuj środowisko wirtualne:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Zainstaluj zależności:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. (Opcjonalnie) Dodaj kernel do Jupyter:
```bash
python -m ipykernel install --user --name dl-projekt --display-name "Python (dl-projekt)"
```

## 3) Uruchamianie projektu

### Szybki test (mały podzbiór danych)
```bash
python src/train_experiments.py --epochs 1 --quick
```

### Pełne uruchomienie (3 eksperymenty)
```bash
python src/train_experiments.py --epochs 5
```

Dane MNIST pobiorą się automatycznie do katalogu `data/`.

## 4) Co robi skrypt

Skrypt `src/train_experiments.py`:
- trenuje 3 konfiguracje CNN,
- zapisuje metryki dla epok (`history.csv`),
- zapisuje model z najlepszą walidacją (`best_model.pt`),
- generuje:
  - `learning_curves.png` (loss/accuracy),
  - `misclassified.png` (błędne predykcje),
  - `feature_maps.png` (mapy cech),
- tworzy globalne podsumowanie `results/all_experiments_summary.csv`.

## 5) Struktura wyników

Dla każdego eksperymentu:
- `results/<nazwa_eksperymentu>/history.csv`
- `results/<nazwa_eksperymentu>/summary.json`
- `results/<nazwa_eksperymentu>/best_model.pt`
- `results/<nazwa_eksperymentu>/learning_curves.png`
- `results/<nazwa_eksperymentu>/misclassified.png`
- `results/<nazwa_eksperymentu>/feature_maps.png`

Globalnie:
- `results/all_experiments_summary.csv`

## 6) Mini-tabela wyników (test)

| Eksperyment | test_acc | test_loss |
|---|---:|---:|
| exp1_adam_lr1e3_depth2_relu_bs128 | 0.990605 | 0.030181 |
| exp2_adam_lr5e4_depth3_relu_bs128 | 0.990803 | 0.025917 |
| exp3_sgd_lr1e2_depth2_lrelu_bs64 | 0.990844 | 0.027990 |

## 7) Dokumentacja projektowa

Notebook dokumentacyjny znajduje się w:
- `dokumentacja_projektowa.ipynb`

Uruchomienie:
```bash
jupyter notebook
```
