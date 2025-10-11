# 🧪 Balance Strength Overnight Experiments

## 📋 Co to robi?

Testuje **5 różnych strategii balansowania danych** podczas trenowania modelu AST:

- `balance_strength = 0.0` - **Brak balansowania** (naturalna dystrybucja: 874 min MUSIC vs 25 min TUNING)
- `balance_strength = 0.25` - **Lekkie balansowanie** (MUSIC nadal dominuje, ale mniej)
- `balance_strength = 0.5` - **Średnie balansowanie** (kompromis 50/50)
- `balance_strength = 0.75` - **Mocne balansowanie** (prawie równe wagi)
- `balance_strength = 1.0` - **Pełne balansowanie** (wszystkie klasy równe wagi - jak teraz)

Każdy eksperyment: **5 epok trenowania**

## ⏱️ Czas wykonania

- **1 eksperyment** = 5 epok × ~40 min = **~3.5h**
- **5 eksperymentów** = **~17h** (idealnie overnight 17:00 → 10:00 rano)

## 🚀 Jak uruchomić?

```bash
cd C:\IT\code\filharmonia-ai\backend

# Aktywuj venv
.\venv\Scripts\activate

# Uruchom eksperymenty
python experiment_balance_strength.py
```

## 📊 Co dostaniesz?

### 1. **Folder z modelami:**
```
Y:\!_FILHARMONIA\ML_EXPERIMENTS\balance_experiments\
├── ast_balance0.00_20251006_170530.pth
├── ast_balance0.25_20251006_210122.pth
├── ast_balance0.50_20251007_003415.pth
├── ast_balance0.75_20251007_040708.pth
├── ast_balance1.00_20251007_074001.pth
└── experiment_results.json  ← GŁÓWNY WYNIK
```

### 2. **experiment_results.json:**
```json
[
  {
    "balance_strength": 0.0,
    "train_acc": 94.2,
    "val_acc": 91.5,
    "test_acc": 89.3,
    "per_class_acc": {
      "MUSIC": 95.2,
      "TUNING": 62.1,
      "APPLAUSE": 92.3,
      "PUBLIC": 88.5,
      "SPEECH": 90.1
    },
    "total_time_minutes": 198.5
  },
  ...
]
```

### 3. **Console output:**
```
=== EXPERIMENT SUMMARY ===

Balance Strength: 0.00
  Test Accuracy: 89.30%
  Training Time: 198.5 min
  Per-class accuracy:
    MUSIC: 95.20%     ← Bardzo dobrze!
    TUNING: 62.10%    ← Słabo (za mało danych)
    APPLAUSE: 92.30%
    ...

Balance Strength: 0.50
  Test Accuracy: 91.80%
  Training Time: 203.2 min
  Per-class accuracy:
    MUSIC: 93.50%     ← Trochę gorzej
    TUNING: 82.40%    ← LEPIEJ!
    APPLAUSE: 90.10%
    ...

🏆 BEST MODEL
Balance Strength: 0.50
Test Accuracy: 91.80%
```

## 🔍 Na co zwrócić uwagę?

### **Trade-off MUSIC vs TUNING:**

- **balance_strength = 0.0:** Świetna MUSIC (95%), słaba TUNING (62%)
- **balance_strength = 0.5:** Dobra MUSIC (93%), lepsza TUNING (82%)
- **balance_strength = 1.0:** Średnia MUSIC (90%), najlepsza TUNING (88%)

### **Overall Accuracy:**

Może być że `0.5` da najlepszą **overall accuracy** bo balansuje dobrze wszystkie klasy.

## 📁 Struktura datasetu

Script automatycznie:
1. Czyta z `Y:\!_FILHARMONIA\TRAINING DATA\DATA\` (bez kopiowania!)
2. Robi random split 80/10/10 (train/val/test)
3. Tworzy tymczasowy dataset w `Y:\!_FILHARMONIA\ML_EXPERIMENTS\datasets\direct_from_source\`

**Ten dataset można potem usunąć** - lub zachować jeśli będziesz chciał trenować więcej modeli.

## 💾 Bieżące dane (2025-10-06 17:00):

```
APPLAUSE: 167.89 min (223 files)
MUSIC: 874.63 min (109 files)    ← 35x więcej niż TUNING
PUBLIC: 261.90 min (244 files)
SPEECH: 81.72 min (81 files)
TUNING: 25.03 min (53 files)     ← Najmniej
TOTAL: 1411.18 min = 23.5h audio
```

## ⚠️ Ważne

- **Seed = 42** (reproducible split)
- Script zapisuje **intermediate results** po każdym eksperymencie (jeśli crash - nie stracisz wszystkiego)
- **GPU required** (inaczej będzie ~10x dłużej)
- Możesz zatrzymać w każdej chwili (Ctrl+C) - zapisane wyniki są już w JSON

## 🎯 Co zrobić po eksperymentach?

1. Otwórz `experiment_results.json`
2. Znajdź najlepszy `balance_strength` (highest test_acc)
3. **Sprawdź per_class_acc** - czy TUNING jest wystarczająco dobre?
4. Jeśli potrzebujesz więcej testów (np. 0.3, 0.4, 0.6) - możesz edytować `BALANCE_STRENGTHS` w scripcie

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# W experiment_balance_strength.py zmień:
batch_size=16  →  batch_size=8
```

### Script się zawiesza
- Sprawdź czy backend nie jest uruchomiony (może blokować GPU)
- Zamknij inne aplikacje używające GPU

### Chcesz przerwać i wrócić później
- Ctrl+C
- Zapisane wyniki w `experiment_results.json`
- Możesz usunąć przetestowane wartości z `BALANCE_STRENGTHS` i uruchomić ponownie

---

**Powodzenia! Sprawdź rano wyniki 🌅**
