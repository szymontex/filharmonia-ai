# Filharmonia AI

## What This Is

Narzędzie do automatycznej analizy nagrań koncertowych. Wrzucasz MP3 z koncertów filharmonii, model AST (wytrenowany na twoich nagraniach) klasyfikuje segmenty audio: MUSIC, APPLAUSE, SPEECH, PUBLIC, TUNING. Dostajesz timestampy gdzie co jest, korygujesz błędy w UI, wysyłasz wyniki do osoby przygotowującej dokumenty ZAIKS.

## Core Value

**Zamiast ręcznie słuchać ~6-8h nagrań tygodniowo i spisywać czasy, AI robi to za ciebie.** Wszystko inne może być uproszczone — ta automatyzacja musi działać.

## Current Milestone: v0.9 — Polish & Stability

**Goal:** Make the existing tool production-ready: fix security issues, improve performance, enable cross-platform deployment, and polish the UX.

**Target features:**
- Security hardening (path traversal prevention, proper error handling)
- Performance improvements (pandas→Polars, waveform caching)
- Cross-platform paths (eliminate hardcoded Windows paths)
- Component refactoring (split 1268-line CsvViewer)
- UX polish (keyboard shortcuts, undo, better feedback)

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ Upload MP3 i uruchom analizę AST — existing
- ✓ Klasyfikacja segmentów (MUSIC, APPLAUSE, SPEECH, PUBLIC, TUNING) — existing
- ✓ Przeglądanie wyników w UI z waveformem — existing
- ✓ Edycja/korekta granic segmentów — existing
- ✓ Autosave edytowanych CSVów — existing
- ✓ Eksport segmentów do training data — existing
- ✓ Trenowanie modelu AST na własnych danych — existing
- ✓ Przełączanie między modelami — existing
- ✓ Batch analysis wielu plików — existing
- ✓ Uncertainty review (przegląd niepewnych predykcji) — existing
- ✓ Kalendarzowy browser nagrań — existing

### Active

<!-- Current scope. Building toward these. -->

**Stabilność/Portability:**
- [ ] Jedna wersja kodu działająca na NVIDIA (CUDA), AMD (ROCm), i CPU
- [ ] Eliminacja hardcoded ścieżek Windows
- [ ] Stabilne działanie bez łatania kodu na bieżąco
- [ ] Działa lokalnie i przez sieć (remote access)

**Performance:**
- [ ] Szybsza analiza na CPU (obecnie zbyt wolna)
- [ ] Cachowanie waveform data (obecnie generowane na każdy request)
- [ ] Optymalizacja ładowania audio (streaming zamiast full load)

**UI/UX Polish:**
- [ ] Uproszczony workflow — mniej kroków do wyniku
- [ ] Czystszy, profesjonalny wygląd
- [ ] Lepsze error handling i feedback do usera
- [ ] Refactor monolitycznych komponentów (CsvViewer 1268 linii)

**Robustness:**
- [ ] Obsługa błędów zamiast bare except
- [ ] Walidacja ścieżek (path traversal prevention)
- [ ] Cleanup job files w /tmp
- [ ] Podstawowe testy dla critical paths

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Automatyczny eksport do szablonu ZAIKS — następny milestone
- Sprawdzanie czy utwór jest chroniony (AI + web search) — następny milestone
- Autentykacja/autoryzacja — local tool, trusted network
- Real-time chat/collaboration — single user tool
- Mobile app — desktop-first
- Usunięcie legacy Keras training service — działa, nie przeszkadza

## Context

**Środowisko użycia:**
- ~12 nagrań/tydzień × 30-40 min = ok. 6-8h audio tygodniowo
- Główny user pracuje w studio z różnym dostępem do sprzętu
- Trzy maszyny: serwer (CPU), Windows PC z RTX, PC z Radeonem
- Potrzeba działania lokalnego i zdalnego

**Stan techniczny:**
- React SPA + FastAPI + PyTorch AST model
- Działa ale jest kruchy — różne wersje na różnych kompach
- Brak testów, hardcoded ścieżki, bare except blocks
- Waveform generowany on-demand (wolne)
- Polling zamiast WebSocket dla job status

**Istniejąca dokumentacja codebase:**
- .planning/codebase/ARCHITECTURE.md
- .planning/codebase/STACK.md
- .planning/codebase/STRUCTURE.md
- .planning/codebase/CONCERNS.md

## Constraints

- **Hardware:** Musi działać na CUDA (RTX), ROCm (Radeon), i CPU — różne maszyny w użyciu
- **Audio params:** Sample rate 48kHz, frame duration 2.97s — NIEZMIENNE (kompatybilność z wytrenowanymi modelami)
- **Labels:** APPLAUSE, MUSIC, PUBLIC, SPEECH, TUNING — alfabetycznie, NIEZMIENNE (model output order)
- **Backward compat:** Istniejące CSVy i modele muszą nadal działać

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch AST over Keras CNN | Lepsze wyniki, nowszy stack | ✓ Good |
| File-based job status (/tmp JSON) | Prostsze niż baza danych | ⚠️ Revisit (cleanup, scaling) |
| No auth | Local trusted tool | ✓ Good for now |
| Brownfield improvement | Istniejący kod działa, refactor > rewrite | — Pending |

---
*Last updated: 2026-01-21 after v0.9 milestone start*
