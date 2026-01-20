# Codebase Concerns

**Analysis Date:** 2026-01-20

## Tech Debt

**Bare except clauses throughout codebase:**
- Issue: Many `except:` blocks without specific exception types catch and silently ignore all errors
- Files: `backend/app/api/v1/analyze.py:36-38`, `backend/app/api/v1/batch.py:39-41`, `backend/app/main.py:84`, `backend/app/services/analyze.py:66`, `backend/app/services/training.py:514`, `backend/app/services/ast_training.py:214`
- Impact: Debugging is difficult; errors are swallowed silently; unexpected failures go unnoticed
- Fix approach: Replace `except:` with specific exception types (e.g., `except FileNotFoundError:`, `except json.JSONDecodeError:`). Add logging for caught exceptions.

**Duplicate training services (Keras CNN + PyTorch AST):**
- Issue: Two parallel training implementations exist - legacy Keras CNN (`backend/app/services/training.py`, 535 lines) and new PyTorch AST (`backend/app/services/ast_training.py`, 833 lines)
- Files: `backend/app/services/training.py`, `backend/app/services/ast_training.py`
- Impact: Maintenance burden; confusion about which to use; duplicate code paths
- Fix approach: Remove legacy Keras `TrainingService` once AST is fully validated. Keep only `ASTTrainingService`.

**Hardcoded paths and Windows-specific path separators:**
- Issue: Code contains hardcoded Windows paths (e.g., `Y:\\!_FILHARMONIA\\SORTED\\`) in frontend
- Files: `frontend/src/pages/CsvViewer.tsx:206-207`, `frontend/src/pages/CalendarBrowser.tsx:268-269`
- Impact: Application will not work on Linux/macOS; deployment flexibility limited
- Fix approach: Use backend API to resolve paths. Never construct file paths in frontend.

**TODO comment - incomplete feature:**
- Issue: Play button in CalendarBrowser does nothing
- Files: `frontend/src/pages/CalendarBrowser.tsx:108`
- Impact: User-facing button does nothing when clicked; poor UX
- Fix approach: Implement `handlePlayRecording` to open WaveformEditor or remove the button

**Large React components with excessive state:**
- Issue: `CsvViewer.tsx` (1268 lines) and `UncertaintyReview.tsx` (970 lines) are monolithic components with 20+ useState hooks
- Files: `frontend/src/pages/CsvViewer.tsx`, `frontend/src/pages/UncertaintyReview.tsx`
- Impact: Hard to maintain; difficult to test; state management becomes complex; render performance issues
- Fix approach: Extract custom hooks (e.g., `useTrackEditor`, `useWaveformPlayer`). Split into smaller presentational components.

**Job status stored in /tmp with no cleanup:**
- Issue: Analysis job status files are written to `/tmp/filharmonia_jobs/` without cleanup mechanism
- Files: `backend/app/api/v1/analyze.py:15-16`, `backend/app/api/v1/batch.py:16-17`, `backend/app/workers/analyze_worker.py:22`
- Impact: Temp directory fills up over time; stale job files remain indefinitely
- Fix approach: Add TTL-based cleanup on startup or use database for job tracking

## Known Bugs

**Race condition in autosave:**
- Symptoms: Autosave triggers on every track change immediately, can cause excessive writes
- Files: `frontend/src/pages/CsvViewer.tsx:666-683`
- Trigger: Rapidly editing multiple track fields
- Workaround: User can wait between edits; debouncing not applied to autosave

**Toast component props inconsistency:**
- Symptoms: Toast component has two different prop interfaces used in different pages
- Files: `frontend/src/components/Toast.tsx`, `frontend/src/pages/UncertaintyReview.tsx:952-967`
- Trigger: UncertaintyReview passes `type` and `duration` props, but CsvViewer uses different prop names
- Workaround: Component adapts, but API is confusing

## Security Considerations

**No input validation on file paths:**
- Risk: Path traversal attacks possible via `path` query parameters
- Files: `backend/app/api/v1/files.py:104-108`, `backend/app/api/v1/csv_parser.py`, `backend/app/api/v1/waveform.py`
- Current mitigation: None - paths are used directly from query params
- Recommendations: Validate paths are within allowed directories (SORTED_FOLDER, TRAINING_DATA_FOLDER). Reject paths containing `..` or absolute paths outside allowed roots.

**No authentication/authorization:**
- Risk: Any user with network access can modify training data, delete CSVs, trigger analysis
- Files: All API endpoints
- Current mitigation: Assumes trusted network (local deployment)
- Recommendations: Add API key authentication for production deployment. Consider user roles if multiple operators.

**Subprocess with user-controlled path:**
- Risk: Command injection if mp3_path contains shell metacharacters
- Files: `backend/app/api/v1/analyze.py:85-91`
- Current mitigation: Path is passed as list argument (not shell string), reducing risk
- Recommendations: Ensure `shell=False` (current). Validate mp3_path is within SORTED_FOLDER.

## Performance Bottlenecks

**Waveform generation on every load:**
- Problem: Waveform data is computed on-demand for each audio file view
- Files: `backend/app/api/v1/waveform.py`, `frontend/src/pages/UncertaintyReview.tsx:178-181`
- Cause: No caching of waveform data
- Improvement path: Pre-generate waveform data during analysis; store alongside CSV; cache in memory with LRU

**Polling-based job status updates:**
- Problem: Frontend polls every 1-2 seconds for job status
- Files: `frontend/src/pages/CsvViewer.tsx:61-80`, `frontend/src/pages/CalendarBrowser.tsx:124-188`, `frontend/src/pages/CalendarBrowser.tsx:237-257`
- Cause: No WebSocket or SSE implementation
- Improvement path: Implement WebSocket for real-time job status updates

**Analysis loads entire MP3 into memory:**
- Problem: Full audio file loaded via librosa before segmentation
- Files: `backend/app/services/analyze.py:38`
- Cause: librosa.load() loads entire file
- Improvement path: Use streaming/chunked audio loading for large files; process segments without loading full file

**Uncertainty review iterates all CSVs on every request:**
- Problem: `/uncertainty/segments` endpoint scans all CSV files in ANALYSIS_RESULTS
- Files: `backend/app/api/v1/uncertainty.py:167-241`
- Cause: No index or database for segment metadata
- Improvement path: Build SQLite index of segments with confidence scores; query index instead of scanning files

## Fragile Areas

**CSV format assumptions:**
- Files: `backend/app/api/v1/csv_parser.py`, `backend/app/api/v1/uncertainty.py`
- Why fragile: Code assumes specific column names ('segment_time', 'predicted_class', 'confidence', 'model_version'). Multiple column name variations handled with fallbacks.
- Safe modification: Always check for column existence before access. Use schema validation.
- Test coverage: None - no tests exist

**Model version tracking:**
- Files: `backend/app/services/model_registry.py`, `backend/app/services/ast_training.py:502-517`
- Why fragile: `model_id` is derived from filename; `active_model` field added to metadata post-hoc; migration from old format required
- Safe modification: Always use `generate_model_id()` for consistency. Test metadata loading/saving.
- Test coverage: None

**Time format parsing:**
- Files: `frontend/src/pages/CsvViewer.tsx:339-346`, `backend/app/api/v1/uncertainty.py:19-22`
- Why fragile: Time strings assumed to be HH:MM:SS format; parseInt without error handling
- Safe modification: Add try/catch around time parsing; validate format before parsing
- Test coverage: None

## Scaling Limits

**Single-threaded analysis worker:**
- Current capacity: 1 concurrent analysis job
- Limit: Cannot parallelize analysis of multiple files
- Scaling path: Implement worker pool; use Celery or similar task queue

**In-memory job tracking:**
- Current capacity: Hundreds of jobs
- Limit: Jobs lost on server restart; no persistence
- Scaling path: Move job state to database (SQLite or PostgreSQL)

**File-based CSV storage:**
- Current capacity: Thousands of analysis results
- Limit: Directory listing becomes slow; no queryable metadata
- Scaling path: Store analysis results in database; keep CSV as export format only

## Dependencies at Risk

**PyTorch/Transformers version pinning:**
- Risk: HuggingFace model "MIT/ast-finetuned-audioset-10-10-0.4593" may become unavailable or API may change
- Impact: Training and inference would break
- Migration plan: Pin transformers version; cache model weights locally; consider self-hosting model

**eyed3 for ID3 tag reading:**
- Risk: Used only for extracting recording time from ID3 title field
- Impact: Minor - time extraction would fail gracefully
- Migration plan: Could use mutagen or tinytag as alternatives

## Missing Critical Features

**No backup/restore mechanism:**
- Problem: Training data, models, and analysis results have no backup system
- Blocks: Safe production operation; disaster recovery

**No model versioning history:**
- Problem: Replacing active model loses previous model's predictions
- Blocks: A/B testing models; rollback after bad model deployment

**No batch export of training data:**
- Problem: Can only export segments one-by-one from UI
- Blocks: Efficient dataset creation for retraining

## Test Coverage Gaps

**No automated tests exist:**
- What's not tested: Entire codebase - backend API, services, frontend components
- Files: All files in `backend/app/`, `frontend/src/`
- Risk: Regressions go unnoticed; refactoring is dangerous; no CI/CD possible
- Priority: High

**Specific high-risk untested areas:**
- CSV parsing and autosave logic (`backend/app/api/v1/csv_parser.py`)
- Model training and activation flow (`backend/app/services/ast_training.py`)
- Export to training data pipeline (`backend/app/api/v1/export.py`, `backend/app/api/v1/uncertainty.py`)
- Time/date parsing utilities (various files)
- Track boundary manipulation (`frontend/src/pages/CsvViewer.tsx:231-284`)

---

*Concerns audit: 2026-01-20*
