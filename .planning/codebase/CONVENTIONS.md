# Coding Conventions

**Analysis Date:** 2026-01-20

## Naming Patterns

**Files:**
- Python: `snake_case.py` (e.g., `ast_inference.py`, `model_registry.py`)
- TypeScript/React: `PascalCase.tsx` for components/pages (e.g., `CsvViewer.tsx`, `TrainingManager.tsx`)
- TypeScript constants: `camelCase.ts` (e.g., `colors.ts`)
- Config files: lowercase with dots (e.g., `vite.config.ts`, `tailwind.config.js`)

**Functions:**
- Python: `snake_case` (e.g., `get_analyze_service()`, `load_model()`, `predict_batch()`)
- TypeScript: `camelCase` (e.g., `loadCsvList()`, `handleStartTraining()`, `formatDate()`)
- React event handlers: `handle` prefix (e.g., `handlePlayPause`, `handleActivateModel`)

**Variables:**
- Python: `snake_case` (e.g., `model_path`, `audio_segment`, `num_segments`)
- TypeScript: `camelCase` (e.g., `csvFiles`, `trainingStatus`, `selectedCsv`)
- Constants: `SCREAMING_SNAKE_CASE` in Python (e.g., `LABELS`, `SAMPLE_RATE`, `BATCH_SIZE`)

**Types/Interfaces:**
- TypeScript interfaces: `PascalCase` (e.g., `Track`, `Recording`, `TrainingStatus`, `ModelInfo`)
- Python dataclasses: `PascalCase` (e.g., `TrainingStatus`, `ModelInfo`)
- Pydantic models: `PascalCase` (e.g., `AnalyzeRequest`, `BatchResponse`, `FileInfo`)

**React Components:**
- `PascalCase` function components (e.g., `CsvViewer`, `Toast`, `StickyPlayer`)
- Props interfaces: `{ComponentName}Props` (e.g., `ToastProps`, `WaveformEditorProps`)

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc` or ESLint in project root)
- Python: 4-space indentation, 80-100 char lines
- TypeScript: 2-space indentation (Vite default)
- Tailwind CSS for styling (no custom CSS files)

**Linting:**
- TypeScript: `strict: true` in `tsconfig.json`
- `noUnusedLocals: true`, `noUnusedParameters: true`
- No ESLint configured for this project

**TypeScript Configuration:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

## Import Organization

**Python Order:**
1. Standard library (`os`, `time`, `json`, `uuid`, `threading`)
2. Third-party packages (`fastapi`, `torch`, `numpy`, `librosa`)
3. Local application imports (`from app.config import settings`, `from app.services...`)

**TypeScript Order:**
1. React imports (`import { useState, useEffect } from 'react'`)
2. Third-party libraries (`import axios from 'axios'`)
3. Local components/pages (`import Toast from '../components/Toast'`)
4. Constants/utilities (`import { CLASS_COLORS } from '../constants/colors'`)

**Path Aliases:**
- TypeScript: `@/*` maps to `./src/*` (configured in `tsconfig.json` and `vite.config.ts`)
- Usage: `import { something } from '@/components/Something'`

## Error Handling

**Python Patterns:**
```python
# Try-except with specific exceptions
try:
    audiofile = eyed3.load(str(mp3_file))
    if audiofile and audiofile.tag and audiofile.tag.title:
        # Process
except:
    pass  # Silent failure for optional operations

# HTTPException for API errors
if not mp3_path.exists():
    raise HTTPException(status_code=404, detail=f"File not found: {mp3_path}")

# InterruptedError for cancellation
if check_cancelled and check_cancelled():
    raise InterruptedError("Analysis cancelled by user")
```

**TypeScript Patterns:**
```typescript
// Axios error handling with toast notifications
try {
  const response = await axios.post('/api/v1/training/start')
  setSuccessToast({ show: true, message: 'Training started!' })
} catch (error: any) {
  console.error('Error starting training:', error)
  setErrorToast({
    show: true,
    message: error.response?.data?.detail || 'Failed to start training'
  })
}
```

**Error Display:**
- Use Toast component for user-facing errors
- Console.error for debugging
- Optional chaining for safe property access: `error.response?.data?.detail`

## Logging

**Framework:** `print()` statements (no logging framework)

**Patterns:**
```python
# Status messages with emoji prefixes
print(f"[OK] AST model loaded: {model_path.name}")
print(f"[Training {job_id}] Loading training data...")
print(f"  - {class_name}: {count}")  # Indented sub-items

# Warning messages
print("⚠ AST model not found - will need to train first")

# Shutdown messages
print("🔄 Graceful shutdown: marking active analysis jobs as interrupted...")
print("✓ Shutdown complete")
```

**Frontend:**
```typescript
console.log('Autosaved at', new Date().toLocaleTimeString())
console.error('Error loading waveform:', error)
```

## Comments

**When to Comment:**
- Module-level docstrings explaining purpose
- Complex algorithm explanations
- Critical warnings about synchronization (e.g., training/inference consistency)
- API endpoint documentation

**Docstring Pattern (Python):**
```python
"""
Audio Analysis Service
Uses PyTorch AST (Audio Spectrogram Transformer) for inference
"""

def analyze_file(self, mp3_path: Path, output_csv: Path = None,
                 on_progress=None, check_cancelled=None) -> dict:
    """
    Analyze one MP3 file using PyTorch AST

    Args:
        mp3_path: Path to MP3 file
        output_csv: Optional output CSV path
        on_progress: Callback function(current, total, percent)
        check_cancelled: Callback to check if cancelled

    Returns:
        dict with csv_path, segments_analyzed, duration_seconds
    """
```

**Critical Comments:**
```python
# IMPORTANT: Limit CPU threads to prevent 100% CPU usage blocking the system
torch.set_num_threads(2)

# DO NOT CHANGE - training consistency!
FRAME_DURATION_SEC: float = 2.97

# IDENTICAL to training! (repeated for emphasis)
logmel = torch.log(melspec + 1e-9)
```

**JSDoc (TypeScript):**
```typescript
/**
 * Shared color constants for audio classification categories
 */
export const CLASS_COLORS = { ... }
```

## Function Design

**Size:**
- Most functions 10-50 lines
- Large components like `CsvViewer.tsx` (1200+ lines) contain many small handlers
- Service classes group related functions

**Parameters:**
- Python: Use keyword arguments for optional params
- TypeScript: Use object destructuring for props: `{ onBack, initialCsv }: CsvViewerProps`
- Callbacks for progress/cancellation: `on_progress=None, check_cancelled=None`

**Return Values:**
- Python API endpoints: Return dicts or Pydantic models
- Python services: Return dicts or domain objects
- TypeScript: Use typed state setters, no explicit returns from handlers

## Module Design

**Exports (Python):**
- Singleton pattern for services:
```python
_service = None

def get_analyze_service():
    global _service
    if _service is None:
        _service = AnalyzeService()
    return _service
```

**Exports (TypeScript):**
- Default exports for components: `export default function CsvViewer`
- Named exports for constants: `export const CLASS_COLORS`
- Type exports: `export type ClassType = keyof typeof CLASS_COLORS`

**Barrel Files:**
- Python: `__init__.py` files are empty or minimal
- TypeScript: No barrel files; direct imports

## React Patterns

**State Management:**
- Local state with `useState` for component-specific data
- Multiple related state variables (not grouped objects)
```typescript
const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
const [selectedCsv, setSelectedCsv] = useState<string | null>(null)
const [loading, setLoading] = useState(false)
```

**Side Effects:**
- `useEffect` for data fetching on mount
- `useEffect` with dependencies for derived state
- Cleanup functions for intervals/event listeners

**Component Structure:**
1. State declarations
2. useEffect hooks
3. Handler functions
4. Helper functions
5. JSX return

**Conditional Rendering:**
```typescript
if (page === 'csv') {
  return <CsvViewer onBack={() => setPage('home')} />
}
// ...
return <HomePage onNavigate={setPage} />
```

## API Patterns

**FastAPI Router Structure:**
```python
router = APIRouter(prefix="/analyze", tags=["analyze"])

@router.post("/", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    ...

@router.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    ...
```

**Request/Response Models:**
```python
class AnalyzeRequest(BaseModel):
    mp3_path: str

class AnalyzeResponse(BaseModel):
    job_id: str
    message: str
```

**Async Background Tasks:**
- Use `subprocess.Popen` for CPU-heavy analysis
- Use `multiprocessing.Process` for batch jobs
- Write status to JSON files for IPC

---

*Convention analysis: 2026-01-20*
