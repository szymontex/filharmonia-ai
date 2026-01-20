# Testing Patterns

**Analysis Date:** 2026-01-20

## Test Framework

**Runner:**
- Backend: `pytest==8.0.0` with `pytest-asyncio==0.23.0` (in `requirements.txt`)
- Frontend: No test framework configured (no Jest/Vitest in `package.json`)
- Config: No `pytest.ini`, `pyproject.toml`, or test config files detected

**Assertion Library:**
- Python: pytest (standard assertions)
- Frontend: Not configured

**Run Commands:**
```bash
# Backend (theoretical - no tests exist)
cd backend && pytest              # Run all tests
cd backend && pytest -v           # Verbose output
cd backend && pytest --cov        # With coverage (requires pytest-cov)

# Frontend
# No test commands in package.json
```

## Test File Organization

**Location:**
- **No test files exist in the project**
- Backend: No `tests/` directory, no `test_*.py` files
- Frontend: No `*.test.tsx`, `*.spec.tsx`, or `__tests__/` directories

**Expected Pattern (not implemented):**
```
backend/
  tests/
    test_analyze.py
    test_batch.py
    test_training.py
    conftest.py

frontend/
  src/
    components/
      __tests__/
        Toast.test.tsx
    pages/
      __tests__/
        CsvViewer.test.tsx
```

## Test Structure

**No existing tests to reference.**

**Recommended Pattern (Python/pytest):**
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestAnalyzeAPI:
    """Test suite for analyze endpoints"""

    def test_analyze_file_not_found(self):
        """Should return 404 for non-existent file"""
        response = client.post("/api/v1/analyze/", json={
            "mp3_path": "/nonexistent/file.mp3"
        })
        assert response.status_code == 404

    def test_analyze_invalid_extension(self):
        """Should return 400 for non-MP3 file"""
        response = client.post("/api/v1/analyze/", json={
            "mp3_path": "/some/file.wav"
        })
        assert response.status_code == 400
```

**Recommended Pattern (React/Vitest):**
```typescript
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Toast from '../Toast'

describe('Toast', () => {
  it('renders when show is true', () => {
    render(
      <Toast
        show={true}
        onClose={() => {}}
        title="Test Title"
        message="Test message"
      />
    )
    expect(screen.getByText('Test Title')).toBeInTheDocument()
  })

  it('calls onClose when clicked', () => {
    const onClose = vi.fn()
    render(
      <Toast show={true} onClose={onClose} title="Test" />
    )
    fireEvent.click(screen.getByRole('button'))
    expect(onClose).toHaveBeenCalled()
  })
})
```

## Mocking

**Framework:** Not configured

**Expected Patterns (Python):**
```python
from unittest.mock import patch, MagicMock

@patch('app.services.ast_inference.torch.load')
def test_model_loading(mock_load):
    mock_load.return_value = {'model_state_dict': MagicMock()}
    service = ASTInferenceService()
    service.load_model(Path('/fake/model.pth'))
    mock_load.assert_called_once()

# Mock file system
@patch('pathlib.Path.exists', return_value=True)
def test_file_exists(mock_exists):
    ...

# Mock external APIs
@patch('librosa.load')
def test_audio_loading(mock_librosa):
    mock_librosa.return_value = (np.zeros(48000), 48000)
    ...
```

**What to Mock:**
- File system operations (`Path.exists`, `Path.glob`)
- Audio loading (`librosa.load`, `eyed3.load`)
- ML model inference (`torch.load`, model predictions)
- External processes (`subprocess.Popen`)

**What NOT to Mock:**
- Pydantic validation
- FastAPI route handling
- Pure functions (utility calculations)

## Fixtures and Factories

**Test Data:**
No fixtures exist. Expected patterns:

```python
# conftest.py
import pytest
from pathlib import Path
import numpy as np

@pytest.fixture
def sample_audio():
    """Generate 2.97 seconds of silent audio at 48kHz"""
    return np.zeros(142560, dtype=np.float32)

@pytest.fixture
def sample_mp3_path(tmp_path):
    """Create a temporary MP3 file"""
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()
    return mp3_file

@pytest.fixture
def mock_settings(monkeypatch):
    """Override settings for testing"""
    monkeypatch.setattr('app.config.settings.SORTED_FOLDER', Path('/tmp/test_sorted'))
    monkeypatch.setattr('app.config.settings.AST_MODEL_PATH', Path('/tmp/test_model.pth'))
```

**Location:**
- Place in `backend/tests/conftest.py` (not yet created)

## Coverage

**Requirements:** Not enforced

**Recommended Setup:**
```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest --cov=app --cov-report=html

# View report
open htmlcov/index.html
```

**Target Coverage:**
- Critical paths (analysis, training): 80%+
- API endpoints: 70%+
- Utility functions: 90%+

## Test Types

**Unit Tests:**
- Not implemented
- Scope: Individual functions, service methods
- Approach: Mock dependencies, test in isolation

**Integration Tests:**
- Not implemented
- Scope: API endpoints with TestClient, database operations
- Approach: Use test database, mock external services

**E2E Tests:**
- Not implemented
- Framework: Could use Playwright or Cypress for frontend
- Scope: Full user workflows (upload, analyze, edit, export)

## Common Patterns

**Async Testing (pytest-asyncio):**
```python
import pytest

@pytest.mark.asyncio
async def test_async_endpoint():
    """Test async API endpoint"""
    from httpx import AsyncClient
    from app.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
```

**Error Testing:**
```python
import pytest
from fastapi import HTTPException

def test_file_not_found_error():
    """Should raise HTTPException for missing file"""
    with pytest.raises(HTTPException) as exc_info:
        # Call function that should raise
        ...
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.detail.lower()
```

**Parameterized Tests:**
```python
import pytest

@pytest.mark.parametrize("class_name,expected_index", [
    ("APPLAUSE", 0),
    ("MUSIC", 1),
    ("PUBLIC", 2),
    ("SPEECH", 3),
    ("TUNING", 4),
])
def test_label_mapping(class_name, expected_index):
    from app.config import settings
    assert settings.LABELS.index(class_name) == expected_index
```

## Testing Gaps

**Critical Untested Areas:**

1. **ML Inference (`backend/app/services/ast_inference.py`)**
   - Model loading/initialization
   - Audio preprocessing
   - Batch prediction accuracy
   - Risk: Model changes could break inference silently

2. **Analysis Pipeline (`backend/app/services/analyze.py`)**
   - File parsing and segmentation
   - CSV output format
   - Progress callbacks
   - Risk: Changes could corrupt analysis results

3. **Training Service (`backend/app/services/ast_training.py`)**
   - Data loading from TRAINING_DATA folder
   - Model architecture consistency
   - Checkpoint saving/loading
   - Risk: Training could produce incompatible models

4. **API Endpoints (all `backend/app/api/v1/*.py`)**
   - Request validation
   - Error responses
   - Background job management
   - Risk: API contract changes could break frontend

5. **Frontend Components (all `frontend/src/**/*.tsx`)**
   - User interactions
   - State management
   - API integration
   - Risk: UI regressions, broken user workflows

**Priority for Adding Tests:**
1. **High:** API endpoints (contract stability)
2. **High:** ML inference service (correctness critical)
3. **Medium:** Analysis pipeline (data integrity)
4. **Medium:** Training service (reproducibility)
5. **Low:** Utility functions (low complexity)

## Recommended Test Setup

**Backend (`backend/pyproject.toml` to create):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["app"]
omit = ["app/workers/*"]
```

**Frontend (`package.json` additions):**
```json
{
  "scripts": {
    "test": "vitest",
    "test:coverage": "vitest --coverage"
  },
  "devDependencies": {
    "vitest": "^1.0.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "@vitest/coverage-v8": "^1.0.0"
  }
}
```

---

*Testing analysis: 2026-01-20*
