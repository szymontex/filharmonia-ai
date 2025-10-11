# API Documentation

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## File Management

### List Files

```http
GET /api/v1/files/list?path={folder_path}
```

Lists files in specified directory.

**Response:**
```json
{
  "files": [
    {"name": "concert.mp3", "path": "/full/path", "type": "file", "size": 12345678}
  ]
}
```

### Get Sorted Recordings

```http
GET /api/v1/files/sorted
```

Returns tree structure of sorted recordings by date.

## Sorting

### Sort Recording

```http
POST /api/v1/sort/file
Content-Type: application/json

{
  "source_path": "/path/to/unsorted/file.mp3",
  "target_date": "2025-10-11"  // optional, extracted from ID3 if not provided
}
```

Organizes recording into `SORTED/YYYY/MM/DD/` structure.

## Analysis

### Analyze Single File

```http
POST /api/v1/analyze/single
Content-Type: application/json

{
  "audio_path": "/path/to/concert.mp3"
}
```

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "processing"
}
```

### Batch Analysis

```http
POST /api/v1/batch/analyze
Content-Type: application/json

{
  "file_paths": ["/path/to/file1.mp3", "/path/to/file2.mp3"]
}
```

### Get Analysis Status

```http
GET /api/v1/analyze/status/{job_id}
```

**Response:**
```json
{
  "status": "completed",
  "progress": 100,
  "result_csv": "/path/to/predictions.csv"
}
```

## CSV Parsing

### Parse CSV

```http
GET /api/v1/csv/parse?file_path={csv_path}
```

**Response:**
```json
{
  "segments": [
    {
      "timestamp": "00:00:00",
      "class": "MUSIC",
      "confidence": 0.9823,
      "duration": 2.97
    }
  ]
}
```

### Save CSV

```http
POST /api/v1/csv/save
Content-Type: application/json

{
  "file_path": "/path/to/predictions.csv",
  "segments": [...]
}
```

### Check Autosave

```http
GET /api/v1/csv/check-autosave?file_path={csv_path}
```

Compares autosave with original, auto-deletes if identical.

## Waveform

### Get Waveform Data

```http
GET /api/v1/waveform?audio_path={path}&samples={points}
```

Returns downsampled waveform for visualization.

**Response:**
```json
{
  "waveform": [0.1, 0.5, -0.3, ...],  // amplitude values
  "duration": 3600.5,                  // seconds
  "sample_rate": 48000
}
```

## Export

### Export Segments

```http
POST /api/v1/export/segments
Content-Type: application/json

{
  "audio_path": "/path/to/concert.mp3",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.97,
      "class": "MUSIC"
    }
  ]
}
```

Exports segments to `TRAINING_DATA/DATA/{CLASS}/` as stereo WAV (44.1kHz).

Prevents duplicates via `exported_segments.csv`.

## Training

### Start Training

```http
POST /api/v1/training/start
Content-Type: application/json

{
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.00001,
  "balance_strength": 0.75,
  "notes": "Retrain after manual corrections"
}
```

**Response:**
```json
{
  "job_id": "training-xyz-789",
  "status": "started"
}
```

### Get Training Status

```http
GET /api/v1/training/status/{job_id}
```

**Response:**
```json
{
  "status": "training",
  "progress": 45,
  "current_epoch": 5,
  "total_epochs": 10,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.956
  }
}
```

### List Models

```http
GET /api/v1/training/models
```

**Response:**
```json
{
  "models": [
    {
      "filename": "ast_20251009_222204.pth",
      "size_mb": 345.2,
      "created_at": "2025-10-09T22:22:04",
      "is_active": true,
      "accuracy": {
        "test": 97.75,
        "train": 99.12,
        "val": 98.45
      },
      "epochs": 10,
      "notes": "Best model so far"
    }
  ]
}
```

### Activate Model

```http
POST /api/v1/training/activate-model
Content-Type: application/json

{
  "filename": "ast_20251009_222204.pth"
}
```

Copies model to `ast_active.pth` and updates metadata.

### Measure Model Accuracy

```http
POST /api/v1/training/measure-accuracy/{filename}
```

Evaluates model on test set, updates metadata.

**Response:**
```json
{
  "test_accuracy": 97.75,
  "train_accuracy": 99.12,
  "val_accuracy": 98.45,
  "per_class": {
    "APPLAUSE": 100.0,
    "MUSIC": 100.0,
    "PUBLIC": 96.2,
    "SPEECH": 100.0,
    "TUNING": 85.7
  }
}
```

### Get Training Data Stats

```http
GET /api/v1/training/data-stats
```

**Response:**
```json
{
  "stats_per_class": {
    "APPLAUSE": {"count": 1234, "duration_min": 61.2},
    "MUSIC": {"count": 5678, "duration_min": 874.5},
    "PUBLIC": {"count": 890, "duration_min": 132.1},
    "SPEECH": {"count": 456, "duration_min": 78.9},
    "TUNING": {"count": 123, "duration_min": 25.3}
  },
  "total_duration_min": 1172.0,
  "total_count": 8381
}
```

### Delete Model

```http
DELETE /api/v1/training/models/{filename}
```

Deletes model file and removes from metadata.

## Uncertainty Review

### Get Uncertain Segments

```http
GET /api/v1/uncertainty/segments?threshold=0.7&limit=100
```

Returns segments with confidence < threshold.

**Response:**
```json
{
  "segments": [
    {
      "csv_path": "/path/to/predictions.csv",
      "audio_path": "/path/to/concert.mp3",
      "timestamp": "00:15:23",
      "predicted_class": "PUBLIC",
      "confidence": 0.6234,
      "segment_index": 308
    }
  ]
}
```

### Mark Segment Reviewed

```http
POST /api/v1/uncertainty/mark-reviewed
Content-Type: application/json

{
  "csv_path": "/path/to/predictions.csv",
  "segment_index": 308,
  "corrected_class": "APPLAUSE"  // optional
}
```

Updates `reviewed_segments.csv` to prevent re-showing.

## Audio

### Get Audio Segment

```http
GET /api/v1/audio/segment?path={audio_path}&start={seconds}&duration={seconds}
```

Returns audio segment for playback.

**Response:** Binary audio data (WAV format)

## Error Responses

All endpoints return errors in format:

```json
{
  "detail": "Error message description"
}
```

Common status codes:
- `400` - Bad request (invalid parameters)
- `404` - File/resource not found
- `500` - Internal server error

## WebSocket (Future)

Real-time training progress via WebSocket planned for future release.
