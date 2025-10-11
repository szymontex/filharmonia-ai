"""
Sort API - scan and organize files from !NAGRANIA KONCERTÓW
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.sort import get_sort_service

router = APIRouter(prefix="/sort", tags=["sort"])

class ScanResponse(BaseModel):
    files: List[dict]
    total: int
    ready_to_move: int
    duplicates: int
    errors: int

class SortRequest(BaseModel):
    file_paths: List[str]

class SortResponse(BaseModel):
    moved: List[dict]
    duplicates_removed: List[dict]
    renamed: List[dict]
    errors: List[dict]
    summary: dict

@router.get("/scan", response_model=ScanResponse)
async def scan_new_files():
    """
    Skanuje !NAGRANIA KONCERTÓW i zwraca listę plików do przeniesienia
    """
    service = get_sort_service()
    files = service.scan_new_files()

    # Count statuses
    ready_to_move = sum(1 for f in files if f.get('status') == 'ready_to_move')
    duplicates = sum(1 for f in files if f.get('status') == 'duplicate')
    errors = sum(1 for f in files if f.get('status') in ['error', 'invalid_date', 'no_metadata'])

    return ScanResponse(
        files=files,
        total=len(files),
        ready_to_move=ready_to_move,
        duplicates=duplicates,
        errors=errors
    )

@router.post("/execute", response_model=SortResponse)
async def sort_files(request: SortRequest):
    """
    Przenosi wybrane pliki do SORTED/YYYY/MM/DD
    """
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No files selected")

    service = get_sort_service()
    results = service.sort_files(request.file_paths)

    summary = {
        'total_processed': len(request.file_paths),
        'moved': len(results['moved']),
        'duplicates_removed': len(results['duplicates_removed']),
        'renamed': len(results['renamed']),
        'errors': len(results['errors'])
    }

    return SortResponse(
        moved=results['moved'],
        duplicates_removed=results['duplicates_removed'],
        renamed=results['renamed'],
        errors=results['errors'],
        summary=summary
    )

class DeleteDuplicatesRequest(BaseModel):
    file_paths: List[str]

@router.post("/delete-duplicates")
async def delete_duplicates(request: DeleteDuplicatesRequest):
    """
    Usuwa duplikaty z !NAGRANIA KONCERTÓW (zostawia wersję w SORTED)
    """
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No files selected")

    import os
    deleted = []
    errors = []

    for file_path in request.file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted.append(file_path)
            else:
                errors.append({'path': file_path, 'error': 'File not found'})
        except Exception as e:
            errors.append({'path': file_path, 'error': str(e)})

    return {
        'deleted': deleted,
        'deleted_count': len(deleted),
        'errors': errors,
        'error_count': len(errors)
    }
