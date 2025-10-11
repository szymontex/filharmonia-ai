"""
Sort Service - port z sortuj.py
Skanuje !NAGRANIA KONCERTÓW i przenosi do SORTED/YYYY/MM/DD
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import eyed3
from app.config import settings

class SortService:
    def __init__(self):
        self.source_folder = settings.NAGRANIA_FOLDER
        self.sorted_folder = settings.SORTED_FOLDER

    def scan_new_files(self) -> List[Dict]:
        """
        Skanuje !NAGRANIA KONCERTÓW i zwraca listę nowych plików do przeniesienia
        """
        new_files = []
        excluded_folders = {'SORTED', 'TRAINING DATA', '!POBRANE'}

        for root, dirs, files in os.walk(self.source_folder):
            # Pomijamy wykluczone foldery
            dirs[:] = [d for d in dirs if d not in excluded_folders]

            for file in files:
                if file.lower().endswith(('.mp3', '.MP3')):
                    path = os.path.join(root, file)

                    try:
                        audiofile = eyed3.load(path)
                        if audiofile is None or audiofile.tag is None or audiofile.tag.title is None:
                            new_files.append({
                                'path': path,
                                'name': file,
                                'size': os.path.getsize(path),
                                'status': 'no_metadata',
                                'error': 'Brak tagu ID3 lub tytułu'
                            })
                            continue

                        title = audiofile.tag.title

                        try:
                            # Parse date from ID3 title: "Untitled %m/%d/%Y %H:%M:%S"
                            record_date = datetime.strptime(title, 'Untitled %m/%d/%Y %H:%M:%S')

                            # Calculate target path
                            target_folder = self.sorted_folder / str(record_date.year) / f"{record_date.month:02d}" / f"{record_date.day:02d}"
                            target_path = target_folder / file

                            # Check if already exists
                            file_info = {
                                'path': path,
                                'name': file,
                                'size': os.path.getsize(path),
                                'date': record_date.strftime('%Y-%m-%d'),
                                'target_path': str(target_path),
                            }

                            if target_path.exists():
                                existing_size = os.path.getsize(target_path)
                                if file_info['size'] == existing_size:
                                    file_info['status'] = 'duplicate'
                                    file_info['existing_path'] = str(target_path)
                                    file_info['existing_size'] = existing_size
                                else:
                                    file_info['status'] = 'exists_different_size'
                                    file_info['existing_path'] = str(target_path)
                                    file_info['existing_size'] = existing_size
                            else:
                                file_info['status'] = 'ready_to_move'

                            new_files.append(file_info)

                        except ValueError as e:
                            new_files.append({
                                'path': path,
                                'name': file,
                                'size': os.path.getsize(path),
                                'status': 'invalid_date',
                                'error': f'Niepoprawny format daty w tytule: {title}'
                            })

                    except Exception as e:
                        new_files.append({
                            'path': path,
                            'name': file,
                            'size': os.path.getsize(path),
                            'status': 'error',
                            'error': str(e)
                        })

        return new_files

    def sort_files(self, file_paths: List[str]) -> Dict:
        """
        Przenosi wybrane pliki do SORTED/YYYY/MM/DD
        """
        results = {
            'moved': [],
            'duplicates_removed': [],
            'renamed': [],
            'errors': []
        }

        for path in file_paths:
            try:
                audiofile = eyed3.load(path)
                if audiofile is None or audiofile.tag is None or audiofile.tag.title is None:
                    results['errors'].append({
                        'path': path,
                        'error': 'Brak metadanych ID3'
                    })
                    continue

                title = audiofile.tag.title
                record_date = datetime.strptime(title, 'Untitled %m/%d/%Y %H:%M:%S')

                # Target folder
                target_folder = self.sorted_folder / str(record_date.year) / f"{record_date.month:02d}" / f"{record_date.day:02d}"
                target_folder.mkdir(parents=True, exist_ok=True)

                file_name = os.path.basename(path)
                target_path = target_folder / file_name

                # Check if exists
                if target_path.exists():
                    if os.path.getsize(path) == os.path.getsize(target_path):
                        # Duplicate - remove source
                        os.remove(path)
                        results['duplicates_removed'].append({
                            'path': path,
                            'name': file_name
                        })
                    else:
                        # Different size - copy with new name
                        base, ext = os.path.splitext(file_name)
                        new_name = f"{base}_copy{ext}"
                        new_target = target_folder / new_name
                        shutil.copy(path, new_target)
                        results['renamed'].append({
                            'path': path,
                            'target': str(new_target)
                        })
                else:
                    # Move file
                    shutil.copy(path, target_path)
                    os.remove(path)  # Remove after successful copy
                    results['moved'].append({
                        'path': path,
                        'target': str(target_path),
                        'date': record_date.strftime('%Y-%m-%d')
                    })

            except Exception as e:
                results['errors'].append({
                    'path': path,
                    'error': str(e)
                })

        return results

# Singleton
_service = None

def get_sort_service():
    global _service
    if _service is None:
        _service = SortService()
    return _service
