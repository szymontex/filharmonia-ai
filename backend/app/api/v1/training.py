"""
Training API - Model retraining endpoints
Now uses PyTorch AST instead of Keras CNN
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict
from pydantic import BaseModel

# Use new AST training service
from app.services.ast_training import get_ast_training_service, TrainingStatus, ModelInfo

router = APIRouter(prefix="/training", tags=["training"])


class ActivateModelRequest(BaseModel):
    filename: str


@router.post("/start")
async def start_training():
    """
    Start a new PyTorch AST model training job
    Returns job_id for status polling
    """
    try:
        service = get_ast_training_service()
        job_id = service.start_training_job()
        return {
            "job_id": job_id,
            "message": "AST training started in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-job")
async def get_active_job() -> Dict:
    """
    Get currently active training job (if any)
    Returns the most recent job in 'training' or 'preparing' status
    """
    service = get_ast_training_service()

    # Find active job
    for job_id, status in service.jobs.items():
        if status.status in ['training', 'preparing']:
            return {
                "job_id": status.job_id,
                "status": status.status,
                "current_epoch": status.current_epoch,
                "total_epochs": status.total_epochs,
                "training_acc": status.training_acc,
                "training_loss": status.training_loss,
                "val_acc": status.val_acc,
                "val_loss": status.val_loss,
                "progress": status.progress,
                "time_elapsed_sec": status.time_elapsed_sec,
                "time_remaining_sec": status.time_remaining_sec,
                "samples_per_class": status.samples_per_class,
                "error": status.error,
                "model_filename": status.model_filename
            }

    return {"job_id": None, "status": "none"}


@router.get("/status/{job_id}")
async def get_training_status(job_id: str) -> Dict:
    """
    Get current status of training job
    Poll this endpoint for real-time updates
    """
    service = get_ast_training_service()
    status = service.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    # Convert dataclass to dict for JSON serialization
    return {
        "job_id": status.job_id,
        "status": status.status,
        "current_epoch": status.current_epoch,
        "total_epochs": status.total_epochs,
        "training_acc": status.training_acc,
        "training_loss": status.training_loss,
        "val_acc": status.val_acc,
        "val_loss": status.val_loss,
        "progress": status.progress,
        "time_elapsed_sec": status.time_elapsed_sec,
        "time_remaining_sec": status.time_remaining_sec,
        "samples_per_class": status.samples_per_class,
        "error": status.error,
        "model_filename": status.model_filename
    }


@router.post("/{job_id}/cancel")
async def cancel_training(job_id: str):
    """
    Request cancellation of training job
    Training will stop gracefully after current epoch
    """
    service = get_ast_training_service()
    status = service.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    service.cancel_job(job_id)
    return {"message": "Cancellation requested - will stop after current epoch"}


@router.get("/models")
async def list_models():
    """
    List all trained models with metadata
    Includes currently active model
    """
    service = get_ast_training_service()
    models = service.get_models_list()
    active_model = service.get_active_model()

    # Convert ModelInfo dataclasses to dicts
    models_data = []
    for model in models:
        # Skip ast_active.pth - it's just a copy, not a source model
        if model.filename == "ast_active.pth":
            continue

        models_data.append({
            "filename": model.filename,
            "model_id": model.model_id,
            "trained_date": model.trained_date,
            "accuracy": model.test_accuracy,
            "val_accuracy": model.val_accuracy,
            "test_accuracy": model.test_accuracy,
            "loss": model.loss,
            "val_loss": model.val_loss,
            "epochs_trained": model.epochs_trained,
            "training_samples": model.training_samples,
            "notes": model.notes,
            "per_class_acc": getattr(model, 'per_class_acc', None),
            "measured_train_acc": getattr(model, 'measured_train_acc', None),
            "dataset_measured_on": getattr(model, 'dataset_measured_on', None)
        })

    # Sort by trained_date descending (newest first)
    models_data.sort(key=lambda m: m["trained_date"], reverse=True)

    return {
        "active_model": active_model,  # Now returns model_id, not filename
        "models": models_data
    }


@router.post("/activate-model")
async def activate_model(request: ActivateModelRequest = Body(...)):
    """
    Switch active AST model (copy selected model to ast_active.pth)
    Automatically reloads model in memory
    """
    service = get_ast_training_service()

    try:
        # 1. Copy model file to ast_active.pth
        service.activate_model(request.filename)

        # 2. Reload model in inference service (without restarting backend)
        print("\n" + "="*60)
        print(f"🔄 RELOADING MODEL: {request.filename}")
        print("="*60)

        from app.services.ast_inference import get_ast_inference_service
        inference_service = get_ast_inference_service()
        inference_service.load_model()  # Reloads from ast_active.pth

        print("✅ Model reloaded successfully!")
        print("="*60 + "\n")

        return {
            "message": f"AST model {request.filename} activated and loaded successfully",
            "reloaded": True,
            "info": "Model is now active and ready for analysis"
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{filename}")
async def delete_model(filename: str):
    """
    Delete a trained AST model file
    Cannot delete active model (ast_active.pth)
    """
    service = get_ast_training_service()

    try:
        service.delete_model(filename)
        return {"message": f"Model {filename} deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-stats")
async def get_training_data_stats():
    """
    Get training data statistics (duration per class)
    Useful for checking data balance before training
    """
    service = get_ast_training_service()
    stats = service.get_training_data_stats()

    total_duration_sec = sum(s["duration_sec"] for s in stats.values())
    total_count = sum(s["count"] for s in stats.values())

    return {
        "stats_per_class": stats,
        "total_duration_sec": total_duration_sec,
        "total_duration_min": total_duration_sec / 60,
        "total_count": total_count
    }


@router.post("/measure-accuracy/{filename}")
async def measure_model_accuracy(filename: str):
    """
    Measure true train/val/test accuracy for a model
    Returns measured accuracies and updates metadata
    """
    service = get_ast_training_service()

    try:
        results = service.measure_model_accuracy(filename)
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {filename} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
