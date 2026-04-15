from fastapi import APIRouter
from app.ml.checkpoint_loader import model_registry
from app.core.model_manifest import MODEL_MANIFEST

router = APIRouter(prefix='/health', tags=['Health'])

@router.get('')
def health_check():
    return {
        'status': 'ok',
        'model_loaded': model_registry.model is not None,
        'model_metadata': model_registry.metadata,
        'manifest': MODEL_MANIFEST,
    }
