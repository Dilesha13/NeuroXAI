from pathlib import Path
import torch
from app.core.config import settings
from app.ml.models.mst_gat import MSTGAT

class ModelRegistry:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.metadata = {}

    def load(self):
        ckpt_path = Path(settings.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        model = MSTGAT().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.metadata = {
            'checkpoint_path': str(ckpt_path),
            'best_epoch': ckpt.get('epoch'),
            'best_auc': ckpt.get('best_auc'),
            'best_thr': float(ckpt.get('best_thr', settings.default_threshold)),
        }
        return self

model_registry = ModelRegistry()
