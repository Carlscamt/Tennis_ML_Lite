"""
Model registry for versioning and tracking.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Simple model registry for versioning and tracking.
    """
    
    def __init__(self, models_dir: Path):
        """
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": [], "active_model": None}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def register(
        self,
        model_path: Path,
        metrics: Dict,
        description: str = "",
        set_active: bool = True
    ) -> str:
        """
        Register a trained model.
        
        Args:
            model_path: Path to model file
            metrics: Training metrics
            description: Model description
            set_active: Set as active model
            
        Returns:
            Model version string
        """
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        
        entry = {
            "version": version,
            "path": str(model_path),
            "metrics": metrics,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        
        self.registry["models"].append(entry)
        
        if set_active:
            self.registry["active_model"] = version
        
        self._save_registry()
        
        logger.info(f"Registered model {version}")
        return version
    
    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model entry."""
        active = self.registry.get("active_model")
        if not active:
            return None
        
        for model in self.registry["models"]:
            if model["version"] == active:
                return model
        
        return None
    
    def get_model(self, version: str) -> Optional[Dict]:
        """Get a specific model by version."""
        for model in self.registry["models"]:
            if model["version"] == version:
                return model
        return None
    
    def list_models(self, limit: int = 10) -> List[Dict]:
        """List recent models."""
        return self.registry["models"][-limit:]
    
    def set_active(self, version: str) -> None:
        """Set a model version as active."""
        if not self.get_model(version):
            raise ValueError(f"Model {version} not found")
        
        self.registry["active_model"] = version
        self._save_registry()
        logger.info(f"Set active model to {version}")
    
    def compare_models(self, versions: List[str]) -> "pl.DataFrame":
        """Compare metrics across models."""
        import polars as pl
        
        models = [self.get_model(v) for v in versions if self.get_model(v)]
        
        rows = []
        for m in models:
            row = {"version": m["version"], "created_at": m["created_at"]}
            row.update(m.get("metrics", {}))
            rows.append(row)
        
        return pl.DataFrame(rows)
