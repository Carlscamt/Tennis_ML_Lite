"""
Model registry for versioning and tracking.
"""
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog
from src.utils.observability import Logger

logger = Logger(__name__)

@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    stage: str  # 'Experimental' | 'Staging' | 'Production' | 'Archived'
    trained_at: str  # ISO timestamp
    auc: float
    precision: float
    recall: float
    feature_schema_version: str
    training_dataset_size: int
    model_file: str  # Relative path to model file
    notes: Optional[str] = None

class ModelRegistry:
    """
    MLflow-inspired model registry.
    
    Manages model lifecycle: Experimental → Staging → Production → Archived.
    """
    
    REGISTRY_FILE = "models/registry.json"
    MODELS_DIR = "models"
    
    def __init__(self, model_name: str = "tennis_xgboost", root_dir: Optional[Path] = None):
        """
        Args:
            model_name: Name of the model artifact
            root_dir: Optional root directory override (default: project root)
        """
        self.model_name = model_name
        
        # Determine root based on file location if not provided
        if root_dir is None:
            # Assuming src/model/registry.py -> ROOT is ../../
            self.root_dir = Path(__file__).parent.parent.parent
        else:
            self.root_dir = root_dir
            
        self.models_dir = self.root_dir / self.MODELS_DIR
        self.registry_path = self.models_dir / "registry.json"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def reload(self):
        """Reload registry from disk to pick up external changes."""
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from JSON."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    # DEBUG LOGGING
                    self.registry = data.get(self.model_name, {})
                    logger.log_event('registry_loaded', num_versions=len(self.registry))
            else:
                self.registry = {}
        except Exception as e:
            logger.log_error('registry_load_failed', error=str(e))
            self.registry = {}
    
    def _save_registry(self):
        """Save registry to JSON."""
        try:
            # Load complete file first if it exists to preserve other models
            full_data = {}
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    full_data = json.load(f)
            
            full_data[self.model_name] = self.registry
            
            with open(self.registry_path, 'w') as f:
                json.dump(full_data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            
            # logger.log_event('registry_saved', num_versions=len(self.registry))
        except Exception as e:
            logger.log_error('registry_save_failed', error=str(e))
            raise
    
    def register_model(
        self,
        model_path: str,
        auc: float,
        precision: float,
        recall: float,
        feature_schema_version: str,
        training_dataset_size: int,
        notes: Optional[str] = None,
        stage: str = "Experimental"
    ) -> str:
        """
        Register new model version.
        
        Returns: version identifier (e.g., 'v1.2.3')
        """
        # Generate semantic version
        versions = list(self.registry.keys())
        if not versions:
            major, minor, patch = 1, 0, 0
        else:
            # Parse versions assuming vX.Y.Z
            parsed_versions = []
            for v_str in versions:
                try:
                    v_nums = tuple(map(int, v_str[1:].split('.')))
                    parsed_versions.append(v_nums)
                except ValueError:
                    continue # Skip non-compliant versions if any
            
            if not parsed_versions:
                major, minor, patch = 1, 0, 0
            else:
                latest = max(parsed_versions)
                major, minor, patch = latest
                
                if stage == "Production":
                    major += 1
                    minor, patch = 0, 0
                elif stage == "Staging":
                    minor += 1
                    patch = 0
                else:
                    patch += 1
        
        new_version = f"v{major}.{minor}.{patch}"
        
        # Copy model to versioned location
        version_dir = self.models_dir / new_version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(model_path)
        dest_filename = f"model{source_path.suffix}" # e.g. model.joblib or model.bin or model.json
        dest_path = version_dir / dest_filename
        shutil.copy2(source_path, dest_path)
        
        # Copy metadata if exists
        source_meta = source_path.with_suffix(".meta.json")
        if source_meta.exists():
            dest_meta = version_dir / "model.meta.json"
            shutil.copy2(source_meta, dest_meta)
        
        # Relative path for storage
        rel_model_path = f"{new_version}/{dest_filename}"
        
        # Create model version
        model_version = ModelVersion(
            version=new_version,
            stage=stage,
            trained_at=datetime.now(timezone.utc).isoformat(),
            auc=auc,
            precision=precision,
            recall=recall,
            feature_schema_version=feature_schema_version,
            training_dataset_size=training_dataset_size,
            model_file=rel_model_path,
            notes=notes,
        )
        
        self.registry[new_version] = asdict(model_version)
        self._save_registry()
        
        logger.log_event(
            'model_registered',
            version=new_version,
            stage=stage,
            auc=auc,
        )
        
        return new_version
    
    def transition_stage(self, version: str, new_stage: str) -> None:
        """
        Transition model version to new stage.
        
        Valid stages: Experimental → Staging → Production → Archived
        """
        if version not in self.registry:
            raise ValueError(f"Version {version} not found")
        
        current_stage = self.registry[version]['stage']
        valid_transitions = {
            'Experimental': ['Staging', 'Archived'],
            'Staging': ['Production', 'Experimental', 'Archived'],
            'Production': ['Archived', 'Staging'],
            'Archived': [],
        }
        
        if new_stage not in valid_transitions.get(current_stage, []):
            # Allow force transition? Better to warn
            logger.log_event('forcing_stage_transition', version=version, from_stage=current_stage, to_stage=new_stage)
        
        # Enforce single Production invariant
        if new_stage == 'Production':
            # Enforce minimum AUC threshold
            model_auc = self.registry[version].get('auc', 0.0)
            if model_auc < 0.80:
                raise ValueError(f"Minimum AUC of 0.80 required for Production (Got {model_auc})")

            for v, meta in self.registry.items():
                if meta['stage'] == 'Production' and v != version:
                    logger.log_event('demoting_previous_production', version=v)
                    self.registry[v]['stage'] = 'Archived'
        
        self.registry[version]['stage'] = new_stage
        self._save_registry()
        
        logger.log_event(
            'model_stage_transitioned',
            version=version,
            from_stage=current_stage,
            to_stage=new_stage,
        )
    
    def get_production_model(self) -> Tuple[str, str]:
        """
        Get production model path.
        
        Returns: (version, absolute_model_path)
        """
        
        production_models = [
            v for v, meta in self.registry.items()
            if meta.get('stage') == 'Production'
        ]
        
        if not production_models:
            # Fallback to latout Staging or Experimental mostly for Dev?
            # User wants production logic. 
            # If no production model, maybe return None or raise
            raise RuntimeError("No Production model available")
        
        # Sort by version to get latest if multiple (though transition_stage prevents multiple)
        # Using string sort for vX.Y.Z is risky (v10 < v2), but with tuple mapping it's fine.
        # We rely on strict transition ensuring usually only 1 prod.
        # But let's be robust
        def parse_v(v): 
            try: return tuple(map(int, v[1:].split('.')))
            except: return (0,0,0)
            
        version = max(production_models, key=parse_v)
        
        rel_path = self.registry[version]['model_file']
        abs_path = str(self.models_dir / rel_path)
        
        logger.log_event('production_model_selected', version=version)
        return version, abs_path
    
    def get_challenger_model(self) -> Optional[Tuple[str, str]]:
        """Get Staging model for canary/shadow testing."""
        staging_models = [
            v for v, meta in self.registry.items()
            if meta['stage'] == 'Staging'
        ]
        
        if not staging_models:
            return None
        
        def parse_v(v): 
                try: return tuple(map(int, v[1:].split('.')))
                except: return (0,0,0)

        version = max(staging_models, key=parse_v)
        rel_path = self.registry[version]['model_file']
        abs_path = str(self.models_dir / rel_path)
        
        logger.log_event('challenger_model_selected', version=version)
        return version, abs_path
    
    def list_models(self, stage: Optional[str] = None) -> List[ModelVersion]:
        """List models optionally filtered by stage."""
        filtered = [
            ModelVersion(**meta) for v, meta in self.registry.items()
            if not stage or meta['stage'] == stage
        ]
        
        def parse_v(m): 
            try: return tuple(map(int, m.version[1:].split('.')))
            except: return (0,0,0)
            
        return sorted(filtered, key=parse_v, reverse=True)
    
    def get_model_metadata(self, version: str) -> Dict[str, Any]:
        """Get metadata for specific version."""
        if version not in self.registry:
            raise ValueError(f"Version {version} not found")
        return self.registry[version]
