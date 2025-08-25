"""Model versioning and registry system."""

import json
import pickle
import joblib
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class ModelVersion:
    """Represents a versioned model with metadata."""
    
    def __init__(
        self,
        model: Any,
        version: str,
        model_type: str,
        metrics: Dict[str, float],
        features: List[str],
        parameters: Dict[str, Any],
        training_data_hash: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        self.model = model
        self.version = version
        self.model_type = model_type
        self.metrics = metrics
        self.features = features
        self.parameters = parameters
        self.training_data_hash = training_data_hash
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        self.model_hash = self._compute_model_hash()
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model for integrity checking."""
        try:
            model_bytes = pickle.dumps(self.model)
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        except:
            return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without model object)."""
        return {
            'version': self.version,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'features': self.features,
            'parameters': self.parameters,
            'training_data_hash': self.training_data_hash,
            'tags': self.tags,
            'created_at': self.created_at,
            'model_hash': self.model_hash
        }


class ModelRegistry:
    """Central registry for model versions."""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelVersion] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                # Note: Models themselves are loaded on-demand
                self.metadata = metadata
        else:
            self.metadata = {
                'models': {},
                'current_version': None,
                'champion': None,
                'challengers': []
            }
    
    def _save_registry(self):
        """Save registry metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model: Any,
        model_type: str,
        metrics: Dict[str, float],
        features: List[str],
        parameters: Dict[str, Any],
        version: Optional[str] = None,
        training_data_hash: Optional[str] = None,
        tags: Optional[List[str]] = None,
        set_as_champion: bool = False
    ) -> str:
        """Register a new model version."""
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version()
        
        # Create model version object
        model_version = ModelVersion(
            model=model,
            version=version,
            model_type=model_type,
            metrics=metrics,
            features=features,
            parameters=parameters,
            training_data_hash=training_data_hash,
            tags=tags
        )
        
        # Save model to disk
        model_path = self.registry_dir / f"model_{version}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        self.metadata['models'][version] = model_version.to_dict()
        self.metadata['current_version'] = version
        
        if set_as_champion:
            self.set_champion(version)
        
        self._save_registry()
        
        print(f"✓ Model registered: {version} ({model_type})")
        return version
    
    def _generate_version(self) -> str:
        """Generate a new version string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_versions = list(self.metadata['models'].keys())
        
        if not existing_versions:
            return f"v1.0.0_{timestamp}"
        
        # Parse latest version and increment
        latest = sorted(existing_versions)[-1]
        if latest.startswith('v'):
            parts = latest.split('_')[0][1:].split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"v{major}.{minor}.{patch+1}_{timestamp}"
        
        return f"v1.0.0_{timestamp}"
    
    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Load a specific model version."""
        if version not in self.metadata['models']:
            return None
        
        # Load model from disk
        model_path = self.registry_dir / f"model_{version}.joblib"
        if not model_path.exists():
            return None
        
        model = joblib.load(model_path)
        metadata = self.metadata['models'][version]
        
        return ModelVersion(
            model=model,
            version=version,
            model_type=metadata['model_type'],
            metrics=metadata['metrics'],
            features=metadata['features'],
            parameters=metadata['parameters'],
            training_data_hash=metadata.get('training_data_hash'),
            tags=metadata.get('tags', [])
        )
    
    def get_champion(self) -> Optional[ModelVersion]:
        """Get the current champion model."""
        if self.metadata['champion']:
            return self.get_model(self.metadata['champion'])
        return None
    
    def get_challengers(self) -> List[ModelVersion]:
        """Get all challenger models."""
        challengers = []
        for version in self.metadata['challengers']:
            model = self.get_model(version)
            if model:
                challengers.append(model)
        return challengers
    
    def set_champion(self, version: str):
        """Set a model as the champion."""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")
        
        # Move current champion to challengers if exists
        if self.metadata['champion']:
            if self.metadata['champion'] not in self.metadata['challengers']:
                self.metadata['challengers'].append(self.metadata['champion'])
        
        self.metadata['champion'] = version
        
        # Remove from challengers if present
        if version in self.metadata['challengers']:
            self.metadata['challengers'].remove(version)
        
        self._save_registry()
        print(f"✓ Champion model set: {version}")
    
    def add_challenger(self, version: str):
        """Add a model as a challenger."""
        if version not in self.metadata['models']:
            raise ValueError(f"Model version {version} not found")
        
        if version not in self.metadata['challengers']:
            self.metadata['challengers'].append(version)
        
        self._save_registry()
        print(f"✓ Challenger added: {version}")
    
    def compare_models(self, versions: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare metrics across model versions."""
        if versions is None:
            versions = list(self.metadata['models'].keys())
        
        comparison = []
        for version in versions:
            if version in self.metadata['models']:
                meta = self.metadata['models'][version]
                row = {
                    'version': version,
                    'model_type': meta['model_type'],
                    'created_at': meta['created_at'],
                    'is_champion': version == self.metadata['champion'],
                    'is_challenger': version in self.metadata['challengers']
                }
                row.update(meta['metrics'])
                comparison.append(row)
        
        return pd.DataFrame(comparison).sort_values('created_at', ascending=False)
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models."""
        return self.compare_models()
    
    def delete_model(self, version: str):
        """Delete a model version."""
        if version not in self.metadata['models']:
            return
        
        # Remove from registry
        del self.metadata['models'][version]
        
        # Remove from champion/challengers
        if self.metadata['champion'] == version:
            self.metadata['champion'] = None
        if version in self.metadata['challengers']:
            self.metadata['challengers'].remove(version)
        
        # Delete model file
        model_path = self.registry_dir / f"model_{version}.joblib"
        if model_path.exists():
            model_path.unlink()
        
        self._save_registry()
        print(f"✓ Model deleted: {version}")
    
    def export_champion(self, output_path: str):
        """Export champion model for deployment."""
        champion = self.get_champion()
        if not champion:
            raise ValueError("No champion model set")
        
        output = Path(output_path)
        output.parent.mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(champion.model, output)
        
        # Save metadata
        metadata_path = output.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(champion.to_dict(), f, indent=2)
        
        print(f"✓ Champion exported: {output}")
        return str(output)


class ModelComparator:
    """Compare and evaluate models for A/B testing."""
    
    @staticmethod
    def ab_test(
        champion_model: ModelVersion,
        challenger_model: ModelVersion,
        test_data: pd.DataFrame,
        target_col: str,
        traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """Perform A/B test between champion and challenger."""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        # Split test data
        n = len(test_data)
        split_idx = int(n * traffic_split)
        
        # Ensure features match
        champion_features = champion_model.features
        challenger_features = challenger_model.features
        
        # Get predictions
        X_champion = test_data.iloc[:split_idx][champion_features]
        y_champion = test_data.iloc[:split_idx][target_col]
        
        X_challenger = test_data.iloc[split_idx:][challenger_features]
        y_challenger = test_data.iloc[split_idx:][target_col]
        
        # Predict
        champion_pred = champion_model.model.predict_proba(X_champion)[:, 1]
        challenger_pred = challenger_model.model.predict_proba(X_challenger)[:, 1]
        
        # Calculate metrics
        results = {
            'champion': {
                'version': champion_model.version,
                'auc': roc_auc_score(y_champion, champion_pred),
                'samples': len(y_champion)
            },
            'challenger': {
                'version': challenger_model.version,
                'auc': roc_auc_score(y_challenger, challenger_pred),
                'samples': len(y_challenger)
            }
        }
        
        # Statistical significance test (simplified)
        from scipy import stats
        _, p_value = stats.mannwhitneyu(champion_pred, challenger_pred, alternative='two-sided')
        
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05
        results['winner'] = (
            'challenger' if results['challenger']['auc'] > results['champion']['auc']
            else 'champion'
        )
        
        return results