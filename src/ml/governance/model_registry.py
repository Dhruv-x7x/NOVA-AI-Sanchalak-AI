# src/ml/governance/model_registry.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - ML Model Registry v1.0
Version Control and Artifact Management for ML Models

Features:
- Model version tracking with metadata
- Artifact storage and hash verification
- Production promotion/demotion workflow
- Rollback capabilities
- Model lineage tracking
- Complete audit trail for 21 CFR Part 11 compliance
"""

import json
import uuid
# SQLite removed - using PostgreSQL
import shutil
import hashlib
import threading
import pickle
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path


# =============================================================================
# ENUMS
# =============================================================================

class ModelStatus(Enum):
    """Status of a model version"""
    DEVELOPMENT = "development"       # In development/testing
    STAGING = "staging"               # Ready for validation
    PRODUCTION = "production"         # Active in production
    DEPRECATED = "deprecated"         # Marked for removal
    ARCHIVED = "archived"             # Stored but not active


class ModelType(Enum):
    """Types of ML models"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    RANKER = "ranker"
    ANOMALY_DETECTOR = "anomaly_detector"
    ENSEMBLE = "ensemble"


class ArtifactType(Enum):
    """Types of model artifacts"""
    MODEL_PICKLE = "model_pickle"      # Serialized model (pickle/joblib)
    MODEL_ONNX = "model_onnx"          # ONNX format
    SCALER = "scaler"                  # Feature scaler
    ENCODER = "encoder"                # Label/feature encoder
    FEATURE_LIST = "feature_list"      # Feature names
    CALIBRATOR = "calibrator"          # Probability calibrator
    SHAP_EXPLAINER = "shap_explainer"  # SHAP explainer
    METADATA = "metadata"              # Model metadata JSON


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelMetrics:
    """Performance metrics for a model version"""
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    # Regression metrics  
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    
    # Ranking metrics
    ndcg_5: Optional[float] = None
    ndcg_10: Optional[float] = None
    map_score: Optional[float] = None
    
    # Calibration metrics
    brier_score: Optional[float] = None
    ece: Optional[float] = None
    
    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetrics':
        custom = data.pop('custom', {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, custom=custom)


@dataclass
class ModelArtifact:
    """A model artifact file"""
    artifact_id: str
    version_id: str
    artifact_type: ArtifactType
    file_path: str
    file_name: str
    file_hash: str           # SHA-256 hash
    size_bytes: int
    created_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            'artifact_id': self.artifact_id,
            'version_id': self.version_id,
            'artifact_type': self.artifact_type.value,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_hash': self.file_hash,
            'size_bytes': self.size_bytes,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelArtifact':
        return cls(
            artifact_id=data['artifact_id'],
            version_id=data['version_id'],
            artifact_type=ArtifactType(data['artifact_type']),
            file_path=data['file_path'],
            file_name=data['file_name'],
            file_hash=data['file_hash'],
            size_bytes=data['size_bytes'],
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class TrainingConfig:
    """Training configuration for reproducibility"""
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_count: int = 0
    features: List[str] = field(default_factory=list)
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    cross_validation_folds: int = 0
    random_seed: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PromotionRecord:
    """Record of model status change"""
    record_id: str
    version_id: str
    from_status: ModelStatus
    to_status: ModelStatus
    changed_at: datetime
    changed_by: str
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'version_id': self.version_id,
            'from_status': self.from_status.value,
            'to_status': self.to_status.value,
            'changed_at': self.changed_at.isoformat(),
            'changed_by': self.changed_by,
            'reason': self.reason
        }


@dataclass
class ModelVersion:
    """A specific version of a model"""
    version_id: str
    model_name: str
    version: str                      # Semantic version (e.g., "1.2.0")
    model_type: ModelType
    status: ModelStatus
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    # Metadata
    created_by: str
    description: str = ""
    
    # Training info
    training_data_hash: str = ""       # Hash of training data
    training_config: Optional[TrainingConfig] = None
    
    # Metrics
    metrics: Optional[ModelMetrics] = None
    
    # Artifacts
    artifacts: List[ModelArtifact] = field(default_factory=list)
    
    # Lineage
    parent_version: Optional[str] = None   # Previous version this was derived from
    
    # Promotion history
    promotion_history: List[PromotionRecord] = field(default_factory=list)
    
    # Checksums
    signature_hash: str = ""            # Combined hash of all artifacts
    
    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'version': self.version,
            'model_type': self.model_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description,
            'training_data_hash': self.training_data_hash,
            'training_config': self.training_config.to_dict() if self.training_config else None,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'artifacts': [a.to_dict() for a in self.artifacts],
            'parent_version': self.parent_version,
            'promotion_history': [p.to_dict() for p in self.promotion_history],
            'signature_hash': self.signature_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        return cls(
            version_id=data['version_id'],
            model_name=data['model_name'],
            version=data['version'],
            model_type=ModelType(data['model_type']),
            status=ModelStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            created_by=data['created_by'],
            description=data.get('description', ''),
            training_data_hash=data.get('training_data_hash', ''),
            training_config=TrainingConfig.from_dict(data['training_config']) if data.get('training_config') else None,
            metrics=ModelMetrics.from_dict(data['metrics']) if data.get('metrics') else None,
            artifacts=[ModelArtifact.from_dict(a) for a in data.get('artifacts', [])],
            parent_version=data.get('parent_version'),
            promotion_history=[],  # Loaded separately
            signature_hash=data.get('signature_hash', '')
        )


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """
    Version control and artifact management for ML models.
    
    Features:
    - Register new model versions with artifacts
    - Track training configuration and metrics
    - Promote/demote models through status workflow
    - Rollback to previous versions
    - Verify artifact integrity with checksums
    - Full audit trail for regulatory compliance
    """
    
    def __init__(
        self,
        db_path: str = "data/ml_governance/model_registry.db",
        artifact_dir: str = "data/ml_governance/artifacts"
    ):
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Model versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    training_data_hash TEXT,
                    training_config TEXT,
                    metrics TEXT,
                    parent_version TEXT,
                    signature_hash TEXT,
                    version_data TEXT NOT NULL,
                    UNIQUE(model_name, version)
                )
            ''')
            
            # Model artifacts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
                )
            ''')
            
            # Promotion history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS promotion_history (
                    record_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_model ON model_versions(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_status ON model_versions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifact_version ON model_artifacts(version_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_promo_version ON promotion_history(version_id)')
            
            conn.commit()
    
    # =========================================================================
    # Registration
    # =========================================================================
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_type: ModelType,
        artifact_paths: Dict[ArtifactType, str],
        metrics: ModelMetrics,
        training_config: TrainingConfig,
        created_by: str,
        description: str = "",
        training_data_hash: str = "",
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., "risk_classifier")
            version: Semantic version (e.g., "1.0.0")
            model_type: Type of model
            artifact_paths: Dict mapping artifact types to file paths
            metrics: Model performance metrics
            training_config: Training configuration
            created_by: User who created this version
            description: Optional description
            training_data_hash: Hash of training data for reproducibility
            parent_version: Optional parent version ID
            
        Returns:
            ModelVersion object
        """
        version_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Copy artifacts and compute hashes
        artifacts = []
        hash_inputs = []
        
        for artifact_type, source_path in artifact_paths.items():
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Artifact not found: {source_path}")
            
            # Compute hash
            file_hash = self._compute_file_hash(source)
            
            # Copy to artifact directory
            dest_dir = self.artifact_dir / model_name / version
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / source.name
            
            shutil.copy2(source, dest_path)
            
            artifact = ModelArtifact(
                artifact_id=str(uuid.uuid4()),
                version_id=version_id,
                artifact_type=artifact_type,
                file_path=str(dest_path),
                file_name=source.name,
                file_hash=file_hash,
                size_bytes=source.stat().st_size,
                created_at=now
            )
            artifacts.append(artifact)
            hash_inputs.append(file_hash)
        
        # Compute signature hash (combined hash of all artifacts)
        signature_hash = hashlib.sha256(''.join(sorted(hash_inputs)).encode()).hexdigest()
        
        # Create model version
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            status=ModelStatus.DEVELOPMENT,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            description=description,
            training_data_hash=training_data_hash,
            training_config=training_config,
            metrics=metrics,
            artifacts=artifacts,
            parent_version=parent_version,
            signature_hash=signature_hash
        )
        
        # Save to database
        self._save_version(model_version)
        
        return model_version
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _save_version(self, version: ModelVersion):
        """Save model version to database - DISABLED (PostgreSQL migration)"""
        pass

    # =========================================================================
    # Retrieval
    # =========================================================================
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get a model version.
        
        Args:
            model_name: Name of the model
            version: Specific version, or None for latest
            
        Returns:
            ModelVersion or None
        """
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            if version:
                cursor.execute('''
                    SELECT version_data, status FROM model_versions
                    WHERE model_name = ? AND version = ?
                ''', (model_name, version))
            else:
                # Get latest version
                cursor.execute('''
                    SELECT version_data, status FROM model_versions
                    WHERE model_name = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (model_name,))
            
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                # Override status from database column (which is current)
                data['status'] = row[1]
                model_version = ModelVersion.from_dict(data)
                
                # Load artifacts
                model_version.artifacts = self._load_artifacts(model_version.version_id)
                
                # Load promotion history
                model_version.promotion_history = self._load_promotion_history(model_version.version_id)
                
                return model_version
        
        return None
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the production version of a model"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT version_data, status FROM model_versions
                WHERE model_name = ? AND status = 'production'
                ORDER BY updated_at DESC LIMIT 1
            ''', (model_name,))
            
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                # Override status from database column (which is current)
                data['status'] = row[1]
                model_version = ModelVersion.from_dict(data)
                model_version.artifacts = self._load_artifacts(model_version.version_id)
                return model_version
        
        return None
    
    def _load_artifacts(self, version_id: str) -> List[ModelArtifact]:
        """Load artifacts for a version"""
        artifacts = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT artifact_id, version_id, artifact_type, file_path,
                       file_name, file_hash, size_bytes, created_at
                FROM model_artifacts WHERE version_id = ?
            ''', (version_id,))
            
            for row in cursor.fetchall():
                artifact = ModelArtifact(
                    artifact_id=row[0],
                    version_id=row[1],
                    artifact_type=ArtifactType(row[2]),
                    file_path=row[3],
                    file_name=row[4],
                    file_hash=row[5],
                    size_bytes=row[6],
                    created_at=datetime.fromisoformat(row[7])
                )
                artifacts.append(artifact)
        
        return artifacts
    
    def _load_promotion_history(self, version_id: str) -> List[PromotionRecord]:
        """Load promotion history for a version"""
        history = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT record_id, version_id, from_status, to_status,
                       changed_at, changed_by, reason
                FROM promotion_history WHERE version_id = ?
                ORDER BY changed_at DESC
            ''', (version_id,))
            
            for row in cursor.fetchall():
                record = PromotionRecord(
                    record_id=row[0],
                    version_id=row[1],
                    from_status=ModelStatus(row[2]),
                    to_status=ModelStatus(row[3]),
                    changed_at=datetime.fromisoformat(row[4]),
                    changed_by=row[5],
                    reason=row[6] or ""
                )
                history.append(record)
        
        return history
    
    def get_version_history(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        versions = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT version_data FROM model_versions
                WHERE model_name = ?
                ORDER BY created_at DESC
            ''', (model_name,))
            
            for row in cursor.fetchall():
                data = json.loads(row[0])
                versions.append(ModelVersion.from_dict(data))
        
        return versions
    
    def list_models(self) -> List[str]:
        """Get list of all registered model names"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT model_name FROM model_versions ORDER BY model_name')
            return [row[0] for row in cursor.fetchall()]
    
    # =========================================================================
    # Promotion / Demotion
    # =========================================================================
    
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        approved_by: str,
        reason: str = "Approved for production"
    ) -> bool:
        """Promote a model version to production status - DISABLED (PostgreSQL migration)"""
        return True

    def demote_from_production(
        self,
        model_name: str,
        version: str,
        demoted_by: str,
        reason: str = "Demoted from production"
    ) -> bool:
        """Demote a model version from production - DISABLED (PostgreSQL migration)"""
        return True

    def rollback(
        self,
        model_name: str,
        target_version: str,
        rolled_back_by: str,
        reason: str = "Rollback to previous version"
    ) -> bool:
        """
        Rollback to a previous version.
        
        Demotes current production and promotes target version.
        """
        return self.promote_to_production(
            model_name,
            target_version,
            rolled_back_by,
            f"ROLLBACK: {reason}"
        )
    
    # =========================================================================
    # Verification
    # =========================================================================
    
    def verify_artifact_integrity(
        self,
        model_name: str,
        version: str
    ) -> Tuple[bool, List[str]]:
        """
        Verify integrity of model artifacts.
        
        Returns:
            Tuple of (all_valid, list_of_issues)
        """
        model_version = self.get_model(model_name, version)
        
        if not model_version:
            return False, [f"Model version not found: {model_name} v{version}"]
        
        issues = []
        
        for artifact in model_version.artifacts:
            artifact_path = Path(artifact.file_path)
            
            # Check file exists
            if not artifact_path.exists():
                issues.append(f"Missing artifact: {artifact.file_name}")
                continue
            
            # Verify hash
            current_hash = self._compute_file_hash(artifact_path)
            if current_hash != artifact.file_hash:
                issues.append(f"Hash mismatch for {artifact.file_name}")
            
            # Verify size
            current_size = artifact_path.stat().st_size
            if current_size != artifact.size_bytes:
                issues.append(f"Size mismatch for {artifact.file_name}")
        
        return len(issues) == 0, issues
    
    # =========================================================================
    # Comparison
    # =========================================================================
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two versions of a model"""
        v1 = self.get_model(model_name, version1)
        v2 = self.get_model(model_name, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'metrics_comparison': {},
            'config_differences': {},
            'artifact_differences': []
        }
        
        # Compare metrics
        if v1.metrics and v2.metrics:
            m1 = v1.metrics.to_dict()
            m2 = v2.metrics.to_dict()
            
            all_metrics = set(m1.keys()) | set(m2.keys())
            for metric in all_metrics:
                val1 = m1.get(metric)
                val2 = m2.get(metric)
                
                if val1 is not None and val2 is not None:
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        diff = val2 - val1
                        pct_change = (diff / val1 * 100) if val1 != 0 else 0
                        comparison['metrics_comparison'][metric] = {
                            'v1': val1,
                            'v2': val2,
                            'difference': diff,
                            'percent_change': pct_change
                        }
        
        # Compare training config
        if v1.training_config and v2.training_config:
            c1 = v1.training_config.to_dict()
            c2 = v2.training_config.to_dict()
            
            for key in set(c1.keys()) | set(c2.keys()):
                if c1.get(key) != c2.get(key):
                    comparison['config_differences'][key] = {
                        'v1': c1.get(key),
                        'v2': c2.get(key)
                    }
        
        # Compare artifacts
        v1_artifacts = {a.artifact_type: a for a in v1.artifacts}
        v2_artifacts = {a.artifact_type: a for a in v2.artifacts}
        
        all_types = set(v1_artifacts.keys()) | set(v2_artifacts.keys())
        for atype in all_types:
            a1 = v1_artifacts.get(atype)
            a2 = v2_artifacts.get(atype)
            
            if a1 and a2:
                if a1.file_hash != a2.file_hash:
                    comparison['artifact_differences'].append({
                        'type': atype.value,
                        'change': 'modified',
                        'v1_hash': a1.file_hash[:8],
                        'v2_hash': a2.file_hash[:8]
                    })
            elif a1:
                comparison['artifact_differences'].append({
                    'type': atype.value,
                    'change': 'removed'
                })
            else:
                comparison['artifact_differences'].append({
                    'type': atype.value,
                    'change': 'added'
                })
        
        return comparison
    
    def get_model_lineage(
        self,
        model_name: str,
        version: str
    ) -> List[ModelVersion]:
        """Get the lineage of a model version"""
        lineage = []
        current = self.get_model(model_name, version)
        
        while current:
            lineage.append(current)
            if current.parent_version:
                # Find parent by version_id
                # DISABLED: SQLite replaced with PostgreSQL
                if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT version FROM model_versions
                        WHERE version_id = ?
                    ''', (current.parent_version,))
                    row = cursor.fetchone()
                    if row:
                        current = self.get_model(model_name, row[0])
                    else:
                        current = None
            else:
                current = None
        
        return lineage
    
    # =========================================================================
    # Artifact Loading
    # =========================================================================
    
    def load_model_artifact(
        self,
        model_name: str,
        version: Optional[str] = None,
        artifact_type: ArtifactType = ArtifactType.MODEL_PICKLE
    ) -> Any:
        """
        Load a model artifact from storage.
        
        Args:
            model_name: Name of the model
            version: Specific version or None for production
            artifact_type: Type of artifact to load
            
        Returns:
            Loaded artifact (e.g., sklearn model)
        """
        if version:
            model_version = self.get_model(model_name, version)
        else:
            model_version = self.get_production_model(model_name)
        
        if not model_version:
            raise ValueError(f"Model not found: {model_name}")
        
        # Find artifact
        artifact = next(
            (a for a in model_version.artifacts if a.artifact_type == artifact_type),
            None
        )
        
        if not artifact:
            raise ValueError(f"Artifact type {artifact_type.value} not found for {model_name}")
        
        artifact_path = Path(artifact.file_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        
        # Load based on type
        if artifact_type in [ArtifactType.MODEL_PICKLE, ArtifactType.SCALER, 
                             ArtifactType.ENCODER, ArtifactType.CALIBRATOR]:
            with open(artifact_path, 'rb') as f:
                return pickle.load(f)
        elif artifact_type == ArtifactType.FEATURE_LIST:
            with open(artifact_path, 'r') as f:
                return json.load(f)
        elif artifact_type == ArtifactType.METADATA:
            with open(artifact_path, 'r') as f:
                return json.load(f)
        else:
            # Return raw bytes for other types
            with open(artifact_path, 'rb') as f:
                return f.read()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        # Initialize default values to avoid UnboundLocalError if disabled block is skipped
        total_models = 0
        total_versions = 0
        by_status = {}
        total_storage = 0
        recent_promotions = 0
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Model counts
            cursor.execute('SELECT COUNT(DISTINCT model_name) FROM model_versions')
            total_models = cursor.fetchone()[0]
            
            # Version counts
            cursor.execute('SELECT COUNT(*) FROM model_versions')
            total_versions = cursor.fetchone()[0]
            
            # By status
            cursor.execute('''
                SELECT status, COUNT(*) FROM model_versions GROUP BY status
            ''')
            by_status = dict(cursor.fetchall())
            
            # Artifact storage
            cursor.execute('SELECT SUM(size_bytes) FROM model_artifacts')
            total_storage = cursor.fetchone()[0] or 0
            
            # Recent promotions
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) FROM promotion_history
                WHERE changed_at >= ? AND to_status = 'production'
            ''', (week_ago,))
            recent_promotions = cursor.fetchone()[0]
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'versions_by_status': by_status,
            'production_models': by_status.get('production', 0),
            'total_storage_bytes': total_storage,
            'total_storage_mb': total_storage / (1024 * 1024),
            'promotions_last_7_days': recent_promotions
        }


# =============================================================================
# SINGLETON
# =============================================================================

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create the model registry singleton"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def reset_model_registry():
    """Reset the model registry singleton (for testing)"""
    global _model_registry
    _model_registry = None
