"""
a6on-i - Risk Classifier v1.2 (FIXED)
Phase 3.2: XGBoost + LightGBM Ensemble with SHAP Explainability

FIXES v1.2:
- Fixed sklearn calibration compatibility
- Fixed SHAP multi-class array handling

Features:
- XGBoost + LightGBM ensemble
- Calibrated probabilities
- SHAP explainability
- Threshold optimization
- Cross-validation with stratification
- Model persistence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ML imports
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    warnings.warn(f"XGBoost import failed: {e}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    warnings.warn(f"LightGBM import failed: {e}")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    warnings.warn(f"SHAP import failed: {e}")

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class RiskClassifierConfig:
    """Risk Classifier Configuration"""
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'verbosity': 0
    })
    
    lgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'force_col_wise': True
    })
    
    xgb_weight: float = 0.5
    lgb_weight: float = 0.5
    calibration_method: str = 'sigmoid'
    calibration_cv: int = 3
    threshold_metric: str = 'f1'
    
    class_names: List[str] = field(default_factory=lambda: [
        'Critical', 'High', 'Low', 'Medium'
    ])


class RiskClassifier:
    """Patient Risk Classifier using XGBoost + LightGBM Ensemble"""
    
    def __init__(self, config: RiskClassifierConfig = None):
        self.config = config or RiskClassifierConfig()
        self.xgb_model = None
        self.lgb_model = None
        self.xgb_calibrated = None
        self.lgb_calibrated = None
        self.is_calibrated = False
        self.xgb_explainer = None
        self.lgb_explainer = None
        self.cv_results: Dict = {}
        self.feature_importance: pd.DataFrame = None
        self.shap_values = None
        self.optimal_thresholds: Dict = {}
        self.training_history: Dict = {}
        self.feature_names: List[str] = []
        self.n_classes: int = 4
        
    def _create_xgb_model(self):
        params = self.config.xgb_params.copy()
        params['random_state'] = self.config.random_state
        params['n_jobs'] = self.config.n_jobs
        params['num_class'] = self.n_classes
        return xgb.XGBClassifier(**params)
    
    def _create_lgb_model(self):
        params = self.config.lgb_params.copy()
        params['random_state'] = self.config.random_state
        params['n_jobs'] = self.config.n_jobs
        params['num_class'] = self.n_classes
        return lgb.LGBMClassifier(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train the ensemble model"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING RISK CLASSIFIER")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        self.feature_names = list(X_train.columns)
        self.n_classes = len(np.unique(y_train))
        
        logger.info(f"  Training samples: {len(X_train):,}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Classes: {self.n_classes}")
        
        if XGBOOST_AVAILABLE:
            logger.info("\n  Training XGBoost...")
            self.xgb_model = self._create_xgb_model()
            if X_val is not None:
                self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                self.xgb_model.fit(X_train, y_train)
            logger.info("  XGBoost trained ✓")
        
        if LIGHTGBM_AVAILABLE:
            logger.info("  Training LightGBM...")
            self.lgb_model = self._create_lgb_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if X_val is not None:
                    self.lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    self.lgb_model.fit(X_train, y_train)
            logger.info("  LightGBM trained ✓")
        
        self._calculate_feature_importance()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n  Training completed in {duration:.2f} seconds")
        
        self.training_history = {
            'duration': duration,
            'n_samples': len(X_train),
            'n_features': len(self.feature_names)
        }
        return self.training_history
    
    def _calculate_feature_importance(self):
        """Calculate combined feature importance"""
        logger.info("\n  Calculating feature importance...")
        
        importance_df = pd.DataFrame({'feature': self.feature_names})
        
        if self.xgb_model is not None:
            importance_df['xgb_importance'] = self.xgb_model.feature_importances_
        
        if self.lgb_model is not None:
            importance_df['lgb_importance'] = self.lgb_model.feature_importances_
        
        if 'xgb_importance' in importance_df and 'lgb_importance' in importance_df:
            importance_df['combined_importance'] = (
                importance_df['xgb_importance'] * self.config.xgb_weight +
                importance_df['lgb_importance'] * self.config.lgb_weight
            )
        elif 'xgb_importance' in importance_df:
            importance_df['combined_importance'] = importance_df['xgb_importance']
        elif 'lgb_importance' in importance_df:
            importance_df['combined_importance'] = importance_df['lgb_importance']
        
        if 'combined_importance' in importance_df:
            total = importance_df['combined_importance'].sum()
            if total > 0:
                importance_df['importance_pct'] = (
                    importance_df['combined_importance'] / total * 100
                ).round(2)
        
        importance_df = importance_df.sort_values(
            'combined_importance', ascending=False
        ).reset_index(drop=True)
        
        self.feature_importance = importance_df
        
        logger.info("  Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row.get('importance_pct', 0):.1f}%")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions"""
        proba = np.zeros((len(X), self.n_classes))
        weight_sum = 0
        
        if self.is_calibrated:
            if self.xgb_calibrated is not None:
                proba += self.xgb_calibrated.predict_proba(X) * self.config.xgb_weight
                weight_sum += self.config.xgb_weight
            if self.lgb_calibrated is not None:
                proba += self.lgb_calibrated.predict_proba(X) * self.config.lgb_weight
                weight_sum += self.config.lgb_weight
        else:
            if self.xgb_model is not None:
                proba += self.xgb_model.predict_proba(X) * self.config.xgb_weight
                weight_sum += self.config.xgb_weight
            if self.lgb_model is not None:
                proba += self.lgb_model.predict_proba(X) * self.config.lgb_weight
                weight_sum += self.config.lgb_weight
        
        if weight_sum > 0:
            proba /= weight_sum
        
        return proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def calibrate(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate probability predictions"""
        logger.info("\n" + "=" * 70)
        logger.info("CALIBRATING PROBABILITIES")
        logger.info("=" * 70)
        
        y_proba_before = self.predict_proba(X_val)
        
        try:
            if self.xgb_model is not None:
                logger.info("  Calibrating XGBoost...")
                self.xgb_calibrated = CalibratedClassifierCV(
                    estimator=self.xgb_model,
                    method=self.config.calibration_method,
                    cv=self.config.calibration_cv
                )
                self.xgb_calibrated.fit(X_val, y_val)
                logger.info("  XGBoost calibrated ✓")
            
            if self.lgb_model is not None:
                logger.info("  Calibrating LightGBM...")
                self.lgb_calibrated = CalibratedClassifierCV(
                    estimator=self.lgb_model,
                    method=self.config.calibration_method,
                    cv=self.config.calibration_cv
                )
                self.lgb_calibrated.fit(X_val, y_val)
                logger.info("  LightGBM calibrated ✓")
            
            self.is_calibrated = True
            
            y_proba_after = self.predict_proba(X_val)
            
            brier_before = 0
            brier_after = 0
            for i in range(self.n_classes):
                y_binary = (y_val == i).astype(int)
                brier_before += brier_score_loss(y_binary, y_proba_before[:, i])
                brier_after += brier_score_loss(y_binary, y_proba_after[:, i])
            
            brier_before /= self.n_classes
            brier_after /= self.n_classes
            
            improvement = (brier_before - brier_after) / brier_before * 100 if brier_before > 0 else 0
            
            logger.info(f"  Brier score: {brier_before:.4f} → {brier_after:.4f} ({improvement:+.1f}%)")
            logger.info(f"  Calibration method: {self.config.calibration_method}")
            logger.info("  Calibration complete ✓")
            
        except Exception as e:
            logger.warning(f"  Calibration failed: {e}")
            self.is_calibrated = False
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'test') -> Dict:
        """Evaluate model performance"""
        logger.info("\n" + "=" * 70)
        logger.info(f"EVALUATING ON {dataset_name.upper()} SET")
        logger.info("=" * 70)
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average=None, zero_division=0)
        recall = recall_score(y, y_pred, average=None, zero_division=0)
        f1 = f1_score(y, y_pred, average=None, zero_division=0)
        
        f1_weighted = f1_score(y, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)
        
        try:
            y_bin = label_binarize(y, classes=np.arange(self.n_classes))
            roc_auc = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        except:
            roc_auc = None
        
        cm = confusion_matrix(y, y_pred)
        
        results = {
            'dataset': dataset_name,
            'samples': len(y),
            'accuracy': round(accuracy, 4),
            'f1_weighted': round(f1_weighted, 4),
            'f1_macro': round(f1_macro, 4),
            'roc_auc': round(roc_auc, 4) if roc_auc else None,
            'per_class': {},
            'confusion_matrix': cm.tolist()
        }
        
        for i in range(min(self.n_classes, len(self.config.class_names))):
            class_name = self.config.class_names[i]
            results['per_class'][class_name] = {
                'precision': round(precision[i], 4) if i < len(precision) else 0,
                'recall': round(recall[i], 4) if i < len(recall) else 0,
                'f1': round(f1[i], 4) if i < len(f1) else 0,
                'support': int(cm[i].sum()) if i < len(cm) else 0
            }
        
        logger.info(f"\n  Samples: {len(y):,}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        if roc_auc:
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        logger.info(f"\n  Per-class metrics:")
        for i in range(min(self.n_classes, len(self.config.class_names))):
            class_name = self.config.class_names[i]
            support = int(cm[i].sum()) if i < len(cm) else 0
            p = precision[i] if i < len(precision) else 0
            r = recall[i] if i < len(recall) else 0
            f = f1[i] if i < len(f1) else 0
            logger.info(f"    {class_name}: P={p:.3f} R={r:.3f} F1={f:.3f} (n={support})")
        
        logger.info(f"\n  Confusion Matrix:")
        header = "  " + " ".join([f"{c[:4]:>6}" for c in self.config.class_names[:self.n_classes]])
        logger.info(header)
        for i, row in enumerate(cm):
            if i < len(self.config.class_names):
                row_str = " ".join([f"{v:>6}" for v in row])
                logger.info(f"  {self.config.class_names[i][:4]:>4}: {row_str}")
        
        return results
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform stratified k-fold cross-validation"""
        logger.info("\n" + "=" * 70)
        logger.info(f"CROSS-VALIDATION ({self.config.cv_folds}-FOLD)")
        logger.info("=" * 70)
        
        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"\n  Fold {fold}/{self.config.cv_folds}")
            
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            fold_classifier = RiskClassifier(self.config)
            fold_classifier.train(X_fold_train, y_fold_train)
            
            y_pred = fold_classifier.predict(X_fold_val)
            
            fold_metrics = {
                'fold': fold,
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'f1_weighted': f1_score(y_fold_val, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_fold_val, y_pred, average='macro', zero_division=0)
            }
            fold_results.append(fold_metrics)
            
            logger.info(
                f"    Acc={fold_metrics['accuracy']:.4f} "
                f"F1w={fold_metrics['f1_weighted']:.4f} "
                f"F1m={fold_metrics['f1_macro']:.4f}"
            )
        
        cv_results = {
            'n_folds': self.config.cv_folds,
            'accuracy_mean': np.mean([r['accuracy'] for r in fold_results]),
            'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'f1_weighted_mean': np.mean([r['f1_weighted'] for r in fold_results]),
            'f1_weighted_std': np.std([r['f1_weighted'] for r in fold_results]),
            'f1_macro_mean': np.mean([r['f1_macro'] for r in fold_results]),
            'f1_macro_std': np.std([r['f1_macro'] for r in fold_results]),
            'fold_results': fold_results
        }
        
        logger.info(f"\n  CV Results:")
        logger.info(f"    Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        logger.info(f"    F1 (weighted): {cv_results['f1_weighted_mean']:.4f} ± {cv_results['f1_weighted_std']:.4f}")
        logger.info(f"    F1 (macro): {cv_results['f1_macro_mean']:.4f} ± {cv_results['f1_macro_std']:.4f}")
        
        self.cv_results = cv_results
        return cv_results
    
    def optimize_thresholds(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize classification thresholds per class"""
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZING CLASSIFICATION THRESHOLDS")
        logger.info("=" * 70)
        
        y_proba = self.predict_proba(X)
        y_bin = label_binarize(y, classes=np.arange(self.n_classes))
        
        optimal_thresholds = {}
        
        for i in range(self.n_classes):
            class_name = self.config.class_names[i] if i < len(self.config.class_names) else f"Class_{i}"
            
            precision_arr, recall_arr, thresholds = precision_recall_curve(
                y_bin[:, i], y_proba[:, i]
            )
            
            f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
            
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            else:
                optimal_threshold = 0.5
                best_idx = 0
            
            optimal_thresholds[i] = float(optimal_threshold)
            
            p = precision_arr[best_idx] if best_idx < len(precision_arr) else 0
            r = recall_arr[best_idx] if best_idx < len(recall_arr) else 0
            f = f1_scores[best_idx] if best_idx < len(f1_scores) else 0
            
            logger.info(
                f"  {class_name}: threshold={optimal_threshold:.3f} "
                f"(P={p:.3f}, R={r:.3f}, F1={f:.3f})"
            )
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def explain_shap(self, X: pd.DataFrame, max_samples: int = 1000) -> Dict:
        """Generate SHAP explanations - FIXED for multi-class"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping explainability.")
            return {}
        
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING SHAP EXPLANATIONS")
        logger.info("=" * 70)
        
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=self.config.random_state)
            logger.info(f"  Sampling {max_samples} from {len(X)} for SHAP")
        else:
            X_sample = X
        
        shap_results = {}
        
        # XGBoost SHAP
        if self.xgb_model is not None:
            logger.info("  Computing XGBoost SHAP values...")
            try:
                self.xgb_explainer = shap.TreeExplainer(self.xgb_model)
                xgb_shap = self.xgb_explainer.shap_values(X_sample)
                shap_results['xgb_shap_computed'] = True
                logger.info("  XGBoost SHAP complete ✓")
                
                # Process SHAP values for importance
                mean_shap = self._compute_shap_importance(xgb_shap)
                if mean_shap is not None:
                    shap_results['xgb_mean_shap'] = mean_shap
                    
            except Exception as e:
                logger.warning(f"  XGBoost SHAP failed: {e}")
        
        # LightGBM SHAP
        if self.lgb_model is not None:
            logger.info("  Computing LightGBM SHAP values...")
            try:
                self.lgb_explainer = shap.TreeExplainer(self.lgb_model)
                lgb_shap = self.lgb_explainer.shap_values(X_sample)
                shap_results['lgb_shap_computed'] = True
                logger.info("  LightGBM SHAP complete ✓")
                
                mean_shap = self._compute_shap_importance(lgb_shap)
                if mean_shap is not None:
                    shap_results['lgb_mean_shap'] = mean_shap
                    
            except Exception as e:
                logger.warning(f"  LightGBM SHAP failed: {e}")
        
        # Combine and create importance DataFrame
        self._create_shap_importance_df(shap_results)
        
        self.shap_values = shap_results
        return shap_results
    
    def _compute_shap_importance(self, shap_values) -> Optional[np.ndarray]:
        """Compute mean absolute SHAP values handling different formats"""
        try:
            if isinstance(shap_values, list):
                # Multi-class: list of arrays (n_classes, each is n_samples x n_features)
                mean_shap = np.zeros(len(self.feature_names))
                valid_count = 0
                for class_shap in shap_values:
                    if isinstance(class_shap, np.ndarray):
                        if class_shap.ndim == 2 and class_shap.shape[1] == len(self.feature_names):
                            mean_shap += np.abs(class_shap).mean(axis=0)
                            valid_count += 1
                if valid_count > 0:
                    mean_shap /= valid_count
                    return mean_shap
                    
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    # Shape: (n_samples, n_features, n_classes)
                    if shap_values.shape[1] == len(self.feature_names):
                        return np.abs(shap_values).mean(axis=(0, 2))
                elif shap_values.ndim == 2:
                    # Shape: (n_samples, n_features)
                    if shap_values.shape[1] == len(self.feature_names):
                        return np.abs(shap_values).mean(axis=0)
                        
        except Exception as e:
            logger.warning(f"  SHAP importance computation error: {e}")
        
        return None
    
    def _create_shap_importance_df(self, shap_results: Dict):
        """Create SHAP importance DataFrame from results"""
        mean_shap = None
        
        if 'xgb_mean_shap' in shap_results and shap_results['xgb_mean_shap'] is not None:
            mean_shap = shap_results['xgb_mean_shap']
        elif 'lgb_mean_shap' in shap_results and shap_results['lgb_mean_shap'] is not None:
            mean_shap = shap_results['lgb_mean_shap']
        
        # Average if both available
        if ('xgb_mean_shap' in shap_results and shap_results['xgb_mean_shap'] is not None and
            'lgb_mean_shap' in shap_results and shap_results['lgb_mean_shap'] is not None):
            mean_shap = (shap_results['xgb_mean_shap'] + shap_results['lgb_mean_shap']) / 2
        
        if mean_shap is not None and len(mean_shap) == len(self.feature_names):
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': mean_shap
            }).sort_values('shap_importance', ascending=False).reset_index(drop=True)
            
            total = shap_importance['shap_importance'].sum()
            if total > 0:
                shap_importance['shap_pct'] = (shap_importance['shap_importance'] / total * 100).round(2)
            
            shap_results['shap_importance'] = shap_importance
            
            logger.info("\n  Top 10 SHAP features:")
            for _, row in shap_importance.head(10).iterrows():
                pct = row.get('shap_pct', 0)
                logger.info(f"    {row['feature']}: {pct:.2f}%")
        else:
            logger.warning("  Could not create SHAP importance DataFrame")
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save model and artifacts"""
        logger.info("\n" + "=" * 70)
        logger.info("SAVING MODEL ARTIFACTS")
        logger.info("=" * 70)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        if self.xgb_model is not None:
            xgb_path = output_dir / 'risk_classifier_xgb.pkl'
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            saved_files['xgb_model'] = xgb_path
            logger.info(f"  ✅ XGBoost model: {xgb_path.name}")
        
        if self.lgb_model is not None:
            lgb_path = output_dir / 'risk_classifier_lgb.pkl'
            with open(lgb_path, 'wb') as f:
                pickle.dump(self.lgb_model, f)
            saved_files['lgb_model'] = lgb_path
            logger.info(f"  ✅ LightGBM model: {lgb_path.name}")
        
        if self.xgb_calibrated is not None:
            cal_xgb_path = output_dir / 'risk_classifier_xgb_calibrated.pkl'
            with open(cal_xgb_path, 'wb') as f:
                pickle.dump(self.xgb_calibrated, f)
            saved_files['xgb_calibrated'] = cal_xgb_path
            logger.info(f"  ✅ XGBoost calibrated: {cal_xgb_path.name}")
        
        if self.lgb_calibrated is not None:
            cal_lgb_path = output_dir / 'risk_classifier_lgb_calibrated.pkl'
            with open(cal_lgb_path, 'wb') as f:
                pickle.dump(self.lgb_calibrated, f)
            saved_files['lgb_calibrated'] = cal_lgb_path
            logger.info(f"  ✅ LightGBM calibrated: {cal_lgb_path.name}")
        
        if self.feature_importance is not None:
            imp_path = output_dir / 'risk_classifier_feature_importance.csv'
            self.feature_importance.to_csv(imp_path, index=False)
            saved_files['feature_importance'] = imp_path
            logger.info(f"  ✅ Feature importance: {imp_path.name}")
        
        if self.shap_values and 'shap_importance' in self.shap_values:
            shap_path = output_dir / 'risk_classifier_shap_importance.csv'
            self.shap_values['shap_importance'].to_csv(shap_path, index=False)
            saved_files['shap_importance'] = shap_path
            logger.info(f"  ✅ SHAP importance: {shap_path.name}")
        
        metadata = {
            'model_type': 'RiskClassifier',
            'version': '1.2.0',
            'created_at': datetime.now().isoformat(),
            'n_classes': self.n_classes,
            'class_names': self.config.class_names,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_calibrated': self.is_calibrated,
            'config': {
                'xgb_weight': self.config.xgb_weight,
                'lgb_weight': self.config.lgb_weight,
                'calibration_method': self.config.calibration_method,
                'cv_folds': self.config.cv_folds
            },
            'cv_results': self.cv_results,
            'optimal_thresholds': {str(k): v for k, v in self.optimal_thresholds.items()},
            'training_history': self.training_history
        }
        
        metadata_path = output_dir / 'risk_classifier_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = metadata_path
        logger.info(f"  ✅ Metadata: {metadata_path.name}")
        
        return saved_files
    
    @classmethod
    def load(cls, model_dir: Path) -> 'RiskClassifier':
        """Load saved model"""
        model_dir = Path(model_dir)
        classifier = cls()
        
        xgb_path = model_dir / 'risk_classifier_xgb.pkl'
        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                classifier.xgb_model = pickle.load(f)
        
        lgb_path = model_dir / 'risk_classifier_lgb.pkl'
        if lgb_path.exists():
            with open(lgb_path, 'rb') as f:
                classifier.lgb_model = pickle.load(f)
        
        cal_xgb_path = model_dir / 'risk_classifier_xgb_calibrated.pkl'
        if cal_xgb_path.exists():
            with open(cal_xgb_path, 'rb') as f:
                classifier.xgb_calibrated = pickle.load(f)
                classifier.is_calibrated = True
        
        cal_lgb_path = model_dir / 'risk_classifier_lgb_calibrated.pkl'
        if cal_lgb_path.exists():
            with open(cal_lgb_path, 'rb') as f:
                classifier.lgb_calibrated = pickle.load(f)
        
        metadata_path = model_dir / 'risk_classifier_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                classifier.feature_names = metadata.get('feature_names', [])
                classifier.n_classes = metadata.get('n_classes', 4)
                classifier.optimal_thresholds = {
                    int(k): v for k, v in metadata.get('optimal_thresholds', {}).items()
                }
        
        return classifier


class RiskClassifierRunner:
    """Runner for Risk Classifier training and evaluation"""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.classifier = None
        self.results = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load prepared ML data from PostgreSQL"""
        logger.info("Loading data from PostgreSQL...")
        
        from src.database.pg_data_service import PostgreSQLDataService
        db_service = PostgreSQLDataService()
        
        # In a real scenario, these would benefit from being materialized views or specific tables
        # For now, we'll assume we fetch from a 'training_data' table or construct it
        query = "SELECT * FROM training_data_risk_classifier"
        
        try:
            full_df = db_service.execute_query(query)
            
            if full_df.empty:
                logger.warning("No training data found in DB, checking for legacy parquet as fallback...")
                # Fallback implementation or error
                raise ValueError("No training data found in PostgreSQL table 'training_data_risk_classifier'")
                
            # Split data 
            # Assuming 'split' column exists: 'train', 'val', 'test'
            if 'split' not in full_df.columns:
                # Add random split if missing
                full_df['split'] = np.random.choice(['train', 'val', 'test'], size=len(full_df), p=[0.7, 0.15, 0.15])
            
            train_df = full_df[full_df['split'] == 'train'].drop(columns=['split'])
            val_df = full_df[full_df['split'] == 'val'].drop(columns=['split'])
            test_df = full_df[full_df['split'] == 'test'].drop(columns=['split'])
            
            logger.info(f"  Train: {len(train_df):,} rows")
            logger.info(f"  Val: {len(val_df):,} rows")
            logger.info(f"  Test: {len(test_df):,} rows")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to load data from DB: {e}")
            raise
    
    def run(self, run_cv: bool = True) -> Dict:
        """Run full training pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("a6on-i - RISK CLASSIFIER v1.2")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        train_df, val_df, test_df = self.load_data()
        
        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target']
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        self.classifier = RiskClassifier()
        self.classifier.train(X_train, y_train, X_val, y_val)
        self.classifier.calibrate(X_val, y_val)
        
        if run_cv:
            # Re-combine for CV
            X_train_orig = pd.concat([X_train, X_val])
            y_train_orig = pd.concat([y_train, y_val])
            self.results['cv'] = self.classifier.cross_validate(X_train_orig, y_train_orig)
        
        self.results['val'] = self.classifier.evaluate(X_val, y_val, 'validation')
        self.results['test'] = self.classifier.evaluate(X_test, y_test, 'test')
        self.results['thresholds'] = self.classifier.optimize_thresholds(X_val, y_val)
        self.results['shap'] = self.classifier.explain_shap(X_val)
        self.results['saved_files'] = self.classifier.save(self.output_dir / 'models')
        
        duration = (datetime.now() - start_time).total_seconds()
        self.results['duration'] = duration
        
        logger.info("\n" + "=" * 70)
        logger.info("RISK CLASSIFIER TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total duration: {duration:.2f} seconds")
        
        return self.results
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print("📊 PHASE 3.2 COMPLETE - RISK CLASSIFIER v1.2")
        print("=" * 70)
        
        if 'test' in self.results:
            test = self.results['test']
            print(f"\n🎯 TEST SET PERFORMANCE:")
            print(f"   Accuracy: {test['accuracy']:.4f}")
            print(f"   F1 (weighted): {test['f1_weighted']:.4f}")
            print(f"   F1 (macro): {test['f1_macro']:.4f}")
            if test.get('roc_auc'):
                print(f"   ROC-AUC: {test['roc_auc']:.4f}")
            
            print(f"\n   Per-class F1:")
            for class_name, metrics in test['per_class'].items():
                print(f"     {class_name}: {metrics['f1']:.3f} (n={metrics['support']})")
        
        if 'cv' in self.results:
            cv = self.results['cv']
            print(f"\n📈 CROSS-VALIDATION ({cv['n_folds']}-fold):")
            print(f"   Accuracy: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
            print(f"   F1 (weighted): {cv['f1_weighted_mean']:.4f} ± {cv['f1_weighted_std']:.4f}")
        
        if self.classifier and self.classifier.feature_importance is not None:
            print(f"\n🔧 TOP 5 FEATURES:")
            for _, row in self.classifier.feature_importance.head(5).iterrows():
                print(f"   {row['feature']}: {row.get('importance_pct', 0):.1f}%")
        
        if 'thresholds' in self.results:
            print(f"\n⚖️ OPTIMAL THRESHOLDS:")
            class_names = self.classifier.config.class_names
            for cls_idx, threshold in self.results['thresholds'].items():
                if cls_idx < len(class_names):
                    print(f"   {class_names[cls_idx]}: {threshold:.3f}")
        
        print(f"\n📁 Model saved to: {self.output_dir / 'models'}")
        print(f"⏱️ Total time: {self.results.get('duration', 0):.2f} seconds")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent.parent
    
    data_dir = project_root / 'data' / 'processed' / 'ml'
    output_dir = project_root / 'data' / 'processed' / 'ml'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run Phase 3.1 first: python src/run_ml_data_prep.py")
        return None
    
    runner = RiskClassifierRunner(data_dir, output_dir)
    runner.run(run_cv=True)
    runner.print_summary()
    
    return runner


if __name__ == '__main__':
    main()
