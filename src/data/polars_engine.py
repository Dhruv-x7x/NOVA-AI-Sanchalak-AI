"""
a6on-i - Polars High-Performance Data Engine
=======================================================
Replace critical Pandas paths with Polars for 5-10x speedup:
- UPR aggregation
- DQI calculation
- Batch predictions
- Large dataset operations

Per riyaz.md Section 27: Polars Integration for Large Data
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import os

logger = logging.getLogger(__name__)

# Try to import Polars, fall back gracefully
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    logger.warning("Polars not installed. Using Pandas fallback. Install with: pip install polars")

# Pandas fallback
import pandas as pd
import numpy as np


class PolarsEngine:
    """
    High-performance data engine using Polars.
    
    Provides 5-10x speedup over Pandas for:
    - Large dataset aggregations
    - DQI calculations
    - Batch ML predictions
    - Data transformations
    
    Falls back to Pandas if Polars is not available.
    """
    
    def __init__(self, use_polars: bool = True):
        """
        Initialize Polars engine.
        
        Args:
            use_polars: Force Polars usage (will error if not available)
        """
        self.use_polars = use_polars and POLARS_AVAILABLE
        if use_polars and not POLARS_AVAILABLE:
            logger.warning("Polars requested but not available. Using Pandas.")
        
        logger.info(f"PolarsEngine initialized (using_polars={self.use_polars})")
    
    def read_parquet(self, path: Union[str, Path]) -> Union["pl.DataFrame", pd.DataFrame]:
        """Read parquet file with optimal engine."""
        path = Path(path)
        
        if self.use_polars:
            return pl.read_parquet(path)
        else:
            return pd.read_parquet(path)
    
    def read_csv(self, path: Union[str, Path], **kwargs) -> Union["pl.DataFrame", pd.DataFrame]:
        """Read CSV file with optimal engine."""
        path = Path(path)
        
        if self.use_polars:
            return pl.read_csv(path, **kwargs)
        else:
            return pd.read_csv(path, **kwargs)
    
    def to_pandas(self, df: Any) -> pd.DataFrame:
        """Convert any dataframe to Pandas."""
        if self.use_polars and isinstance(df, pl.DataFrame):
            return df.to_pandas()
        return df
    
    def to_polars(self, df: pd.DataFrame) -> Any:
        """Convert Pandas dataframe to Polars."""
        if self.use_polars:
            return pl.from_pandas(df)
        return df
    
    # =========================================================================
    # DQI CALCULATION (High-Performance)
    # =========================================================================
    
    def calculate_dqi_batch(
        self, 
        patient_data: Union["pl.DataFrame", pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Calculate DQI scores for all patients in batch.
        
        5-10x faster than row-by-row Pandas calculation.
        
        Args:
            patient_data: DataFrame with patient issue columns
            weights: Optional custom weights for DQI components
            
        Returns:
            DataFrame with patient_key and dqi_score columns
        """
        # Default weights
        if weights is None:
            weights = {
                "open_queries": 0.25,
                "signature_gaps": 0.20,
                "sdv_incomplete": 0.15,
                "missing_pages": 0.15,
                "missing_visits": 0.10,
                "sae_pending": 0.10,
                "coding_pending": 0.05
            }
        
        if self.use_polars:
            return self._calculate_dqi_polars(patient_data, weights)
        else:
            return self._calculate_dqi_pandas(patient_data, weights)
    
    def _calculate_dqi_polars(self, df: "pl.DataFrame", weights: Dict[str, float]) -> "pl.DataFrame":
        """Polars-optimized DQI calculation."""
        # Build DQI expression
        dqi_expr = pl.lit(100.0)
        
        for col, weight in weights.items():
            if col in df.columns:
                # Deduct points based on issue counts (capped at 100% deduction per category)
                penalty = (
                    pl.when(pl.col(col) > 0)
                    .then(pl.min_horizontal(pl.col(col) * 5.0, pl.lit(100.0)) * weight)
                    .otherwise(0.0)
                )
                dqi_expr = dqi_expr - penalty
        
        # Ensure DQI is between 0 and 100
        dqi_expr = pl.max_horizontal(dqi_expr, pl.lit(0.0))
        dqi_expr = pl.min_horizontal(dqi_expr, pl.lit(100.0))
        
        return df.select([
            pl.col("patient_key"),
            dqi_expr.alias("dqi_score")
        ])
    
    def _calculate_dqi_pandas(self, df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Pandas fallback for DQI calculation."""
        dqi_scores = pd.Series(100.0, index=df.index)
        
        for col, weight in weights.items():
            if col in df.columns:
                penalty = np.minimum(df[col] * 5.0, 100.0) * weight
                penalty = penalty.fillna(0)
                dqi_scores = dqi_scores - penalty
        
        dqi_scores = dqi_scores.clip(0, 100)
        
        return pd.DataFrame({
            "patient_key": df["patient_key"],
            "dqi_score": dqi_scores
        })
    
    # =========================================================================
    # UPR AGGREGATION (High-Performance)
    # =========================================================================
    
    def aggregate_upr_by_site(
        self, 
        upr_data: Union["pl.DataFrame", pd.DataFrame]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Aggregate patient data by site with all metrics.
        
        Args:
            upr_data: Patient-level DataFrame
            
        Returns:
            Site-level aggregated DataFrame
        """
        if self.use_polars:
            return self._aggregate_upr_polars(upr_data)
        else:
            return self._aggregate_upr_pandas(upr_data)
    
    def _aggregate_upr_polars(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Polars-optimized UPR aggregation."""
        # Define aggregation expressions
        agg_exprs = [
            pl.count().alias("patient_count"),
            pl.col("dqi_score").mean().alias("avg_dqi"),
            pl.col("dqi_score").min().alias("min_dqi"),
            pl.col("dqi_score").max().alias("max_dqi"),
        ]
        
        # Add sum aggregations for issue columns if they exist
        issue_cols = [
            "open_queries", "signature_gaps", "sdv_incomplete", 
            "missing_pages", "missing_visits", "sae_pending"
        ]
        
        for col in issue_cols:
            if col in df.columns:
                agg_exprs.append(pl.col(col).sum().alias(f"total_{col}"))
                agg_exprs.append(
                    (pl.col(col) > 0).sum().alias(f"patients_with_{col}")
                )
        
        return df.group_by(["study_id", "site_id"]).agg(agg_exprs)
    
    def _aggregate_upr_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pandas fallback for UPR aggregation."""
        agg_dict = {
            "patient_key": "count",
            "dqi_score": ["mean", "min", "max"]
        }
        
        issue_cols = [
            "open_queries", "signature_gaps", "sdv_incomplete",
            "missing_pages", "missing_visits", "sae_pending"
        ]
        
        for col in issue_cols:
            if col in df.columns:
                agg_dict[col] = "sum"
        
        result = df.groupby(["study_id", "site_id"]).agg(agg_dict)
        result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
        return result.reset_index()
    
    # =========================================================================
    # BATCH PREDICTIONS (High-Performance)
    # =========================================================================
    
    def prepare_features_for_prediction(
        self, 
        data: Union["pl.DataFrame", pd.DataFrame],
        feature_cols: List[str],
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Prepare feature matrix for ML predictions.
        
        Args:
            data: Source DataFrame
            feature_cols: List of feature column names
            fill_value: Value to fill missing data
            
        Returns:
            NumPy array ready for model prediction
        """
        if self.use_polars:
            # Select and fill nulls
            features_df = data.select([
                pl.col(col).fill_null(fill_value) 
                for col in feature_cols 
                if col in data.columns
            ])
            return features_df.to_numpy()
        else:
            # Pandas version
            features_df = data[feature_cols].fillna(fill_value)
            return features_df.values
    
    def add_predictions_to_dataframe(
        self,
        data: Union["pl.DataFrame", pd.DataFrame],
        predictions: np.ndarray,
        column_name: str = "prediction"
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Add prediction results back to dataframe efficiently.
        
        Args:
            data: Original DataFrame
            predictions: Array of predictions
            column_name: Name for prediction column
            
        Returns:
            DataFrame with predictions added
        """
        if self.use_polars:
            return data.with_columns(
                pl.Series(name=column_name, values=predictions)
            )
        else:
            data = data.copy()
            data[column_name] = predictions
            return data
    
    # =========================================================================
    # DATA QUALITY CHECKS (High-Performance)
    # =========================================================================
    
    def find_data_quality_issues(
        self, 
        data: Union["pl.DataFrame", pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Returns:
            Dictionary with quality metrics and issues found
        """
        if self.use_polars:
            return self._quality_check_polars(data)
        else:
            return self._quality_check_pandas(data)
    
    def _quality_check_polars(self, df: "pl.DataFrame") -> Dict[str, Any]:
        """Polars-optimized data quality check."""
        total_rows = df.height
        
        # Null counts per column
        null_counts = {
            col: df.select(pl.col(col).null_count())[0, 0]
            for col in df.columns
        }
        
        # Duplicate check on patient_key if exists
        duplicates = 0
        if "patient_key" in df.columns:
            duplicates = total_rows - df.select(pl.col("patient_key")).unique().height
        
        # Value range issues
        range_issues = []
        if "dqi_score" in df.columns:
            out_of_range = df.filter(
                (pl.col("dqi_score") < 0) | (pl.col("dqi_score") > 100)
            ).height
            if out_of_range > 0:
                range_issues.append(f"dqi_score: {out_of_range} values out of 0-100 range")
        
        return {
            "total_rows": total_rows,
            "null_counts": null_counts,
            "total_nulls": sum(null_counts.values()),
            "duplicate_keys": duplicates,
            "range_issues": range_issues,
            "completeness_pct": (1 - sum(null_counts.values()) / (total_rows * len(df.columns))) * 100
        }
    
    def _quality_check_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pandas fallback for data quality check."""
        total_rows = len(df)
        
        null_counts = df.isnull().sum().to_dict()
        
        duplicates = 0
        if "patient_key" in df.columns:
            duplicates = df["patient_key"].duplicated().sum()
        
        range_issues = []
        if "dqi_score" in df.columns:
            out_of_range = ((df["dqi_score"] < 0) | (df["dqi_score"] > 100)).sum()
            if out_of_range > 0:
                range_issues.append(f"dqi_score: {out_of_range} values out of 0-100 range")
        
        return {
            "total_rows": total_rows,
            "null_counts": null_counts,
            "total_nulls": sum(null_counts.values()),
            "duplicate_keys": duplicates,
            "range_issues": range_issues,
            "completeness_pct": (1 - sum(null_counts.values()) / (total_rows * len(df.columns))) * 100
        }
    
    # =========================================================================
    # FILTERING AND TRANSFORMATION
    # =========================================================================
    
    def filter_by_study(
        self,
        data: Union["pl.DataFrame", pd.DataFrame],
        study_id: str
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Filter data to single study efficiently."""
        if self.use_polars:
            return data.filter(pl.col("study_id") == study_id)
        else:
            return data[data["study_id"] == study_id]
    
    def filter_high_risk_patients(
        self,
        data: Union["pl.DataFrame", pd.DataFrame],
        dqi_threshold: float = 60.0
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Filter to high-risk patients (low DQI)."""
        if self.use_polars:
            return data.filter(pl.col("dqi_score") < dqi_threshold)
        else:
            return data[data["dqi_score"] < dqi_threshold]
    
    def get_statistics(
        self,
        data: Union["pl.DataFrame", pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get comprehensive statistics for numeric columns."""
        if self.use_polars:
            numeric_cols = [
                col for col, dtype in zip(data.columns, data.dtypes)
                if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            
            stats = {}
            for col in numeric_cols:
                col_stats = data.select([
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).median().alias("median")
                ])
                stats[col] = col_stats.to_dicts()[0]
            
            return stats
        else:
            return data.describe().to_dict()


# Singleton instance
_polars_engine: Optional[PolarsEngine] = None


def get_polars_engine() -> PolarsEngine:
    """Get singleton Polars engine instance."""
    global _polars_engine
    if _polars_engine is None:
        _polars_engine = PolarsEngine()
    return _polars_engine


# Convenience function
def calculate_dqi_fast(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Quick function to calculate DQI using best available engine."""
    engine = get_polars_engine()
    result = engine.calculate_dqi_batch(patient_df)
    return engine.to_pandas(result)
