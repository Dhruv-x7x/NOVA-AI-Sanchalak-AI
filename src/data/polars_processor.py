"""
SANCHALAK AI - Polars High-Performance Data Processor
==============================================================
Fast data processing using Polars for large-scale clinical trial data.

Features:
- Lazy evaluation for memory efficiency
- Parallel processing
- Streaming for large files
- Pandas interoperability
- UPR aggregation acceleration
- Performance benchmarking

Author: Sanchalak AI Team
Date: 2026-01-24
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check Polars availability
POLARS_AVAILABLE = False

try:
    import polars as pl
    from polars import col, lit, when
    POLARS_AVAILABLE = True
    logger.info("✅ Polars available for high-performance processing")
except ImportError:
    logger.warning("⚠️ polars not installed - using pandas fallback")


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

@dataclass
class ProcessingStats:
    """Statistics from data processing operations."""
    operation: str
    rows_processed: int
    columns_processed: int
    duration_ms: float
    memory_mb: float
    engine: str  # "polars" or "pandas"
    speedup: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "rows": self.rows_processed,
            "columns": self.columns_processed,
            "duration_ms": round(self.duration_ms, 2),
            "memory_mb": round(self.memory_mb, 2),
            "engine": self.engine,
            "speedup": round(self.speedup, 2)
        }


def benchmark(func: Callable) -> Callable:
    """Decorator to benchmark processing functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        
        logger.debug(f"{func.__name__}: {duration:.2f}ms")
        return result
    return wrapper


# =============================================================================
# POLARS DATA PROCESSOR
# =============================================================================

class PolarsProcessor:
    """
    High-performance data processor using Polars.
    
    Provides:
    - Fast CSV/Parquet reading with lazy evaluation
    - Parallel aggregations
    - Memory-efficient streaming
    - Pandas interoperability
    - UPR-specific operations
    
    Usage:
        processor = PolarsProcessor()
        
        # Load and process data
        df = processor.read_csv("data.csv")
        result = processor.aggregate_by_site(df)
        
        # Convert to pandas if needed
        pandas_df = processor.to_pandas(result)
    """
    
    def __init__(self, n_threads: Optional[int] = None):
        """
        Initialize Polars processor.
        
        Args:
            n_threads: Number of threads for parallel processing (None = auto)
        """
        self.n_threads = n_threads
        self._use_polars = POLARS_AVAILABLE
        self._stats: List[ProcessingStats] = []
        
        if POLARS_AVAILABLE and n_threads:
            # Note: Polars uses all available cores by default
            os.environ["POLARS_MAX_THREADS"] = str(n_threads)
        
        logger.info(f"PolarsProcessor initialized (polars={self._use_polars})")
    
    @property
    def uses_polars(self) -> bool:
        """Check if using Polars."""
        return self._use_polars
    
    # =========================================================================
    # FILE I/O
    # =========================================================================
    
    @benchmark
    def read_csv(
        self,
        path: Union[str, Path],
        lazy: bool = True,
        columns: Optional[List[str]] = None,
        n_rows: Optional[int] = None,
        **kwargs
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """
        Read CSV file with optimal performance.
        
        Args:
            path: Path to CSV file
            lazy: Use lazy evaluation (Polars only)
            columns: Columns to select
            n_rows: Number of rows to read
            **kwargs: Additional arguments
            
        Returns:
            Polars DataFrame/LazyFrame or Pandas DataFrame
        """
        start = time.perf_counter()
        path = Path(path)
        
        if self._use_polars:
            if lazy:
                df = pl.scan_csv(path, n_rows=n_rows, **kwargs)
                if columns:
                    df = df.select(columns)
            else:
                df = pl.read_csv(path, columns=columns, n_rows=n_rows, **kwargs)
            engine = "polars"
        else:
            df = pd.read_csv(path, usecols=columns, nrows=n_rows, **kwargs)
            engine = "pandas"
        
        duration = (time.perf_counter() - start) * 1000
        rows = len(df) if hasattr(df, '__len__') else -1
        cols = len(df.columns) if hasattr(df, 'columns') else -1
        
        self._stats.append(ProcessingStats(
            operation="read_csv",
            rows_processed=rows,
            columns_processed=cols,
            duration_ms=duration,
            memory_mb=self._estimate_memory(df),
            engine=engine
        ))
        
        return df
    
    @benchmark
    def read_parquet(
        self,
        path: Union[str, Path],
        lazy: bool = True,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """
        Read Parquet file with optimal performance.
        
        Args:
            path: Path to Parquet file
            lazy: Use lazy evaluation (Polars only)
            columns: Columns to select
            **kwargs: Additional arguments
            
        Returns:
            Polars DataFrame/LazyFrame or Pandas DataFrame
        """
        start = time.perf_counter()
        path = Path(path)
        
        if self._use_polars:
            if lazy:
                df = pl.scan_parquet(path, **kwargs)
                if columns:
                    df = df.select(columns)
            else:
                df = pl.read_parquet(path, columns=columns, **kwargs)
            engine = "polars"
        else:
            df = pd.read_parquet(path, columns=columns, **kwargs)
            engine = "pandas"
        
        duration = (time.perf_counter() - start) * 1000
        rows = len(df) if hasattr(df, '__len__') and not lazy else -1
        cols = len(df.columns) if hasattr(df, 'columns') else -1
        
        self._stats.append(ProcessingStats(
            operation="read_parquet",
            rows_processed=rows,
            columns_processed=cols,
            duration_ms=duration,
            memory_mb=self._estimate_memory(df),
            engine=engine
        ))
        
        return df
    
    @benchmark
    def write_parquet(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        path: Union[str, Path],
        compression: str = "zstd"
    ):
        """Write DataFrame to Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._use_polars:
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            df.write_parquet(path, compression=compression)
        else:
            df.to_parquet(path, compression=compression)
    
    @benchmark
    def read_database(
        self,
        query: str,
        connection_string: str,
        lazy: bool = False
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Read from database with optimal performance.
        
        Args:
            query: SQL query
            connection_string: Database connection string
            lazy: Use lazy evaluation
            
        Returns:
            DataFrame
        """
        if self._use_polars:
            try:
                df = pl.read_database(query, connection_string)
                return df
            except Exception as e:
                logger.warning(f"Polars DB read failed: {e}, falling back to pandas")
        
        # Pandas fallback
        from sqlalchemy import create_engine
        engine = create_engine(connection_string)
        return pd.read_sql(query, engine)
    
    # =========================================================================
    # DATA TRANSFORMATION
    # =========================================================================
    
    @benchmark
    def collect(
        self,
        df: Union["pl.LazyFrame", "pl.DataFrame", pd.DataFrame]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Execute lazy query and collect results."""
        if self._use_polars and isinstance(df, pl.LazyFrame):
            return df.collect()
        return df
    
    @benchmark
    def to_pandas(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]
    ) -> pd.DataFrame:
        """Convert to Pandas DataFrame."""
        if self._use_polars:
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            return df.to_pandas()
        return df
    
    @benchmark
    def from_pandas(
        self,
        df: pd.DataFrame
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Convert from Pandas DataFrame."""
        if self._use_polars:
            return pl.from_pandas(df)
        return df
    
    @benchmark
    def filter(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        condition: Any
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """
        Filter DataFrame by condition.
        
        Args:
            df: Input DataFrame
            condition: Filter condition (Polars expr or pandas bool Series)
            
        Returns:
            Filtered DataFrame
        """
        if self._use_polars:
            return df.filter(condition)
        else:
            return df[condition]
    
    @benchmark
    def select(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        columns: List[str]
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """Select specific columns."""
        if self._use_polars:
            return df.select(columns)
        else:
            return df[columns]
    
    @benchmark
    def groupby_agg(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        group_cols: List[str],
        agg_dict: Dict[str, Union[str, List[str]]]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Group by and aggregate.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_dict: Aggregation dict {column: agg_func or [agg_funcs]}
            
        Returns:
            Aggregated DataFrame
        """
        start = time.perf_counter()
        
        if self._use_polars:
            # Build Polars aggregation expressions
            agg_exprs = []
            for col_name, agg_funcs in agg_dict.items():
                funcs = [agg_funcs] if isinstance(agg_funcs, str) else agg_funcs
                for func in funcs:
                    if func == "mean":
                        agg_exprs.append(col(col_name).mean().alias(f"{col_name}_mean"))
                    elif func == "sum":
                        agg_exprs.append(col(col_name).sum().alias(f"{col_name}_sum"))
                    elif func == "count":
                        agg_exprs.append(col(col_name).count().alias(f"{col_name}_count"))
                    elif func == "min":
                        agg_exprs.append(col(col_name).min().alias(f"{col_name}_min"))
                    elif func == "max":
                        agg_exprs.append(col(col_name).max().alias(f"{col_name}_max"))
                    elif func == "std":
                        agg_exprs.append(col(col_name).std().alias(f"{col_name}_std"))
                    elif func == "first":
                        agg_exprs.append(col(col_name).first().alias(f"{col_name}_first"))
                    elif func == "last":
                        agg_exprs.append(col(col_name).last().alias(f"{col_name}_last"))
            
            result = df.group_by(group_cols).agg(agg_exprs)
            
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            
            engine = "polars"
        else:
            # Pandas aggregation
            result = df.groupby(group_cols, as_index=False).agg(agg_dict)
            # Flatten column names
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = ['_'.join(col).strip('_') for col in result.columns.values]
            engine = "pandas"
        
        duration = (time.perf_counter() - start) * 1000
        self._stats.append(ProcessingStats(
            operation="groupby_agg",
            rows_processed=len(df) if hasattr(df, '__len__') else -1,
            columns_processed=len(group_cols) + len(agg_dict),
            duration_ms=duration,
            memory_mb=self._estimate_memory(result),
            engine=engine
        ))
        
        return result
    
    @benchmark
    def join(
        self,
        left: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        right: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        on: Union[str, List[str]],
        how: str = "left"
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """
        Join two DataFrames.
        
        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Join key(s)
            how: Join type (left, right, inner, outer)
            
        Returns:
            Joined DataFrame
        """
        if self._use_polars:
            return left.join(right, on=on, how=how)
        else:
            return pd.merge(left, right, on=on, how=how)
    
    @benchmark
    def sort(
        self,
        df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        by: Union[str, List[str]],
        descending: bool = False
    ) -> Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]:
        """Sort DataFrame by columns."""
        if self._use_polars:
            return df.sort(by, descending=descending)
        else:
            return df.sort_values(by, ascending=not descending)
    
    # =========================================================================
    # UPR-SPECIFIC OPERATIONS
    # =========================================================================
    
    @benchmark
    def calculate_site_metrics(
        self,
        upr_df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Calculate site-level metrics from UPR data.
        
        Args:
            upr_df: Unified Patient Record DataFrame
            
        Returns:
            Site metrics DataFrame
        """
        if self._use_polars:
            # Ensure we have a DataFrame
            if isinstance(upr_df, pl.LazyFrame):
                upr_df = upr_df.collect()
            
            # Define aggregations
            result = upr_df.group_by(["study_id", "site_id"]).agg([
                pl.count().alias("patient_count"),
                col("dqi_score").mean().alias("mean_dqi"),
                col("dqi_score").std().alias("std_dqi"),
                col("dqi_score").min().alias("min_dqi"),
                col("dqi_score").max().alias("max_dqi"),
                col("total_issues").sum().alias("total_issues"),
                col("total_open_queries").sum().alias("total_open_queries"),
                # Clean patient rates
                (col("clean_status_tier").eq("clean").sum() / pl.count() * 100).alias("clean_rate"),
            ])
            
            return result
        else:
            # Pandas implementation
            result = upr_df.groupby(["study_id", "site_id"]).agg({
                "patient_key": "count",
                "dqi_score": ["mean", "std", "min", "max"],
                "total_issues": "sum",
                "total_open_queries": "sum"
            }).reset_index()
            
            # Flatten column names
            result.columns = [
                '_'.join(col).strip('_') if isinstance(col, tuple) else col
                for col in result.columns.values
            ]
            
            # Calculate clean rate
            clean_counts = upr_df.groupby(["study_id", "site_id"])["clean_status_tier"].apply(
                lambda x: (x == "clean").sum() / len(x) * 100 if len(x) > 0 else 0
            ).reset_index(name="clean_rate")
            
            result = result.merge(clean_counts, on=["study_id", "site_id"], how="left")
            
            return result
    
    @benchmark
    def calculate_study_metrics(
        self,
        upr_df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Calculate study-level metrics from UPR data.
        
        Args:
            upr_df: Unified Patient Record DataFrame
            
        Returns:
            Study metrics DataFrame
        """
        if self._use_polars:
            if isinstance(upr_df, pl.LazyFrame):
                upr_df = upr_df.collect()
            
            result = upr_df.group_by("study_id").agg([
                pl.n_unique("site_id").alias("site_count"),
                pl.count().alias("patient_count"),
                col("dqi_score").mean().alias("mean_dqi"),
                col("dqi_score").median().alias("median_dqi"),
                col("total_issues").sum().alias("total_issues"),
                col("total_open_queries").sum().alias("total_open_queries"),
                (col("clean_status_tier").eq("clean").sum() / pl.count() * 100).alias("clean_rate"),
            ])
            
            return result
        else:
            result = upr_df.groupby("study_id").agg({
                "site_id": "nunique",
                "patient_key": "count",
                "dqi_score": ["mean", "median"],
                "total_issues": "sum",
                "total_open_queries": "sum"
            }).reset_index()
            
            result.columns = [
                '_'.join(col).strip('_') if isinstance(col, tuple) else col
                for col in result.columns.values
            ]
            
            return result
    
    @benchmark
    def calculate_dqi_components(
        self,
        upr_df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame]
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Calculate DQI component scores.
        
        Args:
            upr_df: Unified Patient Record DataFrame
            
        Returns:
            DataFrame with DQI components
        """
        if self._use_polars:
            if isinstance(upr_df, pl.LazyFrame):
                upr_df = upr_df.collect()
            
            # Define DQI component weights
            result = upr_df.with_columns([
                # Safety Score (25%)
                (100 - col("sae_count").fill_null(0) * 10).clip(0, 100).alias("safety_score"),
                # Query Score (20%)
                (100 - col("total_open_queries").fill_null(0) * 2).clip(0, 100).alias("query_score"),
                # Completeness Score (15%)
                (100 - col("missing_pages_count").fill_null(0) * 5).clip(0, 100).alias("completeness_score"),
                # SDV Score (8%)
                (col("sdv_complete_pct").fill_null(100)).alias("sdv_score"),
                # Signature Score (5%)
                (100 - col("crfs_never_signed").fill_null(0) * 3).clip(0, 100).alias("signature_score"),
            ])
            
            # Calculate weighted DQI
            result = result.with_columns([
                (
                    col("safety_score") * 0.25 +
                    col("query_score") * 0.20 +
                    col("completeness_score") * 0.15 +
                    col("sdv_score") * 0.08 +
                    col("signature_score") * 0.05 +
                    32  # Base for remaining components
                ).alias("calculated_dqi")
            ])
            
            return result
        else:
            df = upr_df.copy()
            
            df["safety_score"] = (100 - df["sae_count"].fillna(0) * 10).clip(0, 100)
            df["query_score"] = (100 - df["total_open_queries"].fillna(0) * 2).clip(0, 100)
            df["completeness_score"] = (100 - df["missing_pages_count"].fillna(0) * 5).clip(0, 100)
            df["sdv_score"] = df["sdv_complete_pct"].fillna(100)
            df["signature_score"] = (100 - df["crfs_never_signed"].fillna(0) * 3).clip(0, 100)
            
            df["calculated_dqi"] = (
                df["safety_score"] * 0.25 +
                df["query_score"] * 0.20 +
                df["completeness_score"] * 0.15 +
                df["sdv_score"] * 0.08 +
                df["signature_score"] * 0.05 +
                32
            )
            
            return df
    
    @benchmark
    def identify_at_risk_patients(
        self,
        upr_df: Union["pl.DataFrame", "pl.LazyFrame", pd.DataFrame],
        dqi_threshold: float = 70,
        query_threshold: int = 5
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """
        Identify patients at risk based on multiple criteria.
        
        Args:
            upr_df: Unified Patient Record DataFrame
            dqi_threshold: DQI score threshold
            query_threshold: Open query count threshold
            
        Returns:
            At-risk patients DataFrame
        """
        if self._use_polars:
            if isinstance(upr_df, pl.LazyFrame):
                upr_df = upr_df.collect()
            
            # Filter at-risk patients
            at_risk = upr_df.filter(
                (col("dqi_score") < dqi_threshold) |
                (col("total_open_queries") > query_threshold) |
                (col("sae_count") > 0)
            )
            
            # Add risk level
            at_risk = at_risk.with_columns([
                when(col("sae_count") > 0)
                .then(lit("Critical"))
                .when(col("dqi_score") < 50)
                .then(lit("High"))
                .when(col("dqi_score") < 70)
                .then(lit("Medium"))
                .otherwise(lit("Low"))
                .alias("risk_level")
            ])
            
            return at_risk.sort("dqi_score")
        else:
            mask = (
                (upr_df["dqi_score"] < dqi_threshold) |
                (upr_df["total_open_queries"] > query_threshold) |
                (upr_df["sae_count"] > 0)
            )
            at_risk = upr_df[mask].copy()
            
            conditions = [
                at_risk["sae_count"] > 0,
                at_risk["dqi_score"] < 50,
                at_risk["dqi_score"] < 70
            ]
            choices = ["Critical", "High", "Medium"]
            at_risk["risk_level"] = np.select(conditions, choices, default="Low")
            
            return at_risk.sort_values("dqi_score")
    
    # =========================================================================
    # STREAMING & LARGE FILE PROCESSING
    # =========================================================================
    
    def stream_csv(
        self,
        path: Union[str, Path],
        chunk_size: int = 100000,
        process_func: Optional[Callable] = None
    ) -> List[Any]:
        """
        Stream large CSV file in chunks.
        
        Args:
            path: Path to CSV file
            chunk_size: Rows per chunk
            process_func: Function to apply to each chunk
            
        Returns:
            List of processed results
        """
        path = Path(path)
        results = []
        
        if self._use_polars:
            # Polars streaming
            reader = pl.read_csv_batched(path, batch_size=chunk_size)
            
            while True:
                batch = reader.next_batches(1)
                if not batch:
                    break
                
                chunk = batch[0]
                if process_func:
                    result = process_func(chunk)
                    results.append(result)
                else:
                    results.append(chunk)
        else:
            # Pandas chunked reading
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                if process_func:
                    result = process_func(chunk)
                    results.append(result)
                else:
                    results.append(chunk)
        
        return results
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _estimate_memory(self, df: Any) -> float:
        """Estimate memory usage in MB."""
        try:
            if self._use_polars and hasattr(df, 'estimated_size'):
                return df.estimated_size() / (1024 * 1024)
            elif hasattr(df, 'memory_usage'):
                return df.memory_usage(deep=True).sum() / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get processing statistics."""
        return [s.to_dict() for s in self._stats]
    
    def clear_stats(self):
        """Clear processing statistics."""
        self._stats.clear()
    
    def benchmark_comparison(
        self,
        df: pd.DataFrame,
        operation: str = "groupby"
    ) -> Dict[str, Any]:
        """
        Benchmark Polars vs Pandas for an operation.
        
        Args:
            df: Test DataFrame
            operation: Operation to benchmark
            
        Returns:
            Benchmark results
        """
        results = {
            "operation": operation,
            "rows": len(df),
            "pandas": {},
            "polars": {}
        }
        
        if operation == "groupby":
            # Pandas timing
            start = time.perf_counter()
            _ = df.groupby("study_id").agg({"dqi_score": "mean"})
            results["pandas"]["duration_ms"] = (time.perf_counter() - start) * 1000
            
            # Polars timing
            if self._use_polars:
                pl_df = pl.from_pandas(df)
                start = time.perf_counter()
                _ = pl_df.group_by("study_id").agg(col("dqi_score").mean())
                results["polars"]["duration_ms"] = (time.perf_counter() - start) * 1000
                
                results["speedup"] = results["pandas"]["duration_ms"] / results["polars"]["duration_ms"]
        
        elif operation == "filter":
            # Pandas timing
            start = time.perf_counter()
            _ = df[df["dqi_score"] < 80]
            results["pandas"]["duration_ms"] = (time.perf_counter() - start) * 1000
            
            # Polars timing
            if self._use_polars:
                pl_df = pl.from_pandas(df)
                start = time.perf_counter()
                _ = pl_df.filter(col("dqi_score") < 80)
                results["polars"]["duration_ms"] = (time.perf_counter() - start) * 1000
                
                results["speedup"] = results["pandas"]["duration_ms"] / results["polars"]["duration_ms"]
        
        return results


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_processor: Optional[PolarsProcessor] = None


def get_polars_processor() -> PolarsProcessor:
    """Get singleton Polars processor instance."""
    global _processor
    if _processor is None:
        _processor = PolarsProcessor()
    return _processor


def reset_polars_processor():
    """Reset the singleton (for testing)."""
    global _processor
    _processor = None


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    """Demo the Polars processor."""
    print("=" * 70)
    print("SANCHALAK AI - POLARS HIGH-PERFORMANCE PROCESSING DEMO")
    print("=" * 70)
    
    # Initialize processor
    processor = PolarsProcessor()
    
    print(f"\n✅ Processor initialized (polars={processor.uses_polars})")
    
    # Create sample data
    print("\n" + "-" * 50)
    print("Creating sample UPR data...")
    print("-" * 50)
    
    np.random.seed(42)
    n_rows = 10000
    
    sample_data = pd.DataFrame({
        "patient_key": [f"PT_{i:05d}" for i in range(n_rows)],
        "study_id": np.random.choice(["STUDY_001", "STUDY_002", "STUDY_003"], n_rows),
        "site_id": np.random.choice([f"SITE_{i:03d}" for i in range(50)], n_rows),
        "dqi_score": np.random.uniform(60, 100, n_rows),
        "total_issues": np.random.poisson(5, n_rows),
        "total_open_queries": np.random.poisson(3, n_rows),
        "missing_pages_count": np.random.poisson(2, n_rows),
        "sae_count": np.random.poisson(0.1, n_rows),
        "sdv_complete_pct": np.random.uniform(80, 100, n_rows),
        "crfs_never_signed": np.random.poisson(1, n_rows),
        "clean_status_tier": np.random.choice(["clean", "tier1", "tier2", "pending"], n_rows)
    })
    
    print(f"Created {len(sample_data):,} rows")
    
    # Convert to Polars if available
    if processor.uses_polars:
        df = processor.from_pandas(sample_data)
        print(f"Converted to Polars DataFrame")
    else:
        df = sample_data
    
    # Demo: Site metrics
    print("\n" + "-" * 50)
    print("DEMO: Calculate Site Metrics")
    print("-" * 50)
    
    site_metrics = processor.calculate_site_metrics(df)
    result = processor.to_pandas(site_metrics) if processor.uses_polars else site_metrics
    print(f"Calculated metrics for {len(result)} sites")
    print(result.head())
    
    # Demo: Study metrics
    print("\n" + "-" * 50)
    print("DEMO: Calculate Study Metrics")
    print("-" * 50)
    
    study_metrics = processor.calculate_study_metrics(df)
    result = processor.to_pandas(study_metrics) if processor.uses_polars else study_metrics
    print(result)
    
    # Demo: At-risk patients
    print("\n" + "-" * 50)
    print("DEMO: Identify At-Risk Patients")
    print("-" * 50)
    
    at_risk = processor.identify_at_risk_patients(df)
    result = processor.to_pandas(at_risk) if processor.uses_polars else at_risk
    print(f"Found {len(result)} at-risk patients")
    print(result[["patient_key", "dqi_score", "total_open_queries", "sae_count", "risk_level"]].head(10))
    
    # Benchmark comparison
    if processor.uses_polars:
        print("\n" + "-" * 50)
        print("BENCHMARK: Polars vs Pandas")
        print("-" * 50)
        
        benchmark = processor.benchmark_comparison(sample_data, "groupby")
        print(f"GroupBy operation on {benchmark['rows']:,} rows:")
        print(f"  Pandas: {benchmark['pandas']['duration_ms']:.2f}ms")
        print(f"  Polars: {benchmark['polars']['duration_ms']:.2f}ms")
        print(f"  Speedup: {benchmark.get('speedup', 1):.2f}x")
        
        benchmark = processor.benchmark_comparison(sample_data, "filter")
        print(f"\nFilter operation on {benchmark['rows']:,} rows:")
        print(f"  Pandas: {benchmark['pandas']['duration_ms']:.2f}ms")
        print(f"  Polars: {benchmark['polars']['duration_ms']:.2f}ms")
        print(f"  Speedup: {benchmark.get('speedup', 1):.2f}x")
    
    # Print stats
    print("\n" + "=" * 70)
    print("PROCESSING STATISTICS")
    print("=" * 70)
    for stat in processor.get_stats():
        print(f"  {stat['operation']}: {stat['duration_ms']:.2f}ms ({stat['engine']})")
    
    print("\n✅ Polars processing demo complete!")
    return processor


if __name__ == "__main__":
    main()
