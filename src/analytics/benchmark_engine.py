"""
SANCHALAK AI - Benchmark Engine v1.2 (FIXED)
Phase 2.5: Comparative Analytics and Peer Benchmarking

FIXES v1.2:
- Performance tiers now use percentile-based thresholds (bell curve)
- Small sites (<5 patients) flagged and weighted differently
- Study tier thresholds adjusted
- Added confidence penalty for small sample sizes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# PostgreSQL writer
from src.database.pg_writer import get_pg_writer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark Engine Configuration"""
    
    site_size_bins: List[Tuple[int, int, str]] = field(default_factory=lambda: [
        (1, 5, 'Small'),
        (6, 15, 'Medium'),
        (16, 30, 'Large'),
        (31, 9999, 'Very Large')
    ])
    
    outlier_zscore_threshold: float = 2.5
    
    # Minimum patients for reliable scoring
    min_patients_for_full_score: int = 5
    
    # Percentile-based tier thresholds (creates bell curve)
    tier_percentiles: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Exceptional': (90, 100),      # Top 10%
        'Strong': (70, 90),            # Next 20%
        'Average': (30, 70),           # Middle 40%
        'Below Average': (10, 30),     # Next 20%
        'Needs Improvement': (0, 10)   # Bottom 10%
    })


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

class BenchmarkEngine:
    """Benchmark Engine for comparative analytics (Optimized + Fixed)"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        
    def _get_site_size_category(self, patient_count: int) -> str:
        """Get site size category"""
        for min_size, max_size, category in self.config.site_size_bins:
            if min_size <= patient_count <= max_size:
                return category
        return 'Unknown'
    
    def prepare_site_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare site-level data for benchmarking"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING SITE-LEVEL DATA")
        logger.info("=" * 70)
        
        # Build aggregation dict dynamically
        agg_dict = {'patient_key': 'count'}
        
        # Add available metrics
        optional_cols = {
            'dqi_score': 'mean',
            'cascade_impact_score': ['mean', 'sum'],
            'total_queries': 'sum',
            'crfs_require_verification_sdv': 'sum',
            'crfs_never_signed': 'sum',
            'tier1_clean': 'sum',
            'tier2_clean': 'sum',
            'dblock_tier1_ready': 'sum',
            'dblock_days_to_ready': 'mean'
        }
        
        for col, agg in optional_cols.items():
            if col in df.columns:
                agg_dict[col] = agg
        
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        new_cols = []
        for col in site_df.columns:
            if isinstance(col, tuple):
                new_cols.append('_'.join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        site_df.columns = new_cols
        
        # Rename columns
        rename_map = {
            'patient_key_count': 'patient_count',
            'patient_key': 'patient_count',
            'dqi_score_mean': 'dqi_mean',
            'cascade_impact_score_mean': 'cascade_mean',
            'cascade_impact_score_sum': 'cascade_total',
            'total_queries_sum': 'total_queries',
            'crfs_require_verification_sdv_sum': 'sdv_pending',
            'crfs_never_signed_sum': 'unsigned_crfs',
            'tier1_clean_sum': 'tier1_clean_count',
            'tier2_clean_sum': 'tier2_clean_count',
            'dblock_tier1_ready_sum': 'ready_count',
            'dblock_days_to_ready_mean': 'avg_days_to_ready'
        }
        site_df = site_df.rename(columns={k: v for k, v in rename_map.items() if k in site_df.columns})
        
        # Calculate rates
        if 'patient_count' in site_df.columns:
            pc = site_df['patient_count']
            
            if 'tier1_clean_count' in site_df.columns:
                site_df['tier1_clean_rate'] = (site_df['tier1_clean_count'] / pc * 100).round(1)
            if 'tier2_clean_count' in site_df.columns:
                site_df['tier2_clean_rate'] = (site_df['tier2_clean_count'] / pc * 100).round(1)
            if 'ready_count' in site_df.columns:
                site_df['ready_rate'] = (site_df['ready_count'] / pc * 100).round(1)
            if 'total_queries' in site_df.columns:
                site_df['queries_per_patient'] = (site_df['total_queries'] / pc).round(2)
            if 'cascade_total' in site_df.columns:
                site_df['cascade_density'] = (site_df['cascade_total'] / pc).round(2)
        
        # Flag small sites
        site_df['is_small_site'] = site_df['patient_count'] < self.config.min_patients_for_full_score
        
        # Add region
        if 'region' in df.columns:
            region_map = df.groupby(['study_id', 'site_id'])['region'].first().reset_index()
            site_df = site_df.merge(region_map, on=['study_id', 'site_id'], how='left')
        
        # Add country
        if 'country' in df.columns:
            country_map = df.groupby(['study_id', 'site_id'])['country'].first().reset_index()
            site_df = site_df.merge(country_map, on=['study_id', 'site_id'], how='left')
        
        # Site size category
        site_df['site_size'] = site_df['patient_count'].apply(self._get_site_size_category)
        
        small_count = site_df['is_small_site'].sum()
        logger.info(f"  Sites prepared: {len(site_df):,}")
        logger.info(f"  Small sites (<{self.config.min_patients_for_full_score} patients): {small_count:,} ({small_count/len(site_df)*100:.1f}%)")
        logger.info(f"  Site sizes: {site_df['site_size'].value_counts().to_dict()}")
        
        return site_df
    
    def benchmark_sites_vs_peers(self, site_df: pd.DataFrame) -> pd.DataFrame:
        """Benchmark sites vs peers (OPTIMIZED - vectorized)"""
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING SITES VS PEERS")
        logger.info("=" * 70)
        
        result = site_df.copy()
        
        # Peer groups to benchmark against
        peer_groups = [
            ('study', ['study_id']),
            ('region', ['region']),
            ('size', ['site_size']),
        ]
        
        metrics = ['dqi_mean', 'tier1_clean_rate', 'tier2_clean_rate', 
                   'ready_rate', 'queries_per_patient', 'avg_days_to_ready']
        
        for peer_name, peer_cols in peer_groups:
            valid_cols = [c for c in peer_cols if c in result.columns]
            if not valid_cols:
                continue
            
            logger.info(f"  Benchmarking by {peer_name}...")
            
            for metric in metrics:
                if metric not in result.columns:
                    continue
                
                # Calculate peer statistics using transform (vectorized)
                result[f'{metric}_{peer_name}_mean'] = result.groupby(valid_cols)[metric].transform('mean')
                result[f'{metric}_{peer_name}_std'] = result.groupby(valid_cols)[metric].transform('std')
                result[f'{metric}_{peer_name}_median'] = result.groupby(valid_cols)[metric].transform('median')
                
                # Calculate percentile rank within peer group (vectorized)
                result[f'{metric}_{peer_name}_pctl'] = result.groupby(valid_cols)[metric].rank(pct=True) * 100
                
                # Performance vs peer mean
                result[f'{metric}_{peer_name}_vs_mean'] = result[metric] - result[f'{metric}_{peer_name}_mean']
        
        logger.info(f"  Benchmarks calculated for {len(result):,} sites")
        
        return result
    
    def calculate_site_rankings(self, site_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate site rankings with percentile-based tiers"""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING SITE RANKINGS")
        logger.info("=" * 70)
        
        result = site_df.copy()
        
        # Composite score components
        score = pd.Series(0.0, index=result.index)
        weight_total = 0
        
        # DQI (higher is better) - 25%
        if 'dqi_mean' in result.columns:
            dqi_min = result['dqi_mean'].min()
            dqi_max = result['dqi_mean'].max()
            if dqi_max > dqi_min:
                result['dqi_norm'] = (result['dqi_mean'] - dqi_min) / (dqi_max - dqi_min) * 100
            else:
                result['dqi_norm'] = 100
            score += result['dqi_norm'] * 0.25
            weight_total += 0.25
        
        # Tier 2 clean rate (higher is better) - 25%
        if 'tier2_clean_rate' in result.columns:
            score += result['tier2_clean_rate'].fillna(0) * 0.25
            weight_total += 0.25
        
        # Ready rate (higher is better) - 25%
        if 'ready_rate' in result.columns:
            score += result['ready_rate'].fillna(0) * 0.25
            weight_total += 0.25
        
        # Queries per patient (lower is better) - 15%
        if 'queries_per_patient' in result.columns:
            q_max = result['queries_per_patient'].max()
            if q_max > 0:
                result['queries_norm'] = (1 - result['queries_per_patient'] / q_max) * 100
            else:
                result['queries_norm'] = 100
            score += result['queries_norm'].fillna(0) * 0.15
            weight_total += 0.15
        
        # Days to ready (lower is better) - 10%
        if 'avg_days_to_ready' in result.columns:
            d_max = result['avg_days_to_ready'].max()
            if d_max > 0:
                result['days_norm'] = (1 - result['avg_days_to_ready'].fillna(0) / d_max) * 100
            else:
                result['days_norm'] = 100
            score += result['days_norm'].fillna(0) * 0.10
            weight_total += 0.10
        
        # Normalize by actual weight used
        if weight_total > 0:
            result['composite_score_raw'] = (score / weight_total).round(1)
        else:
            result['composite_score_raw'] = 50.0
        
        # Apply confidence penalty for small sites
        # Sites with fewer patients have less reliable metrics
        confidence_factor = np.minimum(
            result['patient_count'] / self.config.min_patients_for_full_score, 
            1.0
        )
        # Blend toward mean for small sites
        overall_mean = result['composite_score_raw'].mean()
        result['composite_score'] = (
            result['composite_score_raw'] * confidence_factor + 
            overall_mean * (1 - confidence_factor)
        ).round(1)
        
        # Log adjustment impact
        adjusted = (result['composite_score_raw'] != result['composite_score']).sum()
        logger.info(f"  Small site adjustment applied to {adjusted:,} sites")
        
        # Rankings
        result['overall_rank'] = result['composite_score'].rank(ascending=False, method='min').astype(int)
        result['study_rank'] = result.groupby('study_id')['composite_score'].rank(ascending=False, method='min').astype(int)
        
        if 'region' in result.columns:
            result['region_rank'] = result.groupby('region')['composite_score'].rank(ascending=False, method='min').astype(int)
        
        # Calculate percentile for tier assignment
        result['composite_percentile'] = result['composite_score'].rank(pct=True) * 100
        
        # FIXED: Percentile-based tiers (creates proper bell curve)
        def get_tier_by_percentile(pctl):
            if pctl >= 90:
                return 'Exceptional'
            elif pctl >= 70:
                return 'Strong'
            elif pctl >= 30:
                return 'Average'
            elif pctl >= 10:
                return 'Below Average'
            else:
                return 'Needs Improvement'
        
        result['performance_tier'] = result['composite_percentile'].apply(get_tier_by_percentile)
        
        logger.info(f"\n  PERFORMANCE TIER DISTRIBUTION (Percentile-Based):")
        tier_counts = result['performance_tier'].value_counts()
        tier_order = ['Exceptional', 'Strong', 'Average', 'Below Average', 'Needs Improvement']
        for tier in tier_order:
            count = tier_counts.get(tier, 0)
            pct = count / len(result) * 100
            logger.info(f"    {tier}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def detect_site_outliers(self, site_df: pd.DataFrame) -> pd.DataFrame:
        """Detect site outliers"""
        logger.info("\n" + "=" * 70)
        logger.info("DETECTING SITE OUTLIERS")
        logger.info("=" * 70)
        
        result = site_df.copy()
        
        metrics = ['dqi_mean', 'tier1_clean_rate', 'tier2_clean_rate', 
                   'ready_rate', 'queries_per_patient', 'avg_days_to_ready']
        
        outlier_cols = []
        
        for metric in metrics:
            if metric not in result.columns:
                continue
            
            mean = result[metric].mean()
            std = result[metric].std()
            
            if std > 0:
                result[f'{metric}_zscore'] = ((result[metric] - mean) / std).round(2)
                result[f'{metric}_outlier'] = result[f'{metric}_zscore'].abs() > self.config.outlier_zscore_threshold
                
                outlier_count = result[f'{metric}_outlier'].sum()
                if outlier_count > 0:
                    outlier_cols.append(f'{metric}_outlier')
                    logger.info(f"  {metric}: {outlier_count} outliers detected")
        
        # Overall outlier flag
        if outlier_cols:
            result['is_any_outlier'] = result[outlier_cols].any(axis=1)
            result['outlier_count'] = result[outlier_cols].sum(axis=1)
            
            total = result['is_any_outlier'].sum()
            logger.info(f"\n  Total sites with outliers: {total} ({total/len(result)*100:.1f}%)")
        else:
            result['is_any_outlier'] = False
            result['outlier_count'] = 0
        
        return result
    
    def benchmark_patients_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple patient benchmarking (FAST - vectorized)"""
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARKING PATIENTS VS COHORT (Optimized)")
        logger.info("=" * 70)
        
        result = df.copy()
        
        metrics = ['dqi_score', 'cascade_impact_score', 'total_queries']
        
        for metric in metrics:
            if metric not in result.columns:
                continue
            
            # Calculate site-level stats using transform (vectorized)
            result[f'{metric}_site_mean'] = result.groupby(['study_id', 'site_id'])[metric].transform('mean')
            result[f'{metric}_site_std'] = result.groupby(['study_id', 'site_id'])[metric].transform('std')
            
            # Difference from site mean
            result[f'{metric}_vs_site'] = result[metric] - result[f'{metric}_site_mean']
            
            # Percentile within site (vectorized)
            result[f'{metric}_site_pctl'] = result.groupby(['study_id', 'site_id'])[metric].rank(pct=True) * 100
        
        logger.info(f"  Patient benchmarks calculated for {len(result):,} patients")
        
        return result
    
    def calculate_study_benchmarks(self, site_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate study-level benchmarks with adjusted thresholds"""
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING STUDY BENCHMARKS")
        logger.info("=" * 70)
        
        agg_dict = {
            'site_id': 'count',
            'patient_count': 'sum',
            'dqi_mean': 'mean',
            'tier1_clean_rate': 'mean',
            'tier2_clean_rate': 'mean',
            'ready_rate': 'mean',
            'composite_score': 'mean'
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in site_df.columns}
        
        study_df = site_df.groupby('study_id').agg(agg_dict).reset_index()
        
        study_df = study_df.rename(columns={
            'site_id': 'site_count',
            'dqi_mean': 'avg_dqi',
            'tier1_clean_rate': 'avg_tier1_rate',
            'tier2_clean_rate': 'avg_tier2_rate',
            'ready_rate': 'avg_ready_rate',
            'composite_score': 'avg_site_score'
        })
        
        # Study rank
        if 'avg_site_score' in study_df.columns:
            study_df['study_rank'] = study_df['avg_site_score'].rank(ascending=False, method='min').astype(int)
            
            # Calculate percentile for tier
            study_df['study_percentile'] = study_df['avg_site_score'].rank(pct=True) * 100
            
            # FIXED: Percentile-based study tiers
            def get_study_tier(pctl):
                if pctl >= 75:
                    return 'Top Tier'
                elif pctl >= 40:
                    return 'Mid Tier'
                else:
                    return 'Needs Focus'
            
            study_df['study_tier'] = study_df['study_percentile'].apply(get_study_tier)
        
        logger.info(f"\n  STUDY RANKINGS:")
        tier_counts = study_df['study_tier'].value_counts()
        logger.info(f"  Tier Distribution: {tier_counts.to_dict()}")
        
        for _, row in study_df.sort_values('study_rank' if 'study_rank' in study_df.columns else 'study_id').iterrows():
            score = row.get('avg_site_score', 0)
            tier = row.get('study_tier', 'Unknown')
            rank = row.get('study_rank', 0)
            sites = row.get('site_count', 0)
            patients = row.get('patient_count', 0)
            logger.info(f"    #{int(rank)} {row['study_id']}: {score:.1f} ({tier}) - {int(sites)} sites, {int(patients)} patients")
        
        return study_df
    
    def generate_narratives(self, site_df: pd.DataFrame) -> List[Dict]:
        """Generate site narratives"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING SITE NARRATIVES")
        logger.info("=" * 70)
        
        narratives = []
        
        for _, site in site_df.iterrows():
            narrative = {
                'site_id': site.get('site_id', 'Unknown'),
                'study_id': site.get('study_id', 'Unknown'),
                'patient_count': int(site.get('patient_count', 0)),
                'composite_score': site.get('composite_score', 0),
                'composite_percentile': round(site.get('composite_percentile', 50), 1),
                'performance_tier': site.get('performance_tier', 'Unknown'),
                'overall_rank': int(site.get('overall_rank', 0)),
                'study_rank': int(site.get('study_rank', 0)),
                'is_small_site': bool(site.get('is_small_site', False)),
                'insights': []
            }
            
            # Generate insights
            insights = []
            
            # Small site warning
            if site.get('is_small_site', False):
                insights.append(f"⚠️ Small site ({int(site.get('patient_count', 0))} patients) - scores less reliable")
            
            # DQI insight
            dqi_pctl = site.get('dqi_mean_study_pctl', 50)
            if pd.notna(dqi_pctl):
                if dqi_pctl >= 75:
                    insights.append(f"DQI in top 25% of study (P{dqi_pctl:.0f})")
                elif dqi_pctl <= 25:
                    insights.append(f"DQI in bottom 25% - needs focus (P{dqi_pctl:.0f})")
            
            # Ready rate insight
            ready = site.get('ready_rate', 0)
            if pd.notna(ready):
                if ready >= 50:
                    insights.append(f"Good DB Lock readiness: {ready:.0f}%")
                elif ready < 20:
                    insights.append(f"Low readiness: {ready:.0f}% - priority")
            
            # Outlier insight
            if site.get('is_any_outlier', False):
                insights.append(f"⚠️ Statistical outlier detected")
            
            narrative['insights'] = insights
            narrative['summary'] = '; '.join(insights[:3]) if insights else 'Normal performance'
            
            narratives.append(narrative)
        
        narratives.sort(key=lambda x: x['composite_score'], reverse=True)
        
        logger.info(f"  Generated {len(narratives)} site narratives")
        
        return narratives
    
    def calculate_full_benchmarks(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Run complete benchmark analysis"""
        logger.info("\n" + "=" * 70)
        logger.info("SANCHALAK AI - BENCHMARK ENGINE v1.2")
        logger.info("=" * 70)
        logger.info(f"Patients: {len(df):,}")
        
        start_time = datetime.now()
        
        # Step 1: Prepare site data
        site_df = self.prepare_site_data(df)
        
        # Step 2: Benchmark sites vs peers
        site_df = self.benchmark_sites_vs_peers(site_df)
        
        # Step 3: Calculate rankings
        site_df = self.calculate_site_rankings(site_df)
        
        # Step 4: Detect outliers
        site_df = self.detect_site_outliers(site_df)
        
        # Step 5: Simple patient benchmarking (FAST)
        patient_df = self.benchmark_patients_simple(df)
        
        # Step 6: Study benchmarks
        study_df = self.calculate_study_benchmarks(site_df)
        
        # Step 7: Narratives
        narratives = self.generate_narratives(site_df)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return patient_df, site_df, study_df, narratives


# =============================================================================
# MAIN
# =============================================================================

class BenchmarkAnalyzer:
    """Main Benchmark Analyzer"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.patient_df = None
        self.site_df = None
        self.study_df = None
        self.narratives = None
        self.engine = BenchmarkEngine()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(df):,} patients with {len(df.columns)} columns")
        return df
    
    def run(self) -> pd.DataFrame:
        df = self.load_data()
        self.patient_df, self.site_df, self.study_df, self.narratives = \
            self.engine.calculate_full_benchmarks(df)
        return self.patient_df
    
    def save_outputs(self) -> Dict[str, Path]:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING BENCHMARK OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save to PostgreSQL
        pg_writer = get_pg_writer()
        
        # Patient benchmarks
        success, msg = pg_writer.safe_to_postgres(self.patient_df, 'patient_benchmarks', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_benchmarks to PostgreSQL: {len(self.patient_df):,} rows")
        else:
            patient_path = self.output_dir / 'patient_benchmarks.parquet'
            self.patient_df.to_parquet(patient_path, index=False)
        
        # Site benchmarks
        if self.site_df is not None:
            success, msg = pg_writer.safe_to_postgres(self.site_df, 'site_benchmarks', if_exists='replace')
            if success:
                logger.info(f"✅ Saved site_benchmarks to PostgreSQL: {len(self.site_df):,} rows")
            else:
                site_path = self.output_dir / 'site_benchmarks.parquet'
                self.site_df.to_parquet(site_path, index=False)
        
        site_csv = self.output_dir / 'site_benchmarks.csv'
        self.site_df.to_csv(site_csv, index=False)
        saved_files['site'] = site_path
        logger.info(f"✅ Site benchmarks: {site_path}")
        
        # 3. Study benchmarks
        study_path = self.output_dir / 'study_benchmarks.csv'
        self.study_df.to_csv(study_path, index=False)
        saved_files['study'] = study_path
        logger.info(f"✅ Study benchmarks: {study_path}")
        
        # 4. Site rankings
        rank_cols = ['study_id', 'site_id', 'patient_count', 'is_small_site',
                     'composite_score', 'composite_percentile',
                     'performance_tier', 'overall_rank', 'study_rank', 'is_any_outlier']
        available = [c for c in rank_cols if c in self.site_df.columns]
        
        rankings = self.site_df[available].sort_values('overall_rank')
        rank_path = self.output_dir / 'site_rankings.csv'
        rankings.to_csv(rank_path, index=False)
        saved_files['rankings'] = rank_path
        logger.info(f"✅ Site rankings: {rank_path}")
        
        # 5. Top performers (exclude small sites from "top")
        large_sites = self.site_df[~self.site_df['is_small_site']]
        top_sites = large_sites.nlargest(50, 'composite_score')[available]
        top_path = self.output_dir / 'top_performing_sites.csv'
        top_sites.to_csv(top_path, index=False)
        saved_files['top'] = top_path
        logger.info(f"✅ Top performers: {len(top_sites)} sites (excluding small sites)")
        
        # 6. Bottom performers
        bottom_sites = large_sites.nsmallest(50, 'composite_score')[available]
        bottom_path = self.output_dir / 'sites_needing_improvement.csv'
        bottom_sites.to_csv(bottom_path, index=False)
        saved_files['bottom'] = bottom_path
        logger.info(f"✅ Sites needing improvement: {len(bottom_sites)} sites")
        
        # 7. Outliers
        outliers = self.site_df[self.site_df['is_any_outlier']][available]
        outlier_path = self.output_dir / 'outlier_sites.csv'
        outliers.to_csv(outlier_path, index=False)
        saved_files['outliers'] = outlier_path
        logger.info(f"✅ Outlier sites: {len(outliers)}")
        
        # 8. Small sites
        small_sites = self.site_df[self.site_df['is_small_site']][available]
        small_path = self.output_dir / 'small_sites.csv'
        small_sites.to_csv(small_path, index=False)
        saved_files['small'] = small_path
        logger.info(f"✅ Small sites: {len(small_sites)}")
        
        # 9. Narratives
        narratives_path = self.output_dir / 'benchmark_narratives.json'
        with open(narratives_path, 'w') as f:
            json.dump(self.narratives[:100], f, indent=2, default=str)  # Top 100
        saved_files['narratives'] = narratives_path
        logger.info(f"✅ Narratives: {narratives_path}")
        
        # 10. Summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.2.0',
            'patient_count': len(self.patient_df),
            'site_count': len(self.site_df),
            'study_count': len(self.study_df),
            'small_site_count': int(self.site_df['is_small_site'].sum()),
            'performance_tiers': self.site_df['performance_tier'].value_counts().to_dict(),
            'study_tiers': self.study_df['study_tier'].value_counts().to_dict(),
            'outlier_count': int(self.site_df['is_any_outlier'].sum()),
            'avg_composite_score': round(float(self.site_df['composite_score'].mean()), 2),
            'median_composite_score': round(float(self.site_df['composite_score'].median()), 2)
        }
        
        summary_path = self.output_dir / 'benchmark_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = summary_path
        logger.info(f"✅ Summary: {summary_path}")
        
        return saved_files
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 2.5 COMPLETE - BENCHMARK ENGINE v1.2 (FIXED)")
        print("=" * 70)
        
        print(f"\n🔢 DATA ANALYZED:")
        print(f"   Patients: {len(self.patient_df):,}")
        print(f"   Sites: {len(self.site_df):,}")
        print(f"   Studies: {len(self.study_df):,}")
        small_count = self.site_df['is_small_site'].sum()
        print(f"   Small Sites (<5 patients): {small_count:,} ({small_count/len(self.site_df)*100:.1f}%)")
        
        print(f"\n🏆 SITE PERFORMANCE TIERS (Percentile-Based):")
        tier_order = ['Exceptional', 'Strong', 'Average', 'Below Average', 'Needs Improvement']
        tier_counts = self.site_df['performance_tier'].value_counts()
        for tier in tier_order:
            count = tier_counts.get(tier, 0)
            pct = count / len(self.site_df) * 100
            icons = {'Exceptional': '🥇', 'Strong': '🥈', 'Average': '🥉', 
                     'Below Average': '⚠️', 'Needs Improvement': '🔴'}
            expected = {'Exceptional': '~10%', 'Strong': '~20%', 'Average': '~40%', 
                       'Below Average': '~20%', 'Needs Improvement': '~10%'}
            print(f"   {icons.get(tier, '•')} {tier}: {count:,} ({pct:.1f}%) [expected: {expected[tier]}]")
        
        print(f"\n📈 COMPOSITE SCORE STATS:")
        print(f"   Mean: {self.site_df['composite_score'].mean():.1f}")
        print(f"   Median: {self.site_df['composite_score'].median():.1f}")
        print(f"   Range: {self.site_df['composite_score'].min():.1f} - {self.site_df['composite_score'].max():.1f}")
        
        print(f"\n🏅 TOP 5 SITES (Excluding Small Sites):")
        large_sites = self.site_df[~self.site_df['is_small_site']]
        top5 = large_sites.nlargest(5, 'composite_score')
        for _, row in top5.iterrows():
            print(f"   {row['site_id']} ({row['study_id']}): {row['composite_score']:.1f} | {int(row['patient_count'])} patients")
        
        print(f"\n⚠️ OUTLIERS: {self.site_df['is_any_outlier'].sum()} sites ({self.site_df['is_any_outlier'].sum()/len(self.site_df)*100:.1f}%)")
        
        print(f"\n📚 STUDY RANKINGS:")
        study_tier_counts = self.study_df['study_tier'].value_counts()
        print(f"   Tier Distribution: {study_tier_counts.to_dict()}")
        for _, row in self.study_df.sort_values('study_rank').head(10).iterrows():
            score = row.get('avg_site_score', 0)
            tier = row.get('study_tier', 'Unknown')
            rank = int(row.get('study_rank', 0))
            print(f"   #{rank} {row['study_id']}: {score:.1f} ({tier})")
        
        print(f"\n📁 Output: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_cascade_analysis.parquet'
    
    if not input_path.exists():
        input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_dblock_status.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return
    
    output_dir = project_root / 'data' / 'processed' / 'analytics'
    
    analyzer = BenchmarkAnalyzer(input_path, output_dir)
    analyzer.run()
    analyzer.save_outputs()
    analyzer.print_summary()
    
    return analyzer


if __name__ == '__main__':
    main()