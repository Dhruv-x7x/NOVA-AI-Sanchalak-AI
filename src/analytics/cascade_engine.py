"""
SANCHALAK AI - Cascade Impact Engine v1.0
Phase 2.4: Issue Dependency Graph and Cascade Analysis

Features:
- Build dependency graph (issues → blocks)
- Calculate unlock scores (PageRank variant)
- Identify critical path to DB lock
- Cluster related issues
- Generate cascade narratives
- Site/Study level cascade analysis
- "Fix X → Unlock Y" recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx

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
class CascadeConfig:
    """Cascade Engine Configuration"""
    
    # Issue types and their dependencies
    # Format: issue_type -> list of issues it blocks
    issue_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        # Tier 1 (Clinical) dependencies
        'missing_visits': ['sdv', 'signatures', 'queries'],  # Can't SDV/sign what doesn't exist
        'missing_pages': ['sdv', 'signatures', 'queries'],
        'queries': ['signatures', 'db_lock'],  # Must answer queries before sign
        'sdv': ['signatures', 'db_lock'],  # Must SDV before final sign
        'signatures': ['db_lock'],  # Must sign before lock
        'meddra_coding': ['db_lock'],
        'whodrug_coding': ['db_lock'],
        
        # Tier 2 (Operational) dependencies
        'sae_dm': ['sae_safety', 'db_lock'],  # DM before safety review
        'sae_safety': ['db_lock'],
        'lab_issues': ['db_lock'],
        'edrr': ['db_lock'],
        'broken_signatures': ['db_lock'],
        'overdue_signatures': ['signatures', 'db_lock'],
        'inactivated': ['db_lock']
    })
    
    # Issue type to column mapping
    issue_columns: Dict[str, str] = field(default_factory=lambda: {
        'missing_visits': 'visit_missing_visit_count',
        'missing_pages': 'pages_missing_page_count',
        'queries': 'total_queries',
        'sdv': 'crfs_require_verification_sdv',
        'signatures': 'crfs_never_signed',
        'meddra_coding': 'meddra_coding_meddra_uncoded',
        'whodrug_coding': 'whodrug_coding_whodrug_uncoded',
        'sae_dm': 'sae_dm_sae_dm_pending',
        'sae_safety': 'sae_safety_sae_safety_pending',
        'lab_issues': 'lab_lab_issue_count',
        'edrr': 'edrr_edrr_issue_count',
        'broken_signatures': 'broken_signatures',
        'overdue_signatures': 'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
        'inactivated': 'inactivated_inactivated_form_count'
    })
    
    # Issue type display names
    issue_names: Dict[str, str] = field(default_factory=lambda: {
        'missing_visits': 'Missing Visits',
        'missing_pages': 'Missing Pages',
        'queries': 'Open Queries',
        'sdv': 'SDV Pending',
        'signatures': 'Unsigned CRFs',
        'meddra_coding': 'MedDRA Uncoded',
        'whodrug_coding': 'WHODrug Uncoded',
        'sae_dm': 'SAE DM Pending',
        'sae_safety': 'SAE Safety Pending',
        'lab_issues': 'Lab Issues',
        'edrr': 'EDRR Issues',
        'broken_signatures': 'Broken Signatures',
        'overdue_signatures': 'Overdue Signatures',
        'inactivated': 'Inactivated Forms'
    })
    
    # Responsible roles
    issue_roles: Dict[str, str] = field(default_factory=lambda: {
        'missing_visits': 'CRA',
        'missing_pages': 'CRA',
        'queries': 'Data Manager',
        'sdv': 'CRA',
        'signatures': 'Site',
        'meddra_coding': 'Medical Coder',
        'whodrug_coding': 'Medical Coder',
        'sae_dm': 'Safety Data Manager',
        'sae_safety': 'Safety Physician',
        'lab_issues': 'Data Manager',
        'edrr': 'Data Manager',
        'broken_signatures': 'Site',
        'overdue_signatures': 'Site',
        'inactivated': 'Data Manager'
    })
    
    # Cascade impact weights (how much fixing this issue helps)
    impact_weights: Dict[str, float] = field(default_factory=lambda: {
        'missing_visits': 3.0,    # High impact - blocks many things
        'missing_pages': 3.0,
        'queries': 2.5,
        'sdv': 2.0,
        'signatures': 2.0,
        'sae_dm': 2.5,           # Safety critical
        'sae_safety': 2.5,
        'meddra_coding': 1.0,
        'whodrug_coding': 1.0,
        'lab_issues': 1.0,
        'edrr': 1.0,
        'broken_signatures': 1.5,
        'overdue_signatures': 1.5,
        'inactivated': 0.5
    })


# =============================================================================
# CASCADE IMPACT ENGINE
# =============================================================================

class CascadeImpactEngine:
    """
    Cascade Impact Engine using NetworkX
    
    Analyzes how issues block each other and identifies
    high-impact fixes that unlock multiple downstream issues.
    """
    
    def __init__(self, config: CascadeConfig = None):
        self.config = config or CascadeConfig()
        self.global_graph = None
        self.patient_graphs = {}
        self.cascade_stats = {}
        self.unlock_scores = {}
        
    def _get_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get column"""
        if col in df.columns:
            return df[col].fillna(0)
        return pd.Series(0, index=df.index)
    
    def build_global_dependency_graph(self) -> nx.DiGraph:
        """
        Build the global issue dependency graph
        """
        logger.info("\n" + "=" * 70)
        logger.info("BUILDING GLOBAL DEPENDENCY GRAPH")
        logger.info("=" * 70)
        
        G = nx.DiGraph()
        
        # Add all issue types as nodes
        for issue_type, display_name in self.config.issue_names.items():
            G.add_node(issue_type, 
                      name=display_name,
                      role=self.config.issue_roles.get(issue_type, 'Unknown'),
                      weight=self.config.impact_weights.get(issue_type, 1.0))
        
        # Add DB Lock as target node
        G.add_node('db_lock', name='DB Lock Ready', role='All', weight=5.0)
        
        # Add edges (dependencies)
        for source, targets in self.config.issue_dependencies.items():
            for target in targets:
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, relationship='blocks')
        
        self.global_graph = G
        
        logger.info(f"  Nodes (Issue Types): {G.number_of_nodes()}")
        logger.info(f"  Edges (Dependencies): {G.number_of_edges()}")
        
        # Log graph structure
        logger.info(f"\n  DEPENDENCY STRUCTURE:")
        for node in G.nodes():
            successors = list(G.successors(node))
            if successors:
                succ_names = [self.config.issue_names.get(s, s) for s in successors]
                logger.info(f"    {self.config.issue_names.get(node, node)} → {', '.join(succ_names)}")
        
        return G
    
    def calculate_pagerank_scores(self) -> Dict[str, float]:
        """
        Calculate PageRank-like scores for issue importance
        Higher score = fixing this issue has higher cascade impact
        """
        logger.info("\n" + "=" * 70)
        logger.info("CALCULATING UNLOCK SCORES (PageRank)")
        logger.info("=" * 70)
        
        if self.global_graph is None:
            self.build_global_dependency_graph()
        
        # Reverse the graph - we want to find nodes that unlock the most
        G_reversed = self.global_graph.reverse()
        
        # Calculate PageRank on reversed graph
        pagerank = nx.pagerank(G_reversed, alpha=0.85, weight='weight')
        
        # Combine with impact weights
        combined_scores = {}
        for node, pr_score in pagerank.items():
            weight = self.config.impact_weights.get(node, 1.0)
            combined_scores[node] = pr_score * weight
        
        # Normalize to 0-100
        max_score = max(combined_scores.values()) if combined_scores else 1
        self.unlock_scores = {k: round(v / max_score * 100, 2) for k, v in combined_scores.items()}
        
        # Sort by score
        sorted_scores = sorted(self.unlock_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n  UNLOCK IMPACT SCORES (Higher = More Impact):")
        for issue, score in sorted_scores:
            name = self.config.issue_names.get(issue, issue)
            logger.info(f"    {name}: {score:.1f}")
        
        return self.unlock_scores
    
    def analyze_patient_cascades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze cascade impact for each patient
        """
        logger.info("\n" + "=" * 70)
        logger.info("ANALYZING PATIENT-LEVEL CASCADES")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Get issue counts for each patient
        def get_patient_issues(row):
            """Get list of issues for a patient"""
            issues = []
            for issue_type, column in self.config.issue_columns.items():
                if column in row.index:
                    count = row[column]
                    if pd.notna(count) and count > 0:
                        issues.append({
                            'type': issue_type,
                            'name': self.config.issue_names.get(issue_type, issue_type),
                            'count': int(count),
                            'unlock_score': self.unlock_scores.get(issue_type, 0)
                        })
            return issues
        
        def get_blocking_chain(row):
            """Get the blocking chain for a patient"""
            issues = get_patient_issues(row)
            if not issues:
                return []
            
            # Sort by unlock score (highest first)
            issues.sort(key=lambda x: x['unlock_score'], reverse=True)
            
            # Build chain
            chain = []
            for issue in issues:
                blocked_by = self.config.issue_dependencies.get(issue['type'], [])
                blocked_names = [self.config.issue_names.get(b, b) for b in blocked_by]
                chain.append({
                    'issue': issue['name'],
                    'count': issue['count'],
                    'blocks': blocked_names,
                    'unlock_score': issue['unlock_score']
                })
            
            return chain
        
        def calculate_cascade_impact(row):
            """Calculate total cascade impact score"""
            issues = get_patient_issues(row)
            if not issues:
                return 0.0
            
            total_impact = 0.0
            for issue in issues:
                # Impact = unlock_score * log(count + 1)
                count = issue['count']
                score = issue['unlock_score']
                impact = score * np.log1p(count)
                total_impact += impact
            
            return round(total_impact, 2)
        
        def get_primary_cascade_blocker(row):
            """Get the issue with highest unlock impact"""
            issues = get_patient_issues(row)
            if not issues:
                return None
            
            # Get issue with highest unlock score
            best = max(issues, key=lambda x: x['unlock_score'])
            return best['type']
        
        def format_cascade_recommendation(row):
            """Format the cascade recommendation"""
            issues = get_patient_issues(row)
            if not issues:
                return 'Clean - No cascade'
            
            # Sort by unlock score
            issues.sort(key=lambda x: x['unlock_score'], reverse=True)
            
            if len(issues) == 1:
                return f"Fix {issues[0]['name']} ({issues[0]['count']}) → DB Lock Ready"
            
            # Show cascade chain
            parts = []
            for i, issue in enumerate(issues[:3]):
                blocked = self.config.issue_dependencies.get(issue['type'], [])
                if blocked and i < len(issues) - 1:
                    next_blocked = [self.config.issue_names.get(b, b) for b in blocked if b != 'db_lock']
                    if next_blocked:
                        parts.append(f"Fix {issue['name']} ({issue['count']}) → Unlocks {next_blocked[0]}")
                    else:
                        parts.append(f"Fix {issue['name']} ({issue['count']})")
                else:
                    parts.append(f"Fix {issue['name']} ({issue['count']})")
            
            return ' → '.join(parts) + ' → DB Lock'
        
        # Apply to all patients
        logger.info("  Calculating cascade metrics for each patient...")
        
        result['cascade_impact_score'] = result.apply(calculate_cascade_impact, axis=1)
        result['cascade_primary_blocker'] = result.apply(get_primary_cascade_blocker, axis=1)
        result['cascade_recommendation'] = result.apply(format_cascade_recommendation, axis=1)
        
        # Count issues per patient
        def count_issue_types(row):
            issues = get_patient_issues(row)
            return len(issues)
        
        result['cascade_issue_count'] = result.apply(count_issue_types, axis=1)
        
        # Log summary
        has_issues = result['cascade_issue_count'] > 0
        logger.info(f"\n  Patients with cascade issues: {has_issues.sum():,}")
        logger.info(f"  Average cascade impact score: {result[has_issues]['cascade_impact_score'].mean():.2f}")
        
        return result
    
    def identify_critical_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify critical paths to DB Lock for each patient
        """
        logger.info("\n" + "=" * 70)
        logger.info("IDENTIFYING CRITICAL PATHS")
        logger.info("=" * 70)
        
        result = df.copy()
        
        def get_critical_path(row):
            """Get the critical path to DB Lock"""
            issues = []
            for issue_type, column in self.config.issue_columns.items():
                if column in row.index:
                    count = row[column]
                    if pd.notna(count) and count > 0:
                        issues.append(issue_type)
            
            if not issues:
                return 'Ready'
            
            # Find path using graph
            if self.global_graph is None:
                return 'Unknown'
            
            # Get all paths to db_lock
            paths = []
            for issue in issues:
                try:
                    if nx.has_path(self.global_graph, issue, 'db_lock'):
                        path = nx.shortest_path(self.global_graph, issue, 'db_lock')
                        paths.append(path)
                except nx.NetworkXError:
                    pass
            
            if not paths:
                return 'Direct to DB Lock'
            
            # Find longest path (critical path)
            critical = max(paths, key=len)
            path_names = [self.config.issue_names.get(p, p) for p in critical]
            
            return ' → '.join(path_names)
        
        def get_path_length(row):
            """Get length of critical path"""
            path = row.get('cascade_critical_path', '')
            if path in ['Ready', 'Unknown', 'Direct to DB Lock']:
                return 0
            return len(path.split(' → '))
        
        result['cascade_critical_path'] = result.apply(get_critical_path, axis=1)
        result['cascade_path_length'] = result.apply(get_path_length, axis=1)
        
        # Log path distribution
        logger.info(f"\n  CRITICAL PATH LENGTH DISTRIBUTION:")
        path_lengths = result['cascade_path_length'].value_counts().sort_index()
        for length, count in path_lengths.items():
            pct = count / len(result) * 100
            logger.info(f"    {length} steps: {count:,} ({pct:.1f}%)")
        
        return result
    
    def cluster_related_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster patients with similar issue patterns
        """
        logger.info("\n" + "=" * 70)
        logger.info("CLUSTERING RELATED ISSUES")
        logger.info("=" * 70)
        
        result = df.copy()
        
        # Create issue pattern fingerprint
        def get_issue_pattern(row):
            """Create binary pattern of issues"""
            pattern = []
            for issue_type in sorted(self.config.issue_columns.keys()):
                column = self.config.issue_columns[issue_type]
                if column in row.index:
                    has_issue = row[column] > 0 if pd.notna(row[column]) else False
                else:
                    has_issue = False
                pattern.append('1' if has_issue else '0')
            return ''.join(pattern)
        
        result['cascade_issue_pattern'] = result.apply(get_issue_pattern, axis=1)
        
        # Name common patterns
        pattern_counts = result['cascade_issue_pattern'].value_counts()
        
        def name_pattern(pattern):
            """Give meaningful name to pattern"""
            if pattern == '0' * len(pattern):
                return 'Clean'
            
            issues = []
            for i, (issue_type, _) in enumerate(sorted(self.config.issue_columns.items())):
                if pattern[i] == '1':
                    issues.append(self.config.issue_names.get(issue_type, issue_type))
            
            if len(issues) == 1:
                return f"Only: {issues[0]}"
            elif len(issues) <= 3:
                return f"Combo: {' + '.join(issues)}"
            else:
                return f"Complex: {len(issues)} issues"
        
        result['cascade_cluster'] = result['cascade_issue_pattern'].apply(name_pattern)
        
        # Log clusters
        logger.info(f"\n  TOP ISSUE CLUSTERS:")
        cluster_counts = result['cascade_cluster'].value_counts().head(10)
        for cluster, count in cluster_counts.items():
            pct = count / len(result) * 100
            logger.info(f"    {cluster}: {count:,} ({pct:.1f}%)")
        
        return result
    
    def generate_site_cascade_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate site-level cascade analysis
        """
        logger.info("\n" + "=" * 70)
        logger.info("SITE-LEVEL CASCADE ANALYSIS")
        logger.info("=" * 70)
        
        # Aggregate by site
        agg_dict = {
            'patient_key': 'count',
            'cascade_impact_score': ['sum', 'mean'],
            'cascade_issue_count': 'sum',
            'cascade_path_length': 'max'
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        agg_dict['patient_key'] = 'count'
        
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        
        # Flatten columns
        new_cols = []
        for col in site_df.columns:
            if isinstance(col, tuple):
                new_cols.append('_'.join([str(c) for c in col if c]))
            else:
                new_cols.append(col)
        site_df.columns = new_cols
        
        site_df = site_df.rename(columns={
            'patient_key_count': 'patient_count',
            'patient_key': 'patient_count',
            'cascade_impact_score_sum': 'total_cascade_impact',
            'cascade_impact_score_mean': 'avg_cascade_impact',
            'cascade_issue_count_sum': 'total_issues',
            'cascade_path_length_max': 'max_path_length'
        })
        
        # Calculate cascade density
        if 'total_issues' in site_df.columns and 'patient_count' in site_df.columns:
            site_df['cascade_density'] = (site_df['total_issues'] / site_df['patient_count']).round(2)
        
        # Find top blocker per site
        def get_site_top_blocker(study_id, site_id):
            site_patients = df[(df['study_id'] == study_id) & (df['site_id'] == site_id)]
            if len(site_patients) == 0:
                return None
            blockers = site_patients['cascade_primary_blocker'].value_counts()
            return blockers.index[0] if len(blockers) > 0 else None
        
        site_df['top_blocker'] = site_df.apply(
            lambda row: get_site_top_blocker(row['study_id'], row['site_id']), axis=1
        )
        
        # Cascade priority score
        if 'total_cascade_impact' in site_df.columns:
            max_impact = site_df['total_cascade_impact'].max()
            site_df['cascade_priority'] = (site_df['total_cascade_impact'] / max_impact * 100).round(1) if max_impact > 0 else 0
        
        logger.info(f"\n  Sites analyzed: {len(site_df):,}")
        if 'avg_cascade_impact' in site_df.columns:
            logger.info(f"  Average cascade impact per site: {site_df['avg_cascade_impact'].mean():.2f}")
        
        return site_df
    
    def generate_cascade_narratives(self, df: pd.DataFrame, top_n: int = 20) -> List[Dict]:
        """
        Generate human-readable cascade narratives
        """
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING CASCADE NARRATIVES")
        logger.info("=" * 70)
        
        narratives = []
        
        # Get patients with highest cascade impact
        eligible_mask = df.get('dblock_eligible', pd.Series(True, index=df.index))
        not_ready_mask = ~df.get('dblock_tier1_ready', pd.Series(False, index=df.index))
        
        candidates = df[eligible_mask & not_ready_mask].nlargest(top_n, 'cascade_impact_score')
        
        for _, row in candidates.iterrows():
            # Build narrative
            patient_key = row.get('patient_key', 'Unknown')
            study = row.get('study_id', 'Unknown')
            site = row.get('site_id', 'Unknown')
            
            # Get issues
            issues = []
            for issue_type, column in self.config.issue_columns.items():
                if column in row.index:
                    count = row[column]
                    if pd.notna(count) and count > 0:
                        issues.append({
                            'type': issue_type,
                            'name': self.config.issue_names.get(issue_type, issue_type),
                            'count': int(count),
                            'role': self.config.issue_roles.get(issue_type, 'Unknown'),
                            'unlock_score': self.unlock_scores.get(issue_type, 0)
                        })
            
            if not issues:
                continue
            
            # Sort by unlock score
            issues.sort(key=lambda x: x['unlock_score'], reverse=True)
            
            # Build narrative
            primary = issues[0]
            downstream_count = len(issues) - 1
            
            narrative = {
                'patient_key': patient_key,
                'study_id': study,
                'site_id': site,
                'cascade_score': row.get('cascade_impact_score', 0),
                'primary_issue': primary['name'],
                'primary_count': primary['count'],
                'primary_role': primary['role'],
                'total_issues': len(issues),
                'narrative': f"Fix {primary['name']} ({primary['count']} items) at {site}",
                'impact': f"Unlocks {downstream_count} downstream issues → Path to DB Lock"
            }
            
            # Build full recommendation
            if len(issues) > 1:
                chain_parts = []
                for i, issue in enumerate(issues[:4]):
                    chain_parts.append(f"{issue['name']} ({issue['count']})")
                
                narrative['action_chain'] = ' → '.join(chain_parts) + ' → DB Lock'
            else:
                narrative['action_chain'] = f"{primary['name']} → DB Lock"
            
            narratives.append(narrative)
        
        logger.info(f"\n  Generated {len(narratives)} cascade narratives")
        
        # Show top 5
        logger.info(f"\n  TOP CASCADE OPPORTUNITIES:")
        for i, n in enumerate(narratives[:5], 1):
            logger.info(f"    {i}. {n['narrative']}")
            logger.info(f"       Impact: {n['impact']}")
        
        return narratives
    
    def calculate_full_cascade_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Run complete cascade analysis
        """
        logger.info("\n" + "=" * 70)
        logger.info("SANCHALAK AI - CASCADE IMPACT ENGINE v1.0")
        logger.info("=" * 70)
        logger.info(f"Patients: {len(df):,}")
        
        start_time = datetime.now()
        
        # Step 1: Build global graph
        self.build_global_dependency_graph()
        
        # Step 2: Calculate unlock scores
        self.calculate_pagerank_scores()
        
        # Step 3: Analyze patient cascades
        result = self.analyze_patient_cascades(df)
        
        # Step 4: Identify critical paths
        result = self.identify_critical_paths(result)
        
        # Step 5: Cluster issues
        result = self.cluster_related_issues(result)
        
        # Step 6: Site-level analysis
        site_df = self.generate_site_cascade_analysis(result)
        
        # Step 7: Generate narratives
        narratives = self.generate_cascade_narratives(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("CASCADE ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return result, site_df, narratives


# =============================================================================
# MAIN ENGINE
# =============================================================================

class CascadeEngine:
    """Main Cascade Impact Engine"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.site_df = None
        self.narratives = None
        self.calculator = CascadeImpactEngine()
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(self.df):,} patients with {len(self.df.columns)} columns")
        return self.df
    
    def run(self) -> pd.DataFrame:
        self.df, self.site_df, self.narratives = self.calculator.calculate_full_cascade_analysis(self.df)
        return self.df
    
    def save_outputs(self) -> Dict[str, Path]:
        logger.info("\n" + "=" * 60)
        logger.info("SAVING CASCADE OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # Save to PostgreSQL
        pg_writer = get_pg_writer()
        success, msg = pg_writer.safe_to_postgres(self.df, 'patient_cascade_analysis', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_cascade_analysis to PostgreSQL: {len(self.df):,} rows")
        else:
            logger.warning(f"PostgreSQL failed, using parquet: {msg}")
            patient_path = self.output_dir / 'patient_cascade_analysis.parquet'
            self.df.to_parquet(patient_path, index=False)
        
        # 2. Site cascade data
        site_path = self.output_dir / 'site_cascade_analysis.csv'
        self.site_df.to_csv(site_path, index=False)
        saved_files['site'] = site_path
        logger.info(f"✅ Site cascade: {site_path}")
        
        # 3. Cascade narratives
        narratives_path = self.output_dir / 'cascade_narratives.json'
        with open(narratives_path, 'w') as f:
            json.dump(self.narratives, f, indent=2, default=str)
        saved_files['narratives'] = narratives_path
        logger.info(f"✅ Narratives: {narratives_path}")
        
        # 4. High impact patients
        high_impact_cols = ['patient_key', 'study_id', 'site_id', 'subject_id',
                           'cascade_impact_score', 'cascade_primary_blocker',
                           'cascade_issue_count', 'cascade_critical_path',
                           'cascade_recommendation']
        available = [c for c in high_impact_cols if c in self.df.columns]
        
        high_impact = self.df[self.df['cascade_impact_score'] > 0].nlargest(1000, 'cascade_impact_score')[available]
        high_path = self.output_dir / 'cascade_high_impact_patients.csv'
        high_impact.to_csv(high_path, index=False)
        saved_files['high_impact'] = high_path
        logger.info(f"✅ High impact patients: {len(high_impact):,}")
        
        # 5. Unlock scores
        scores_df = pd.DataFrame([
            {'issue_type': k, 
             'issue_name': self.calculator.config.issue_names.get(k, k),
             'unlock_score': v,
             'responsible': self.calculator.config.issue_roles.get(k, 'Unknown')}
            for k, v in self.calculator.unlock_scores.items()
        ]).sort_values('unlock_score', ascending=False)
        
        scores_path = self.output_dir / 'cascade_unlock_scores.csv'
        scores_df.to_csv(scores_path, index=False)
        saved_files['unlock_scores'] = scores_path
        logger.info(f"✅ Unlock scores: {scores_path}")
        
        # 6. Issue clusters
        cluster_counts = self.df['cascade_cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'patient_count']
        cluster_counts['percentage'] = (cluster_counts['patient_count'] / len(self.df) * 100).round(2)
        
        cluster_path = self.output_dir / 'cascade_issue_clusters.csv'
        cluster_counts.to_csv(cluster_path, index=False)
        saved_files['clusters'] = cluster_path
        logger.info(f"✅ Issue clusters: {cluster_path}")
        
        # 7. Summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'patient_count': len(self.df),
            'patients_with_cascades': int((self.df['cascade_issue_count'] > 0).sum()),
            'avg_cascade_impact': float(self.df['cascade_impact_score'].mean()),
            'max_cascade_impact': float(self.df['cascade_impact_score'].max()),
            'unlock_scores': self.calculator.unlock_scores,
            'top_clusters': self.df['cascade_cluster'].value_counts().head(5).to_dict(),
            'graph_stats': {
                'nodes': self.calculator.global_graph.number_of_nodes() if self.calculator.global_graph else 0,
                'edges': self.calculator.global_graph.number_of_edges() if self.calculator.global_graph else 0
            }
        }
        
        summary_path = self.output_dir / 'cascade_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = summary_path
        logger.info(f"✅ Summary: {summary_path}")
        
        # 8. Export graph as edge list
        if self.calculator.global_graph:
            edges = []
            for source, target, data in self.calculator.global_graph.edges(data=True):
                edges.append({
                    'source': source,
                    'source_name': self.calculator.config.issue_names.get(source, source),
                    'target': target,
                    'target_name': self.calculator.config.issue_names.get(target, target),
                    'relationship': data.get('relationship', 'blocks')
                })
            
            edges_df = pd.DataFrame(edges)
            edges_path = self.output_dir / 'cascade_dependency_graph.csv'
            edges_df.to_csv(edges_path, index=False)
            saved_files['graph'] = edges_path
            logger.info(f"✅ Dependency graph: {edges_path}")
        
        return saved_files
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 2.4 COMPLETE - CASCADE IMPACT ENGINE v1.0")
        print("=" * 70)
        
        print(f"\n🔢 PATIENTS: {len(self.df):,}")
        
        with_issues = (self.df['cascade_issue_count'] > 0).sum()
        print(f"\n🔗 CASCADE ANALYSIS:")
        print(f"   Patients with cascades: {with_issues:,} ({with_issues/len(self.df)*100:.1f}%)")
        print(f"   Average impact score: {self.df['cascade_impact_score'].mean():.2f}")
        print(f"   Max impact score: {self.df['cascade_impact_score'].max():.2f}")
        
        print(f"\n📈 UNLOCK SCORES (Fix these first for max impact):")
        sorted_scores = sorted(self.calculator.unlock_scores.items(), key=lambda x: x[1], reverse=True)
        for issue, score in sorted_scores[:7]:
            name = self.calculator.config.issue_names.get(issue, issue)
            role = self.calculator.config.issue_roles.get(issue, 'Unknown')
            print(f"   {name}: {score:.1f} ({role})")
        
        print(f"\n🎯 TOP ISSUE CLUSTERS:")
        clusters = self.df['cascade_cluster'].value_counts().head(5)
        for cluster, count in clusters.items():
            pct = count / len(self.df) * 100
            print(f"   {cluster}: {count:,} ({pct:.1f}%)")
        
        print(f"\n💡 TOP CASCADE OPPORTUNITIES:")
        for i, n in enumerate(self.narratives[:5], 1):
            print(f"   {i}. {n['narrative']}")
            print(f"      Chain: {n['action_chain']}")
        
        print(f"\n📁 Output: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    # Use DB Lock status output
    input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_dblock_status.parquet'
    
    if not input_path.exists():
        input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_clean_status.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return
    
    output_dir = project_root / 'data' / 'processed' / 'analytics'
    
    engine = CascadeEngine(input_path, output_dir)
    engine.load_data()
    engine.run()
    engine.save_outputs()
    engine.print_summary()
    
    return engine


if __name__ == '__main__':
    main()