"""
TRIALPULSE NEXUS 10X - Run Causal Hypothesis Engine
Phase 4.4 Runner Script v1.2

Usage:
    cd <project_root>
    python src/run_causal_hypothesis.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the Causal Hypothesis Engine"""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - Causal Hypothesis Engine v1.2")
    print("Phase 4.4: Root Cause Analysis with Evidence Chains")
    print("=" * 70)
    print()
    
    start_time = datetime.now()
    
    # Import engine
    from knowledge.causal_hypothesis_engine import CausalHypothesisEngine
    
    # Initialize
    data_dir = Path('data/processed')
    output_dir = data_dir / 'analytics' / 'causal_hypotheses'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    engine = CausalHypothesisEngine(data_dir)
    
    # Load data
    print("Loading data sources...")
    if not engine.load_data():
        print("Error loading data")
        return
    
    if engine.patient_data is None:
        print("Error: Patient data not loaded")
        return
    
    # Show data info
    print(f"\nData Summary:")
    print(f"  Total patients: {len(engine.patient_data):,}")
    print(f"  Issue columns: {len(engine.issue_columns)}")
    print(f"  Count columns: {len(engine.count_columns)}")
    print(f"  Severity columns: {len(engine.severity_columns)}")
    
    # Count patients with issues
    patients_with_issues = engine._get_patients_with_issues()
    print(f"  Patients with issues: {len(patients_with_issues):,}")
    
    # Analyze population
    print("\nGenerating hypotheses...")
    sample_size = 500  # Larger sample for better coverage
    all_hypotheses = engine.analyze_population(sample_size=sample_size)
    
    print(f"  Generated {len(all_hypotheses):,} hypotheses from {sample_size} patients")
    
    if len(all_hypotheses) == 0:
        print("\nNo hypotheses generated!")
        return
    
    # Save results
    print("\nSaving results...")
    files = engine.save_results(all_hypotheses, output_dir)
    
    # Generate sample reports
    print("\nGenerating hypothesis reports...")
    
    all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    
    reports_file = output_dir / 'top_hypothesis_reports.txt'
    with open(reports_file, 'w', encoding='utf-8') as f:
        f.write("TRIALPULSE NEXUS 10X - Top Causal Hypothesis Reports\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        for h in all_hypotheses[:30]:  # Top 30
            f.write(engine.format_hypothesis_report(h))
            f.write("\n\n")
    
    print(f"  Saved top 30 reports")
    
    # Print summary
    summary = engine.get_summary()
    
    print("\n" + "=" * 70)
    print("CAUSAL HYPOTHESIS ENGINE SUMMARY")
    print("=" * 70)
    
    print(f"\nHypothesis Statistics:")
    print(f"  Total Generated: {summary['total_hypotheses']:,}")
    print(f"  High Confidence (>=70%): {summary['confidence_distribution']['high (>=70%)']:,}")
    print(f"  Medium Confidence (40-70%): {summary['confidence_distribution']['medium (40-70%)']:,}")
    print(f"  Low Confidence (<40%): {summary['confidence_distribution']['low (<40%)']:,}")
    
    if summary['by_issue_type']:
        print(f"\nBy Issue Type:")
        for issue_type, count in sorted(summary['by_issue_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue_type}: {count}")
    
    if summary['top_root_causes']:
        print(f"\nTop Root Causes:")
        for root_cause, count in list(summary['top_root_causes'].items())[:10]:
            print(f"  {root_cause}: {count}")
    
    print(f"\nTemplate Coverage:")
    print(f"  Issue Types: {summary['template_coverage']}")
    print(f"  Total Templates: {summary['total_templates']}")
    
    print(f"\nOutput Files:")
    for file_type, file_path in files.items():
        print(f"  {file_type}: {Path(file_path).name}")
    print(f"  reports: {reports_file.name}")
    
    # Save comprehensive summary
    summary_file = output_dir / 'causal_hypothesis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            **summary,
            'sample_size': sample_size,
            'total_hypotheses': len(all_hypotheses),
            'patients_with_issues': len(patients_with_issues),
            'output_files': {k: str(v) for k, v in files.items()},
            'generated_at': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\nDuration: {duration:.2f} seconds")
    print("\nPhase 4.4 Complete!")
    print("=" * 70)
    
    return engine


if __name__ == "__main__":
    main()