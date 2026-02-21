"""
SANCHALAK AI - Training Data Preparation
==============================================
Generate fine-tuning training data from your clinical trial database.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .config import get_finetuning_config, TRAINING_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    input: str
    output: str
    category: str
    
    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category
        }
    
    def to_chat_format(self, system_prompt: str = "") -> Dict:
        """Convert to chat format for training."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{self.instruction}\n\n{self.input}" if self.input else self.instruction},
                {"role": "assistant", "content": self.output}
            ]
        }


class TrainingDataGenerator:
    """
    Generate training data from your clinical trial database.
    
    Creates high-quality Q&A pairs for fine-tuning.
    """
    
    def __init__(self):
        self.config = get_finetuning_config()
        self._examples: List[TrainingExample] = []
    
    def generate_from_database(self) -> int:
        """Generate training examples from PostgreSQL database."""
        logger.info("Generating training data from database...")
        
        try:
            from src.database.db_loader import get_db_loader

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer

            loader = get_db_loader()
            
            # Get patient data
            patients_df = loader.get_patients_df()
            
            # Generate different types of examples
            count = 0
            count += self._generate_dqi_examples(patients_df)
            count += self._generate_issue_examples(patients_df)
            count += self._generate_site_examples(patients_df)
            count += self._generate_query_examples(patients_df)
            count += self._generate_analysis_examples(patients_df)
            
            logger.info(f"Generated {count} training examples")
            return count
            
        except Exception as e:
            logger.error(f"Failed to generate from database: {e}")
            return 0
    
    def _generate_dqi_examples(self, df) -> int:
        """Generate DQI analysis examples."""
        examples = []
        
        # Sample patients with different DQI levels
        if 'dqi_score' in df.columns:
            for _, patient in df.sample(min(200, len(df))).iterrows():
                dqi = patient.get('dqi_score', 85)
                patient_key = patient.get('patient_key', 'Unknown')
                study = patient.get('study_id', 'Unknown')
                
                # Determine DQI band
                if dqi >= 95:
                    band, desc = "Elite", "exceptional data quality"
                elif dqi >= 90:
                    band, desc = "Optimal", "very good data quality"
                elif dqi >= 85:
                    band, desc = "Standard", "acceptable data quality"
                elif dqi >= 80:
                    band, desc = "Risk", "data quality needs improvement"
                else:
                    band, desc = "Critical", "urgent data quality issues"
                
                # Create training example
                instruction = "Analyze the data quality for this patient and provide recommendations."
                input_text = f"""Patient: {patient_key}
Study: {study}
DQI Score: {dqi:.1f}
Open Queries: {patient.get('total_open_queries', 0)}
Missing Values: {patient.get('missing_value_count', 0)}"""
                
                output = f"""## Data Quality Analysis

**Patient:** {patient_key}
**DQI Score:** {dqi:.1f}/100 ({band} Band)

### Assessment
This patient demonstrates {desc}. The DQI score of {dqi:.1f} places them in the {band} band.

### Key Observations
- Open queries: {patient.get('total_open_queries', 0)} ({"needs attention" if patient.get('total_open_queries', 0) > 3 else "acceptable"})
- Missing values: {patient.get('missing_value_count', 0)} ({"requires review" if patient.get('missing_value_count', 0) > 5 else "within limits"})

### Recommendations
{"1. Priority review required - escalate to DM lead" if dqi < 80 else "1. Continue standard monitoring"}
{"2. Address open queries within 5 business days" if patient.get('total_open_queries', 0) > 0 else "2. No immediate query actions needed"}
{"3. Review missing data fields for completeness" if patient.get('missing_value_count', 0) > 3 else "3. Data completeness is satisfactory"}"""

                examples.append(TrainingExample(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    category="data_quality"
                ))
        
        self._examples.extend(examples)
        return len(examples)
    
    def _generate_issue_examples(self, df) -> int:
        """Generate issue resolution examples."""
        examples = []
        
        issue_types = [
            ("missing_consent_date", "Missing Consent Date", "high", 
             "Verify consent form is signed. Contact site to submit scanned consent form. Update EDC within 48 hours."),
            ("protocol_deviation", "Protocol Deviation", "critical",
             "Document deviation in protocol deviation log. Assess impact on patient safety. Report to IRB if required."),
            ("query_overdue", "Overdue Query", "medium",
             "Send reminder to site. Escalate to CRA if >7 days overdue. Document in monitoring report."),
            ("sae_reconciliation", "SAE Reconciliation Issue", "critical",
             "URGENT: Verify SAE data matches safety database. Reconcile discrepancies within 24 hours. Notify PV team."),
            ("data_entry_error", "Data Entry Error", "low",
             "Issue data clarification form. Site to correct and initial. Verify correction in next SDV.")
        ]
        
        for _, patient in df.sample(min(100, len(df))).iterrows():
            issue = random.choice(issue_types)
            
            instruction = f"How should I resolve this {issue[1]} issue?"
            input_text = f"""Issue Type: {issue[1]}
Severity: {issue[2]}
Patient: {patient.get('patient_key', 'Unknown')}
Study: {patient.get('study_id', 'Unknown')}
Site: {patient.get('site_id', 'Unknown')}"""
            
            output = f"""## Issue Resolution: {issue[1]}

**Severity:** {issue[2].upper()}
**Category:** {issue[0]}

### Resolution Steps
{issue[3]}

### Timeline
- {"Immediate action required" if issue[2] == "critical" else "Resolve within 5 business days" if issue[2] == "high" else "Standard timeline: 10 business days"}

### Documentation
- Log resolution in issue tracker
- Update patient notes
- {"Notify sponsor if required" if issue[2] in ["critical", "high"] else "Standard documentation only"}

### Escalation
{"⚠️ ESCALATE TO: Study Lead, Medical Monitor" if issue[2] == "critical" else "No escalation required for this severity level."}"""

            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output,
                category="issue_resolution"
            ))
        
        self._examples.extend(examples)
        return len(examples)
    
    def _generate_site_examples(self, df) -> int:
        """Generate site performance examples."""
        examples = []
        
        if 'site_id' in df.columns:
            # Group by site
            site_stats = df.groupby('site_id').agg({
                'dqi_score': ['mean', 'count'] if 'dqi_score' in df.columns else ['count'],
                'total_open_queries': 'sum' if 'total_open_queries' in df.columns else 'count'
            }).head(50)
            
            for site_id in site_stats.index:
                try:
                    mean_dqi = site_stats.loc[site_id, ('dqi_score', 'mean')] if 'dqi_score' in df.columns else 85
                    patient_count = site_stats.loc[site_id, ('dqi_score', 'count')] if 'dqi_score' in df.columns else 10
                    total_queries = site_stats.loc[site_id, ('total_open_queries', 'sum')] if 'total_open_queries' in df.columns else 5
                except:
                    continue
                
                performance = "excellent" if mean_dqi >= 90 else "good" if mean_dqi >= 85 else "needs improvement" if mean_dqi >= 80 else "poor"
                
                instruction = "Analyze this site's performance and provide recommendations."
                input_text = f"""Site ID: {site_id}
Patients Enrolled: {patient_count}
Mean DQI: {mean_dqi:.1f}
Total Open Queries: {total_queries}"""
                
                output = f"""## Site Performance Analysis

**Site:** {site_id}
**Performance Rating:** {performance.upper()}

### Metrics Summary
| Metric | Value | Benchmark |
|--------|-------|-----------|
| Mean DQI | {mean_dqi:.1f} | 85+ |
| Patients | {patient_count} | - |
| Open Queries | {total_queries} | <5 per patient |

### Assessment
This site is performing {"above" if mean_dqi >= 85 else "below"} benchmark with a mean DQI of {mean_dqi:.1f}.

### Recommendations
{"✅ Continue current practices - site is a top performer" if performance == "excellent" else ""}
{"📊 Monitor query trends, provide site training if needed" if performance == "good" else ""}
{"⚠️ Schedule site quality review meeting" if performance == "needs improvement" else ""}
{"🚨 URGENT: Intensive monitoring required, consider pause enrollment" if performance == "poor" else ""}"""

                examples.append(TrainingExample(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    category="site_performance"
                ))
        
        self._examples.extend(examples)
        return len(examples)
    
    def _generate_query_examples(self, df) -> int:
        """Generate query response examples."""
        examples = []
        
        query_templates = [
            ("Date discrepancy between visit date and ICF date", 
             "Please clarify: The visit date (Visit 1) appears to be before the Informed Consent Form date. Please verify and correct if needed, or provide explanation."),
            ("Missing laboratory value",
             "Please provide: The hemoglobin value for Visit 2 is missing. Please enter the lab result from the source document or confirm if test was not performed."),
            ("Out of range vital sign",
             "Please clarify: The blood pressure reading of 180/120 mmHg is outside normal range. Please verify this is correct from source, or correct if data entry error."),
            ("Inconsistent medication dates",
             "Please clarify: The medication stop date appears before the start date. Please review source documents and correct the dates."),
            ("Incomplete adverse event form",
             "Please complete: The adverse event severity and outcome fields are blank. Please complete all required fields for this AE.")
        ]
        
        for i, template in enumerate(query_templates):
            for j in range(20):  # 20 variations each
                instruction = "Draft a professional query response for this data discrepancy."
                input_text = f"Query Type: {template[0]}"
                output = f"""## Query Response

**Query:** {template[0]}

### Response
{template[1]}

### Additional Notes
- This query was auto-generated based on edit check rules
- Response deadline: 5 business days from receipt
- Contact: CRA or Data Manager for questions

### Status
Awaiting Site Response"""

                examples.append(TrainingExample(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                    category="query_management"
                ))
        
        self._examples.extend(examples)
        return len(examples)
    
    def _generate_analysis_examples(self, df) -> int:
        """Generate general analysis examples."""
        examples = []
        
        # Portfolio summary
        if len(df) > 0:
            total_patients = len(df)
            mean_dqi = df['dqi_score'].mean() if 'dqi_score' in df.columns else 85
            total_queries = df['total_open_queries'].sum() if 'total_open_queries' in df.columns else 0
            
            instruction = "Provide an executive summary of the clinical trial portfolio status."
            input_text = f"""Total Patients: {total_patients}
Mean DQI: {mean_dqi:.1f}
Total Open Queries: {total_queries}
Studies: {df['study_id'].nunique() if 'study_id' in df.columns else 1}
Sites: {df['site_id'].nunique() if 'site_id' in df.columns else 1}"""
            
            output = f"""## Executive Portfolio Summary

### Overview
The clinical trial portfolio currently tracks **{total_patients:,} patients** across {df['study_id'].nunique() if 'study_id' in df.columns else 1} studies and {df['site_id'].nunique() if 'site_id' in df.columns else 1} sites.

### Data Quality Metrics
- **Portfolio DQI:** {mean_dqi:.1f}/100 ({"Optimal" if mean_dqi >= 90 else "Standard" if mean_dqi >= 85 else "Needs Attention"})
- **Open Queries:** {total_queries:,} (avg {total_queries/max(1,total_patients):.1f} per patient)

### Key Observations
{"✅ Portfolio is performing well with strong data quality" if mean_dqi >= 90 else ""}
{"📊 Data quality is acceptable but monitor trends" if 85 <= mean_dqi < 90 else ""}
{"⚠️ Action needed to improve data quality metrics" if mean_dqi < 85 else ""}

### Recommendations
1. {"Maintain current quality processes" if mean_dqi >= 90 else "Focus on query resolution to improve DQI"}
2. {"Continue regular monitoring" if total_queries/max(1,total_patients) < 2 else "Address query backlog - consider additional resources"}
3. Schedule bi-weekly portfolio review meetings"""

            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=output,
                category="data_quality"
            ))
        
        self._examples.extend(examples)
        return len(examples)
    
    def save_training_data(self, format: str = "chat") -> Path:
        """
        Save training data to JSONL file.
        
        Args:
            format: "chat" for chat format, "alpaca" for instruction format
        """
        output_file = self.config.data_dir / f"training_data_{format}.jsonl"
        
        # Shuffle examples
        random.shuffle(self._examples)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self._examples:
                if format == "chat":
                    system_prompt = TRAINING_SYSTEM_PROMPTS.get(example.category, "")
                    data = example.to_chat_format(system_prompt)
                else:
                    data = example.to_dict()
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self._examples)} examples to {output_file}")
        return output_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training data statistics."""
        category_counts = {}
        for ex in self._examples:
            category_counts[ex.category] = category_counts.get(ex.category, 0) + 1
        
        return {
            "total_examples": len(self._examples),
            "categories": category_counts,
            "avg_instruction_length": sum(len(ex.instruction) for ex in self._examples) / max(1, len(self._examples)),
            "avg_output_length": sum(len(ex.output) for ex in self._examples) / max(1, len(self._examples))
        }


# Singleton
_generator: Optional[TrainingDataGenerator] = None


def get_data_generator() -> TrainingDataGenerator:
    """Get training data generator."""
    global _generator
    if _generator is None:
        _generator = TrainingDataGenerator()
    return _generator
