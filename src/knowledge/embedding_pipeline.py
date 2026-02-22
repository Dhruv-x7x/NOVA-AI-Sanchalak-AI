# src/knowledge/embedding_pipeline.py
"""
a6on-i - Embedding Pipeline v1.3
Phase 4.1: Generate and manage embeddings for all knowledge artifacts

FIXED v1.3: Fixed duplicate ID issue in issue_descriptions
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Manages embedding generation for all a6on-i knowledge artifacts."""
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        output_dir: Optional[Path] = None,
        cache_embeddings: bool = True
    ):
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        if output_dir is None:
            # Try to find in project root first
            try:
                project_root = Path(__file__).parent.parent.parent
                root_output_path = project_root / "data" / "processed" / "knowledge" / "embeddings"
                if root_output_path.exists():
                    self.output_dir = root_output_path
                else:
                    self.output_dir = Path("data/processed/knowledge/embeddings")
            except Exception:
                self.output_dir = Path("data/processed/knowledge/embeddings")
        else:
            self.output_dir = Path(output_dir)
            
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        self.embeddings: Dict[str, Dict] = {}
        self.stats = {
            "total_documents": 0,
            "total_embeddings": 0,
            "categories_processed": [],
            "processing_time": 0
        }
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, 
                                  convert_to_numpy=True, normalize_embeddings=True)
    
    def _compute_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    # =========================================================================
    # CATEGORY 1: RESOLUTION TEMPLATES
    # =========================================================================
    
    def embed_resolution_templates(self, templates_path: Optional[Path] = None) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING RESOLUTION TEMPLATES")
        logger.info("=" * 60)
        
        if templates_path is None:
            templates_path = Path("data/processed/analytics/resolution_genome/resolution_templates.json")
        
        templates = []
        
        if templates_path.exists():
            try:
                with open(templates_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # Handle nested dict structure: {'SDV-001': {...}, 'QRY-001': {...}}
                if isinstance(loaded_data, dict):
                    for key, value in loaded_data.items():
                        if isinstance(value, dict):
                            value['template_id'] = value.get('template_id', key)
                            templates.append(value)
                elif isinstance(loaded_data, list):
                    templates = [t for t in loaded_data if isinstance(t, dict)]
                
                logger.info(f"Loaded {len(templates)} templates from file")
            except Exception as e:
                logger.warning(f"Error loading templates: {e}")
        
        # Supplement with defaults if needed
        if len(templates) < 10:
            logger.info(f"Supplementing with default templates")
            defaults = self._get_default_resolution_templates()
            existing_types = {t.get('issue_type', '') for t in templates}
            for dt in defaults:
                if dt['issue_type'] not in existing_types:
                    templates.append(dt)
            logger.info(f"Total templates: {len(templates)}")
        
        texts, metadata = [], []
        for i, template in enumerate(templates):
            text = self._create_resolution_text(template)
            texts.append(text)
            metadata.append({
                "id": template.get("template_id", f"RES-{i+1:03d}"),
                "issue_type": template.get("issue_type", "unknown"),
                "title": template.get("title", "")[:50],
                "responsible_role": template.get("responsible_role", template.get("role", "")),
                "category": "resolution_templates",
                "text_hash": self._compute_text_hash(text)
            })
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["resolution_templates"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("resolution_templates")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} resolution template embeddings")
        return self.embeddings["resolution_templates"]
    
    def _create_resolution_text(self, template: Dict) -> str:
        parts = []
        for key in ['issue_type', 'title', 'description', 'responsible_role', 'role']:
            if template.get(key):
                parts.append(f"{key}: {template[key]}")
        if template.get('steps'):
            steps = template['steps']
            if isinstance(steps, list):
                parts.append(f"Steps: {' '.join(steps)}")
        if template.get('keywords'):
            keywords = template['keywords']
            if isinstance(keywords, list):
                parts.append(f"Keywords: {' '.join(keywords)}")
        return " | ".join(parts) if parts else str(template)
    
    def _get_default_resolution_templates(self) -> List[Dict]:
        return [
            {"template_id": "RES-001", "issue_type": "sdv_incomplete", "title": "Complete Source Data Verification", 
             "description": "Review and verify all CRF data against source documents", 
             "steps": ["Access source documents", "Compare CRF entries", "Document discrepancies", "Mark SDV complete"],
             "responsible_role": "CRA", "effort_hours": 0.5, "keywords": ["SDV", "source data", "verification"]},
            {"template_id": "RES-002", "issue_type": "open_queries", "title": "Resolve Open Data Query",
             "description": "Review query, investigate discrepancy, provide response",
             "steps": ["Review query", "Check source", "Correct or explain", "Close query"],
             "responsible_role": "Site", "effort_hours": 0.25, "keywords": ["query", "discrepancy", "resolution"]},
            {"template_id": "RES-003", "issue_type": "signature_gaps", "title": "Complete Required Signatures",
             "description": "Obtain missing PI or coordinator signatures",
             "steps": ["Identify unsigned CRFs", "Review data", "Obtain signatures", "Verify"],
             "responsible_role": "Site", "effort_hours": 0.1, "keywords": ["signature", "PI", "sign-off"]},
            {"template_id": "RES-004", "issue_type": "sae_dm_pending", "title": "Complete SAE DM Review",
             "description": "Review SAE data for completeness and reconciliation",
             "steps": ["Access SAE case", "Verify fields", "Reconcile with EDC", "Complete review"],
             "responsible_role": "Safety Data Manager", "effort_hours": 0.3, "keywords": ["SAE", "safety", "reconciliation"]},
            {"template_id": "RES-005", "issue_type": "sae_safety_pending", "title": "Complete SAE Safety Review",
             "description": "Medical review for causality assessment",
             "steps": ["Review case", "Assess causality", "Document assessment", "Complete"],
             "responsible_role": "Safety Physician", "effort_hours": 0.5, "keywords": ["SAE", "causality", "medical"]},
            {"template_id": "RES-006", "issue_type": "missing_visits", "title": "Complete Missing Visit Data",
             "description": "Enter or verify data for missing protocol visits",
             "steps": ["Identify missing", "Check if occurred", "Enter or mark missed", "Document"],
             "responsible_role": "Site", "effort_hours": 0.5, "keywords": ["visit", "missing", "protocol"]},
            {"template_id": "RES-007", "issue_type": "missing_pages", "title": "Complete Missing CRF Pages",
             "description": "Enter data for missing CRF pages",
             "steps": ["Identify missing", "Gather source docs", "Enter data", "Submit"],
             "responsible_role": "Site", "effort_hours": 0.3, "keywords": ["CRF", "pages", "missing"]},
            {"template_id": "RES-008", "issue_type": "meddra_uncoded", "title": "Code to MedDRA",
             "description": "Apply MedDRA codes to verbatim terms",
             "steps": ["Review term", "Search MedDRA", "Select code", "Apply"],
             "responsible_role": "Medical Coder", "effort_hours": 0.05, "keywords": ["MedDRA", "coding", "AE"]},
            {"template_id": "RES-009", "issue_type": "whodrug_uncoded", "title": "Code to WHODrug",
             "description": "Apply WHODrug codes to medications",
             "steps": ["Review medication", "Search WHODrug", "Select code", "Apply"],
             "responsible_role": "Medical Coder", "effort_hours": 0.05, "keywords": ["WHODrug", "coding", "medication"]},
            {"template_id": "RES-010", "issue_type": "lab_issues", "title": "Resolve Lab Issues",
             "description": "Address missing lab values or reference ranges",
             "steps": ["Identify issue", "Contact lab", "Enter/update", "Verify"],
             "responsible_role": "Site", "effort_hours": 0.4, "keywords": ["lab", "reference range", "values"]},
            {"template_id": "RES-011", "issue_type": "edrr_issues", "title": "Resolve EDRR Issues",
             "description": "Address third-party data reconciliation",
             "steps": ["Review discrepancy", "Compare sources", "Correct", "Close"],
             "responsible_role": "Data Manager", "effort_hours": 0.3, "keywords": ["EDRR", "reconciliation", "external"]},
            {"template_id": "RES-012", "issue_type": "broken_signatures", "title": "Re-sign Broken Signatures",
             "description": "Obtain new signatures after data changes",
             "steps": ["Identify broken", "Review changes", "Re-sign", "Verify"],
             "responsible_role": "Site", "effort_hours": 0.1, "keywords": ["signature", "broken", "re-sign"]},
            {"template_id": "RES-013", "issue_type": "inactivated_forms", "title": "Review Inactivated Forms",
             "description": "Review and document inactivated CRF forms",
             "steps": ["Identify forms", "Review reason", "Document", "Close"],
             "responsible_role": "Data Manager", "effort_hours": 0.2, "keywords": ["inactivated", "forms", "review"]},
            {"template_id": "RES-014", "issue_type": "high_query_volume", "title": "Address High Query Volume",
             "description": "Site training and data quality improvement",
             "steps": ["Analyze patterns", "Schedule training", "Monitor improvement"],
             "responsible_role": "CRA", "effort_hours": 2.0, "keywords": ["queries", "training", "quality"]}
        ]
    
    # =========================================================================
    # CATEGORY 2: ISSUE DESCRIPTIONS (FIXED - unique IDs)
    # =========================================================================
    
    def embed_issue_descriptions(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING ISSUE DESCRIPTIONS")
        logger.info("=" * 60)
        
        issues = [
            {"issue_type": "sdv_incomplete", "title": "SDV Incomplete", "description": "CRF data not verified against source", "responsible": "CRA", "priority": "High"},
            {"issue_type": "open_queries", "title": "Open Queries", "description": "Data discrepancies requiring response", "responsible": "Site", "priority": "High"},
            {"issue_type": "signature_gaps", "title": "Missing Signatures", "description": "Required signatures not obtained", "responsible": "Site", "priority": "Medium"},
            {"issue_type": "broken_signatures", "title": "Broken Signatures", "description": "Signatures invalidated by data changes", "responsible": "Site", "priority": "Medium"},
            {"issue_type": "sae_dm_pending", "title": "SAE DM Pending", "description": "SAE requires data management review", "responsible": "Safety DM", "priority": "Critical"},
            {"issue_type": "sae_safety_pending", "title": "SAE Safety Pending", "description": "SAE awaiting medical review", "responsible": "Safety Physician", "priority": "Critical"},
            {"issue_type": "missing_visits", "title": "Missing Visits", "description": "Protocol visits not recorded", "responsible": "Site", "priority": "High"},
            {"issue_type": "missing_pages", "title": "Missing Pages", "description": "CRF pages not completed", "responsible": "Site", "priority": "High"},
            {"issue_type": "lab_issues", "title": "Lab Issues", "description": "Missing lab values or ranges", "responsible": "Site", "priority": "Medium"},
            {"issue_type": "edrr_issues", "title": "EDRR Issues", "description": "Third-party data discrepancies", "responsible": "Data Manager", "priority": "Medium"},
            {"issue_type": "inactivated_forms", "title": "Inactivated Forms", "description": "Deactivated CRF forms", "responsible": "Data Manager", "priority": "Low"},
            {"issue_type": "meddra_uncoded", "title": "MedDRA Uncoded", "description": "Terms not coded to MedDRA", "responsible": "Medical Coder", "priority": "Medium"},
            {"issue_type": "whodrug_uncoded", "title": "WHODrug Uncoded", "description": "Medications not coded", "responsible": "Medical Coder", "priority": "Medium"},
            {"issue_type": "high_query_volume", "title": "High Query Volume", "description": "Unusually high queries at site", "responsible": "CRA", "priority": "High"}
        ]
        
        texts, metadata = [], []
        for i, issue in enumerate(issues):
            text = f"Issue: {issue['title']}. Type: {issue['issue_type']}. {issue['description']}. Responsible: {issue['responsible']}. Priority: {issue['priority']}."
            texts.append(text)
            # FIXED: Use full issue_type as ID to avoid truncation duplicates
            metadata.append({
                "id": f"ISS-{issue['issue_type'].upper()}",
                "issue_type": issue['issue_type'], 
                "title": issue['title'], 
                "responsible": issue['responsible'], 
                "priority": issue['priority'],
                "category": "issue_descriptions", 
                "text_hash": self._compute_text_hash(text)
            })
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["issue_descriptions"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("issue_descriptions")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} issue description embeddings")
        return self.embeddings["issue_descriptions"]
    
    # =========================================================================
    # CATEGORY 3: SOP DOCUMENTS
    # =========================================================================
    
    def embed_sop_documents(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING SOP DOCUMENTS")
        logger.info("=" * 60)
        
        sops = [
            {"sop_id": "SOP-DM-001", "title": "Data Management Plan", "content": "Data entry within 3 days. Queries within 5 days. DB lock requires 100% resolution."},
            {"sop_id": "SOP-MON-001", "title": "Site Monitoring", "content": "100% SDV for primary endpoints and consent. Risk-based for other data."},
            {"sop_id": "SOP-SAE-001", "title": "Safety Reporting", "content": "SAEs within 24 hours. SUSARs to regulators within 7-15 days."},
            {"sop_id": "SOP-COD-001", "title": "Medical Coding", "content": "MedDRA coding within 7 days. WHODrug for all medications."},
            {"sop_id": "SOP-DBL-001", "title": "Database Lock", "content": "Pre-lock: queries resolved, coding complete, SDV complete, signatures obtained."},
            {"sop_id": "SOP-QRY-001", "title": "Query Management", "content": "Standard: 5 days. Critical: 24 hours. Safety: same day. Escalation at 7/14/21 days."}
        ]
        
        texts, metadata = [], []
        for sop in sops:
            text = f"SOP: {sop['title']}. ID: {sop['sop_id']}. {sop['content']}"
            texts.append(text)
            metadata.append({"id": sop['sop_id'], "title": sop['title'], "category": "sop_documents", "text_hash": self._compute_text_hash(text)})
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["sop_documents"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("sop_documents")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} SOP embeddings")
        return self.embeddings["sop_documents"]
    
    # =========================================================================
    # CATEGORY 4: QUERY TEMPLATES
    # =========================================================================
    
    def embed_query_templates(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING QUERY TEMPLATES")
        logger.info("=" * 60)
        
        queries = [
            {"id": "QRY-001", "cat": "missing_data", "title": "Missing Field", "template": "Required field [FIELD] is missing.", "priority": "high"},
            {"id": "QRY-002", "cat": "discrepancy", "title": "Date Discrepancy", "template": "Date [X] inconsistent with [Y].", "priority": "high"},
            {"id": "QRY-003", "cat": "out_of_range", "title": "Out of Range", "template": "Value [X] outside range [MIN-MAX].", "priority": "medium"},
            {"id": "QRY-004", "cat": "consent", "title": "Consent Issue", "template": "Consent must precede all procedures.", "priority": "critical"},
            {"id": "QRY-005", "cat": "ae_coding", "title": "AE Clarification", "template": "Term [X] needs clarification for coding.", "priority": "medium"},
            {"id": "QRY-006", "cat": "medication", "title": "Medication Clarification", "template": "Medication [X] cannot be coded.", "priority": "medium"},
            {"id": "QRY-007", "cat": "lab", "title": "Lab Confirmation", "template": "Lab result [X] is clinically significant.", "priority": "high"},
            {"id": "QRY-008", "cat": "protocol", "title": "Protocol Deviation", "template": "Potential deviation: [DESCRIPTION].", "priority": "high"},
            {"id": "QRY-009", "cat": "sae", "title": "SAE Information", "template": "Additional SAE information required.", "priority": "critical"},
            {"id": "QRY-010", "cat": "visit", "title": "Visit Window", "template": "Visit outside protocol window.", "priority": "medium"}
        ]
        
        texts, metadata = [], []
        for q in queries:
            text = f"Query: {q['title']}. Category: {q['cat']}. Template: {q['template']} Priority: {q['priority']}."
            texts.append(text)
            metadata.append({"id": q['id'], "query_category": q['cat'], "title": q['title'], "priority": q['priority'],
                           "category": "query_templates", "text_hash": self._compute_text_hash(text)})
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["query_templates"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("query_templates")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} query template embeddings")
        return self.embeddings["query_templates"]
    
    # =========================================================================
    # CATEGORY 5: PATTERN DESCRIPTIONS
    # =========================================================================
    
    def embed_pattern_descriptions(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING PATTERN DESCRIPTIONS")
        logger.info("=" * 60)
        
        patterns_path = Path("data/processed/analytics/pattern_library/pattern_definitions.csv")
        if patterns_path.exists():
            try:
                df = pd.read_csv(patterns_path)
                patterns = df.to_dict('records')
                logger.info(f"Loaded {len(patterns)} patterns from file")
            except:
                patterns = self._get_default_patterns()
        else:
            patterns = self._get_default_patterns()
        
        texts, metadata = [], []
        for i, p in enumerate(patterns):
            text = f"Pattern: {p.get('name', p.get('pattern_id'))}. {p.get('description', '')}. Severity: {p.get('severity', '')}."
            texts.append(text)
            metadata.append({
                "id": p.get('pattern_id', f"PAT-{i+1:03d}"),
                "name": p.get('name', ''),
                "severity": p.get('severity', ''),
                "category": "pattern_descriptions",
                "text_hash": self._compute_text_hash(text)
            })
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["pattern_descriptions"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("pattern_descriptions")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} pattern embeddings")
        return self.embeddings["pattern_descriptions"]
    
    def _get_default_patterns(self) -> List[Dict]:
        return [
            {"pattern_id": "PAT-DQ-001", "name": "Query Overload", "description": "High open queries", "severity": "High"},
            {"pattern_id": "PAT-RC-001", "name": "Coordinator Overload", "description": "Too many patients per coordinator", "severity": "High"},
            {"pattern_id": "PAT-SF-001", "name": "SAE Backlog", "description": "SAE reconciliation backlog", "severity": "Critical"},
            {"pattern_id": "PAT-TL-001", "name": "Timeline Risk", "description": "Unlikely to meet deadline", "severity": "High"}
        ]
    
    # =========================================================================
    # CATEGORY 6: REGULATORY GUIDELINES
    # =========================================================================
    
    def embed_regulatory_guidelines(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING REGULATORY GUIDELINES")
        logger.info("=" * 60)
        
        guidelines = [
            {"id": "ICH-E6-5.0", "title": "ICH E6 - Sponsor", "content": "Sponsor responsible for QA/QC systems."},
            {"id": "ICH-E6-5.5", "title": "ICH E6 - Electronic Data", "content": "Systems ensure data integrity and audit trails."},
            {"id": "ICH-E6-5.18", "title": "ICH E6 - Monitoring", "content": "Monitoring verifies subject rights and data accuracy."},
            {"id": "ICH-E6-5.18.4", "title": "ICH E6 - SDV", "content": "Monitor verifies source documents are accurate."},
            {"id": "21CFR11-10", "title": "21 CFR 11 - Controls", "content": "Systems ensure authenticity and integrity."},
            {"id": "21CFR11-10e", "title": "21 CFR 11 - Audit Trail", "content": "Time-stamped audit trails required."},
            {"id": "21CFR11-50", "title": "21 CFR 11 - Signatures", "content": "E-signatures include name, date, meaning."},
            {"id": "ICH-E9-4", "title": "ICH E9 - Design", "content": "Database management and QA essential."},
            {"id": "ICH-E9-5", "title": "ICH E9 - Conduct", "content": "DB lock procedures documented before unblinding."}
        ]
        
        texts, metadata = [], []
        for g in guidelines:
            text = f"Guideline: {g['title']}. ID: {g['id']}. {g['content']}"
            texts.append(text)
            metadata.append({"id": g['id'], "title": g['title'], "category": "regulatory_guidelines", "text_hash": self._compute_text_hash(text)})
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["regulatory_guidelines"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("regulatory_guidelines")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} regulatory embeddings")
        return self.embeddings["regulatory_guidelines"]
    
    # =========================================================================
    # CATEGORY 7: CLINICAL TERMS
    # =========================================================================
    
    def embed_clinical_terms(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("EMBEDDING CLINICAL TERMS")
        logger.info("=" * 60)
        
        terms = [
            {"term": "SDV", "full": "Source Data Verification", "def": "Comparing CRF with source documents"},
            {"term": "CRF", "full": "Case Report Form", "def": "Document recording subject data"},
            {"term": "EDC", "full": "Electronic Data Capture", "def": "System for electronic data collection"},
            {"term": "SAE", "full": "Serious Adverse Event", "def": "Event causing death, hospitalization, disability"},
            {"term": "SUSAR", "full": "Suspected Unexpected Serious Adverse Reaction", "def": "Unexpected serious reaction"},
            {"term": "ICF", "full": "Informed Consent Form", "def": "Document describing study and rights"},
            {"term": "PI", "full": "Principal Investigator", "def": "Person responsible for trial at site"},
            {"term": "CRA", "full": "Clinical Research Associate", "def": "Person monitoring trials"},
            {"term": "CTM", "full": "Clinical Trial Manager", "def": "Person managing trial operations"},
            {"term": "MedDRA", "full": "Medical Dictionary for Regulatory Activities", "def": "Standardized medical terminology"},
            {"term": "WHODrug", "full": "WHO Drug Dictionary", "def": "International drug classification"},
            {"term": "DB Lock", "full": "Database Lock", "def": "Final step before analysis"},
            {"term": "CDISC", "full": "Clinical Data Interchange Standards", "def": "Data standards organization"},
            {"term": "SDTM", "full": "Study Data Tabulation Model", "def": "CDISC submission standard"},
            {"term": "IVRS", "full": "Interactive Voice Response System", "def": "Randomization and supply system"}
        ]
        
        texts, metadata = [], []
        for t in terms:
            text = f"Term: {t['term']}. Full: {t['full']}. Definition: {t['def']}"
            texts.append(text)
            metadata.append({"id": f"TERM-{t['term']}", "term": t['term'], "full_name": t['full'],
                           "category": "clinical_terms", "text_hash": self._compute_text_hash(text)})
        
        embeddings = self.generate_embeddings(texts)
        self.embeddings["clinical_terms"] = {"embeddings": embeddings, "texts": texts, "metadata": metadata}
        self.stats["categories_processed"].append("clinical_terms")
        self.stats["total_documents"] += len(texts)
        self.stats["total_embeddings"] += len(embeddings)
        logger.info(f"Generated {len(embeddings)} clinical term embeddings")
        return self.embeddings["clinical_terms"]
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        start_time = datetime.now()
        
        logger.info("=" * 70)
        logger.info("a6on-i - EMBEDDING PIPELINE v1.3")
        logger.info("=" * 70)
        
        self.embed_resolution_templates()
        self.embed_issue_descriptions()
        self.embed_sop_documents()
        self.embed_query_templates()
        self.embed_pattern_descriptions()
        self.embed_regulatory_guidelines()
        self.embed_clinical_terms()
        
        self.stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        self.save_embeddings()
        
        logger.info("=" * 70)
        logger.info("EMBEDDING PIPELINE COMPLETE")
        logger.info(f"Total Documents: {self.stats['total_documents']}")
        logger.info(f"Total Embeddings: {self.stats['total_embeddings']}")
        logger.info(f"Processing Time: {self.stats['processing_time']:.2f}s")
        logger.info("=" * 70)
        
        return self.generate_summary()
    
    def save_embeddings(self):
        logger.info("Saving embeddings...")
        for cat, data in self.embeddings.items():
            np.save(self.output_dir / f"{cat}_embeddings.npy", data['embeddings'])
            with open(self.output_dir / f"{cat}_metadata.json", 'w') as f:
                json.dump(data['metadata'], f, indent=2)
            with open(self.output_dir / f"{cat}_texts.json", 'w') as f:
                json.dump(data['texts'], f, indent=2)
            logger.info(f"  Saved {cat}: {len(data['embeddings'])} embeddings")
        
        with open(self.output_dir / "embedding_index.json", 'w') as f:
            json.dump({"model": self.model_name, "dim": self.embedding_dim, "stats": self.stats,
                      "categories": {c: len(d['embeddings']) for c, d in self.embeddings.items()}}, f, indent=2)
    
    def generate_summary(self) -> Dict[str, Any]:
        summary = {"model": self.model_name, "total": self.stats["total_embeddings"],
                  "categories": {c: len(d['embeddings']) for c, d in self.embeddings.items()}}
        with open(self.output_dir / "embedding_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        return summary
    
    def search_similar(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        query_emb = self.model.encode(query, normalize_embeddings=True)
        results = []
        for cat in ([category] if category else self.embeddings.keys()):
            if cat not in self.embeddings:
                continue
            data = self.embeddings[cat]
            sims = np.dot(data['embeddings'], query_emb)
            for idx in np.argsort(sims)[::-1][:top_k]:
                results.append({"category": cat, "score": float(sims[idx]), "text": data['texts'][idx], "metadata": data['metadata'][idx]})
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]


def main():
    print("=" * 70)
    print("a6on-i - PHASE 4.1: EMBEDDING PIPELINE v1.3")
    print("=" * 70)
    
    pipeline = EmbeddingPipeline()
    summary = pipeline.run_full_pipeline()
    
    print("\n" + "=" * 70)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 70)
    
    for query in ["How do I complete SDV?", "SAE reconciliation process", "Missing data query"]:
        print(f"\nQuery: '{query}'")
        for i, r in enumerate(pipeline.search_similar(query, top_k=3), 1):
            print(f"  {i}. [{r['category']}] {r['score']:.3f} - {r['text'][:60]}...")
    
    print("\n" + "=" * 70)
    print("PHASE 4.1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
