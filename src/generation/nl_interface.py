# src/generation/nl_interface.py
"""
a6on-i - Natural Language Interface v1.0
Phase 6.4: Query Understanding, Text-to-SQL, Response Formatting

Features:
- Intent classification (query, report, action, status, comparison)
- Entity extraction (study, site, patient, issue type, date range)
- Text-to-SQL generation for data queries
- Response formatting with context
- RAG integration for knowledge-enhanced responses
- Conversational context management
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config.settings import DATA_PROCESSED

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class QueryIntent(Enum):
    """Types of user query intents."""
    STATUS = "status"
    COUNT = "count"
    LIST = "list"
    COMPARE = "compare"
    TREND = "trend"
    WHY = "why"
    WHEN = "when"
    HOW = "how"
    REPORT = "report"
    ACTION = "action"
    SEARCH = "search"
    HELP = "help"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    STUDY = "study"
    SITE = "site"
    PATIENT = "patient"
    REGION = "region"
    COUNTRY = "country"
    ISSUE_TYPE = "issue_type"
    METRIC = "metric"
    DATE = "date"
    DATE_RANGE = "date_range"
    NUMBER = "number"
    PERCENTAGE = "percentage"
    ROLE = "role"
    STATUS = "status"
    PRIORITY = "priority"


@dataclass
class ExtractedEntity:
    """An extracted entity from the query."""
    entity_type: EntityType
    value: Any
    original_text: str
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ParsedQuery:
    """Result of parsing a natural language query."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    intent_confidence: float
    entities: List[ExtractedEntity]
    keywords: List[str]
    is_compound: bool = False
    sub_intents: List[QueryIntent] = field(default_factory=list)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[ExtractedEntity]:
        return [e for e in self.entities if e.entity_type == entity_type]
    
    def get_first_entity(self, entity_type: EntityType) -> Optional[ExtractedEntity]:
        entities = self.get_entities_by_type(entity_type)
        return entities[0] if entities else None


@dataclass
class SQLQuery:
    """Generated SQL query with metadata."""
    sql: str
    parameters: Dict[str, Any]
    tables_used: List[str]
    columns_selected: List[str]
    explanation: str
    confidence: float = 1.0


@dataclass
class QueryResponse:
    """Response to a natural language query."""
    query_id: str
    original_query: str
    parsed: ParsedQuery
    sql_query: Optional[SQLQuery]
    data: Optional[pd.DataFrame]
    formatted_response: str
    summary: str
    visualizations: List[Dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    processing_time_ms: float = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'query_id': self.query_id,
            'original_query': self.original_query,
            'intent': self.parsed.intent.value,
            'entities': [
                {'type': e.entity_type.value, 'value': e.value, 'confidence': e.confidence}
                for e in self.parsed.entities
            ],
            'summary': self.summary,
            'formatted_response': self.formatted_response,
            'row_count': len(self.data) if self.data is not None else 0,
            'suggestions': self.suggestions,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

class IntentClassifier:
    """Classifies the intent of a natural language query."""
    
    INTENT_PATTERNS = {
        QueryIntent.STATUS: {
            'keywords': ['status', 'state', 'current', 'now', 'today', 'overview', 'summary'],
            'phrases': ['what is', 'what are', 'how is', 'show me the status', 'current state'],
            'weight': 1.0
        },
        QueryIntent.COUNT: {
            'keywords': ['how many', 'count', 'number of', 'total', 'quantity'],
            'phrases': ['how many', 'count of', 'number of', 'total number'],
            'weight': 1.2
        },
        QueryIntent.LIST: {
            'keywords': ['list', 'show', 'display', 'get', 'retrieve', 'find', 'all'],
            'phrases': ['show me', 'list all', 'display all', 'get all', 'find all', 'which'],
            'weight': 0.9
        },
        QueryIntent.COMPARE: {
            'keywords': ['compare', 'versus', 'vs', 'difference', 'between', 'comparison'],
            'phrases': ['compare', 'difference between', 'how does', 'versus', 'vs'],
            'weight': 1.3
        },
        QueryIntent.TREND: {
            'keywords': ['trend', 'over time', 'change', 'progress', 'history', 'evolution'],
            'phrases': ['how has', 'trend of', 'over time', 'changed', 'progressed'],
            'weight': 1.2
        },
        QueryIntent.WHY: {
            'keywords': ['why', 'reason', 'cause', 'root cause', 'because', 'explain'],
            'phrases': ['why is', 'why are', 'what caused', 'reason for', 'root cause'],
            'weight': 1.3
        },
        QueryIntent.WHEN: {
            'keywords': ['when', 'timeline', 'date', 'deadline', 'forecast', 'predict'],
            'phrases': ['when will', 'by when', 'timeline for', 'expected date', 'forecast'],
            'weight': 1.2
        },
        QueryIntent.HOW: {
            'keywords': ['how to', 'steps', 'process', 'resolve', 'fix', 'address', 'action'],
            'phrases': ['how do i', 'how to', 'steps to', 'what should', 'how can'],
            'weight': 1.2
        },
        QueryIntent.REPORT: {
            'keywords': ['report', 'generate', 'create report', 'document', 'pdf', 'export'],
            'phrases': ['generate report', 'create report', 'export', 'prepare report'],
            'weight': 1.4
        },
        QueryIntent.ACTION: {
            'keywords': ['send', 'notify', 'email', 'escalate', 'assign', 'create task'],
            'phrases': ['send email', 'notify', 'escalate to', 'create task', 'assign to'],
            'weight': 1.3
        },
        QueryIntent.SEARCH: {
            'keywords': ['search', 'find', 'look for', 'locate', 'where'],
            'phrases': ['search for', 'find', 'look for', 'where is', 'locate'],
            'weight': 0.8
        },
        QueryIntent.HELP: {
            'keywords': ['help', 'what can', 'capabilities', 'options', 'commands'],
            'phrases': ['help me', 'what can you', 'how do i use', 'what are my options'],
            'weight': 1.5
        }
    }
    
    def classify(self, query: str) -> Tuple[QueryIntent, float, List[QueryIntent]]:
        query_lower = query.lower().strip()
        scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1.0 * patterns['weight']
            
            for phrase in patterns['phrases']:
                if phrase in query_lower:
                    score += 2.0 * patterns['weight']
            
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return QueryIntent.UNKNOWN, 0.3, []
        
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_intent = sorted_intents[0][0]
        max_score = sorted_intents[0][1]
        
        confidence = min(max_score / 5.0, 1.0)
        
        sub_intents = []
        if len(sorted_intents) > 1:
            threshold = max_score * 0.5
            sub_intents = [intent for intent, score in sorted_intents[1:] if score >= threshold]
        
        return primary_intent, confidence, sub_intents


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """Extracts entities from natural language queries."""
    
    ISSUE_TYPES = {
        'sdv': 'sdv_incomplete',
        'source data verification': 'sdv_incomplete',
        'query': 'open_queries',
        'queries': 'open_queries',
        'open query': 'open_queries',
        'open queries': 'open_queries',
        'signature': 'signature_gaps',
        'signatures': 'signature_gaps',
        'unsigned': 'signature_gaps',
        'broken signature': 'broken_signatures',
        'broken signatures': 'broken_signatures',
        'sae': 'sae_dm_pending',
        'sae dm': 'sae_dm_pending',
        'sae safety': 'sae_safety_pending',
        'adverse event': 'sae_dm_pending',
        'missing visit': 'missing_visits',
        'missing visits': 'missing_visits',
        'missing page': 'missing_pages',
        'missing pages': 'missing_pages',
        'coding': 'meddra_uncoded',
        'uncoded': 'meddra_uncoded',
        'meddra': 'meddra_uncoded',
        'whodrug': 'whodrug_uncoded',
        'lab': 'lab_issues',
        'lab issues': 'lab_issues',
        'edrr': 'edrr_issues',
        'inactivated': 'inactivated_forms',
        'inactivated forms': 'inactivated_forms'
    }
    
    METRICS = {
        'dqi': 'dqi_score',
        'data quality': 'dqi_score',
        'data quality index': 'dqi_score',
        'clean': 'tier2_clean',
        'clean rate': 'tier2_clean',
        'clean patient': 'tier2_clean',
        'tier 1': 'tier1_clean',
        'tier 2': 'tier2_clean',
        'db lock': 'dblock_ready',
        'database lock': 'dblock_ready',
        'db lock ready': 'dblock_ready',
        'patients': 'patient_count',
        'patient count': 'patient_count',
        'issues': 'issue_count',
        'issue count': 'issue_count'
    }
    
    ROLES = {
        'cra': 'CRA',
        'clinical research associate': 'CRA',
        'dm': 'Data Manager',
        'data manager': 'Data Manager',
        'safety': 'Safety Data Manager',
        'safety dm': 'Safety Data Manager',
        'physician': 'Safety Physician',
        'medical': 'Safety Physician',
        'coder': 'Medical Coder',
        'coding team': 'Medical Coder',
        'site': 'Site',
        'site coordinator': 'Site Coordinator',
        'pi': 'Principal Investigator',
        'principal investigator': 'Principal Investigator',
        'study lead': 'Study Lead',
        'ctm': 'CTM'
    }
    
    STATUSES = {
        'ongoing': 'Ongoing',
        'active': 'Ongoing',
        'completed': 'Completed',
        'finished': 'Completed',
        'discontinued': 'Discontinued',
        'terminated': 'Discontinued',
        'screening': 'Screening',
        'screen failure': 'Screen Failure'
    }
    
    PRIORITIES = {
        'critical': 'Critical',
        'urgent': 'Critical',
        'high': 'High',
        'important': 'High',
        'medium': 'Medium',
        'normal': 'Medium',
        'low': 'Low',
        'minor': 'Low'
    }
    
    def extract(self, query: str) -> List[ExtractedEntity]:
        entities = []
        query_lower = query.lower()
        
        entities.extend(self._extract_studies(query))
        entities.extend(self._extract_sites(query))
        entities.extend(self._extract_patients(query))
        entities.extend(self._extract_issue_types(query_lower))
        entities.extend(self._extract_metrics(query_lower))
        entities.extend(self._extract_roles(query_lower))
        entities.extend(self._extract_statuses(query_lower))
        entities.extend(self._extract_priorities(query_lower))
        entities.extend(self._extract_numbers(query))
        entities.extend(self._extract_dates(query_lower))
        entities.extend(self._extract_regions(query_lower))
        
        return entities
    
    def _extract_studies(self, query: str) -> List[ExtractedEntity]:
        entities = []
        patterns = [r'study[_\-\s]?(\d+)', r'STUDY[_\-\s]?(\d+)']
        
        for pattern in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.STUDY,
                    value=f"Study_{match.group(1)}",
                    original_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return entities
    
    def _extract_sites(self, query: str) -> List[ExtractedEntity]:
        entities = []
        patterns = [r'site[_\-\s]?(\d+)', r'SITE[_\-\s]?(\d+)']
        
        for pattern in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SITE,
                    value=f"Site_{match.group(1)}",
                    original_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return entities
    
    def _extract_patients(self, query: str) -> List[ExtractedEntity]:
        entities = []
        
        pattern = r'(Study_\d+\|Site_\d+\|Subject_\d+)'
        for match in re.finditer(pattern, query):
            entities.append(ExtractedEntity(
                entity_type=EntityType.PATIENT,
                value=match.group(1),
                original_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        pattern2 = r'(?:patient|subject)[_\-\s]?(\d+)'
        for match in re.finditer(pattern2, query, re.IGNORECASE):
            entities.append(ExtractedEntity(
                entity_type=EntityType.PATIENT,
                value=f"Subject_{match.group(1)}",
                original_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.7
            ))
        
        return entities
    
    def _extract_issue_types(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        for keyword, issue_type in self.ISSUE_TYPES.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.ISSUE_TYPE,
                    value=issue_type,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_metrics(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        for keyword, metric in self.METRICS.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.METRIC,
                    value=metric,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_roles(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        for keyword, role in self.ROLES.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.ROLE,
                    value=role,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_statuses(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        for keyword, status in self.STATUSES.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.STATUS,
                    value=status,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_priorities(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        for keyword, priority in self.PRIORITIES.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PRIORITY,
                    value=priority,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities
    
    def _extract_numbers(self, query: str) -> List[ExtractedEntity]:
        entities = []
        
        for match in re.finditer(r'(\d+(?:\.\d+)?)\s*%', query):
            entities.append(ExtractedEntity(
                entity_type=EntityType.PERCENTAGE,
                value=float(match.group(1)),
                original_text=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        for match in re.finditer(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', query):
            full_match = match.group(0)
            end_pos = match.end()
            if end_pos < len(query) and query[end_pos:end_pos+1] != '%':
                entities.append(ExtractedEntity(
                    entity_type=EntityType.NUMBER,
                    value=float(match.group(1).replace(',', '')),
                    original_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.6
                ))
        
        return entities
    
    def _extract_dates(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        relative_dates = {
            'today': datetime.now().date(),
            'yesterday': (datetime.now() - timedelta(days=1)).date(),
            'tomorrow': (datetime.now() + timedelta(days=1)).date(),
            'this week': datetime.now().date(),
            'last week': (datetime.now() - timedelta(weeks=1)).date(),
            'next week': (datetime.now() + timedelta(weeks=1)).date(),
            'this month': datetime.now().date(),
            'last month': (datetime.now() - timedelta(days=30)).date(),
            'next month': (datetime.now() + timedelta(days=30)).date()
        }
        
        for keyword, date_value in relative_dates.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.DATE,
                    value=date_value,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        date_patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{2}/\d{2}/\d{4})', '%m/%d/%Y'),
            (r'(\d{2}-\d{2}-\d{4})', '%m-%d-%Y')
        ]
        
        for pattern, fmt in date_patterns:
            for match in re.finditer(pattern, query_lower):
                try:
                    date_value = datetime.strptime(match.group(1), fmt).date()
                    entities.append(ExtractedEntity(
                        entity_type=EntityType.DATE,
                        value=date_value,
                        original_text=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
                except ValueError:
                    pass
        
        return entities
    
    def _extract_regions(self, query_lower: str) -> List[ExtractedEntity]:
        entities = []
        
        regions = {
            'asia': 'ASIA',
            'europe': 'EUROPE',
            'eu': 'EUROPE',
            'latam': 'LATAM',
            'latin america': 'LATAM',
            'north america': 'NORTH_AMERICA',
            'na': 'NORTH_AMERICA',
            'apac': 'APAC'
        }
        
        countries = {
            'usa': 'USA',
            'us': 'USA',
            'united states': 'USA',
            'japan': 'Japan',
            'jp': 'Japan',
            'germany': 'Germany',
            'de': 'Germany',
            'uk': 'UK',
            'united kingdom': 'UK',
            'france': 'France',
            'brazil': 'Brazil',
            'china': 'China',
            'india': 'India'
        }
        
        for keyword, region in regions.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.REGION,
                    value=region,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        for keyword, country in countries.items():
            if keyword in query_lower:
                pos = query_lower.find(keyword)
                entities.append(ExtractedEntity(
                    entity_type=EntityType.COUNTRY,
                    value=country,
                    original_text=keyword,
                    start_pos=pos,
                    end_pos=pos + len(keyword)
                ))
        
        return entities


# =============================================================================
# SQL GENERATOR
# =============================================================================

class SQLGenerator:
    """Generates SQL queries from parsed natural language queries."""
    
    TABLES = {
        'patients': {
            'file': 'upr/unified_patient_record.parquet',
            'columns': [
                'patient_key', 'study_id', 'site_id', 'subject_id',
                'subject_status', 'region', 'country',
                'total_open_queries', 'dm_queries', 'clinical_queries',
                'total_crfs', 'crfs_signed', 'crfs_never_signed',
                'total_crfs_requiring_sdv', 'crfs_source_data_verified',
                'broken_signatures'
            ],
            'key': 'patient_key'
        },
        'issues': {
            'file': 'analytics/patient_issues.parquet',
            'columns': [
                'patient_key', 'study_id', 'site_id',
                'issue_sdv_incomplete', 'issue_open_queries', 'issue_signature_gaps',
                'issue_broken_signatures', 'issue_sae_dm_pending', 'issue_sae_safety_pending',
                'issue_missing_visits', 'issue_missing_pages', 'issue_lab_issues',
                'issue_edrr_issues', 'issue_inactivated_forms',
                'issue_meddra_uncoded', 'issue_whodrug_uncoded',
                'total_issues', 'highest_priority', 'primary_issue'
            ],
            'key': 'patient_key'
        },
        'dqi': {
            'file': 'analytics/patient_dqi_enhanced.parquet',
            'columns': [
                'patient_key', 'study_id', 'site_id',
                'dqi_score', 'dqi_band',
                'safety_score', 'query_score', 'completeness_score',
                'coding_score', 'lab_score', 'sdv_score',
                'signature_score', 'edrr_score',
                'primary_issue_type'
            ],
            'key': 'patient_key'
        },
        'clean_status': {
            'file': 'analytics/patient_clean_status.parquet',
            'columns': [
                'patient_key', 'study_id', 'site_id',
                'tier1_clean', 'tier2_clean',
                'blocking_reasons', 'effort_to_clean_hours',
                'quick_win_category'
            ],
            'key': 'patient_key'
        },
        'dblock': {
            'file': 'analytics/patient_dblock_status.parquet',
            'columns': [
                'patient_key', 'study_id', 'site_id',
                'dblock_eligible', 'dblock_status', 'dblock_ready',
                'days_to_ready', 'blocking_reason'
            ],
            'key': 'patient_key'
        },
        'sites': {
            'file': 'analytics/site_benchmarks.parquet',
            'columns': [
                'site_id', 'study_id', 'patient_count',
                'composite_score', 'performance_tier', 'percentile',
                'dqi_mean', 'tier2_clean_rate', 'dblock_ready_rate'
            ],
            'key': 'site_id'
        }
    }
    
    def generate(self, parsed: ParsedQuery) -> Optional[SQLQuery]:
        intent = parsed.intent
        entities = parsed.entities
        
        table = self._determine_table(parsed)
        where_clause = self._build_where_clause(entities)
        
        if intent == QueryIntent.COUNT:
            return self._generate_count_query(parsed, table, where_clause)
        elif intent == QueryIntent.LIST:
            return self._generate_list_query(parsed, table, where_clause)
        elif intent == QueryIntent.STATUS:
            return self._generate_status_query(parsed, table, where_clause)
        elif intent == QueryIntent.COMPARE:
            return self._generate_compare_query(parsed, table, where_clause)
        else:
            return self._generate_summary_query(parsed, table, where_clause)
    
    def _determine_table(self, parsed: ParsedQuery) -> str:
        issue_entities = parsed.get_entities_by_type(EntityType.ISSUE_TYPE)
        metric_entities = parsed.get_entities_by_type(EntityType.METRIC)
        site_entities = parsed.get_entities_by_type(EntityType.SITE)
        
        if issue_entities:
            return 'issues'
        
        if metric_entities:
            metric = metric_entities[0].value
            if 'dqi' in metric:
                return 'dqi'
            elif 'clean' in metric:
                return 'clean_status'
            elif 'dblock' in metric:
                return 'dblock'
        
        if 'site' in parsed.original_query.lower() and not site_entities:
            return 'sites'
        
        return 'patients'
    
    def _build_where_clause(self, entities: List[ExtractedEntity]) -> str:
        conditions = []
        
        for entity in entities:
            if entity.entity_type == EntityType.STUDY:
                conditions.append("study_id = '{}'".format(entity.value))
            elif entity.entity_type == EntityType.SITE:
                conditions.append("site_id = '{}'".format(entity.value))
            elif entity.entity_type == EntityType.PATIENT:
                conditions.append("patient_key = '{}'".format(entity.value))
            elif entity.entity_type == EntityType.STATUS:
                conditions.append("subject_status = '{}'".format(entity.value))
            elif entity.entity_type == EntityType.ISSUE_TYPE:
                col = "issue_{}".format(entity.value) if not entity.value.startswith('issue_') else entity.value
                conditions.append("{} = true".format(col))
        
        if conditions:
            return "WHERE " + " AND ".join(conditions)
        return ""
    
    def _generate_count_query(self, parsed: ParsedQuery, table: str, where_clause: str) -> SQLQuery:
        group_entities = []
        for entity in parsed.entities:
            if entity.entity_type in [EntityType.STUDY, EntityType.SITE, EntityType.STATUS]:
                if entity.entity_type == EntityType.STUDY and 'study' in parsed.original_query.lower():
                    group_entities.append('study_id')
                elif entity.entity_type == EntityType.SITE and 'site' in parsed.original_query.lower():
                    group_entities.append('site_id')
        
        if 'by study' in parsed.original_query.lower() or 'per study' in parsed.original_query.lower():
            group_entities.append('study_id')
        if 'by site' in parsed.original_query.lower() or 'per site' in parsed.original_query.lower():
            group_entities.append('site_id')
        
        if group_entities:
            group_by = ', '.join(set(group_entities))
            sql = """
                SELECT {group_by}, COUNT(*) as count
                FROM {{table}}
                {where_clause}
                GROUP BY {group_by}
                ORDER BY count DESC
            """.format(group_by=group_by, where_clause=where_clause)
        else:
            sql = """
                SELECT COUNT(*) as count
                FROM {{table}}
                {where_clause}
            """.format(where_clause=where_clause)
        
        return SQLQuery(
            sql=sql.replace('{table}', '{' + table + '}'),
            parameters={'table': table},
            tables_used=[table],
            columns_selected=['count'] + (list(set(group_entities)) if group_entities else []),
            explanation="Count of records from {}".format(table) + (" grouped by {}".format(', '.join(set(group_entities))) if group_entities else ""),
            confidence=0.9
        )
    
    def _generate_list_query(self, parsed: ParsedQuery, table: str, where_clause: str) -> SQLQuery:
        table_info = self.TABLES.get(table, self.TABLES['patients'])
        columns = table_info['columns'][:10]
        
        order_by = table_info['key']
        if parsed.get_entities_by_type(EntityType.METRIC):
            metric = parsed.get_first_entity(EntityType.METRIC).value
            if metric in columns:
                order_by = "{} DESC".format(metric)
        
        sql = """
            SELECT {columns}
            FROM {{table}}
            {where_clause}
            ORDER BY {order_by}
            LIMIT 100
        """.format(columns=', '.join(columns), where_clause=where_clause, order_by=order_by)
        
        return SQLQuery(
            sql=sql.replace('{table}', '{' + table + '}'),
            parameters={'table': table},
            tables_used=[table],
            columns_selected=columns,
            explanation="List of records from {}".format(table),
            confidence=0.85
        )
    
    def _generate_status_query(self, parsed: ParsedQuery, table: str, where_clause: str) -> SQLQuery:
        sql = """
            SELECT 
                COUNT(*) as total_count,
                COUNT(DISTINCT site_id) as site_count,
                COUNT(DISTINCT study_id) as study_count
            FROM {{table}}
            {where_clause}
        """.format(where_clause=where_clause)
        
        return SQLQuery(
            sql=sql.replace('{table}', '{' + table + '}'),
            parameters={'table': table},
            tables_used=[table],
            columns_selected=['total_count', 'site_count', 'study_count'],
            explanation="Status summary from {}".format(table),
            confidence=0.8
        )
    
    def _generate_compare_query(self, parsed: ParsedQuery, table: str, where_clause: str) -> SQLQuery:
        studies = parsed.get_entities_by_type(EntityType.STUDY)
        sites = parsed.get_entities_by_type(EntityType.SITE)
        
        if len(studies) >= 2:
            group_by = 'study_id'
            filter_values = [s.value for s in studies]
            quoted_values = ", ".join(["'{}'".format(v) for v in filter_values])
            where_clause = "WHERE study_id IN ({})".format(quoted_values)
        elif len(sites) >= 2:
            group_by = 'site_id'
            filter_values = [s.value for s in sites]
            quoted_values = ", ".join(["'{}'".format(v) for v in filter_values])
            where_clause = "WHERE site_id IN ({})".format(quoted_values)
        else:
            group_by = 'study_id'
        
        sql = """
            SELECT 
                {group_by},
                COUNT(*) as patient_count,
                AVG(CASE WHEN dqi_score IS NOT NULL THEN dqi_score ELSE 0 END) as avg_dqi,
                SUM(CASE WHEN tier2_clean THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as clean_rate
            FROM {{table}}
            {where_clause}
            GROUP BY {group_by}
            ORDER BY patient_count DESC
        """.format(group_by=group_by, where_clause=where_clause)
        
        return SQLQuery(
            sql=sql.replace('{table}', '{' + table + '}'),
            parameters={'table': table},
            tables_used=[table],
            columns_selected=[group_by, 'patient_count', 'avg_dqi', 'clean_rate'],
            explanation="Comparison by {}".format(group_by),
            confidence=0.85
        )
    
    def _generate_summary_query(self, parsed: ParsedQuery, table: str, where_clause: str) -> SQLQuery:
        sql = """
            SELECT 
                COUNT(*) as total_patients,
                COUNT(DISTINCT site_id) as total_sites,
                COUNT(DISTINCT study_id) as total_studies
            FROM {{table}}
            {where_clause}
        """.format(where_clause=where_clause)
        
        return SQLQuery(
            sql=sql.replace('{table}', '{' + table + '}'),
            parameters={'table': table},
            tables_used=[table],
            columns_selected=['total_patients', 'total_sites', 'total_studies'],
            explanation="General summary",
            confidence=0.7
        )


# =============================================================================
# DATA EXECUTOR
# =============================================================================

class DataExecutor:
    """Executes queries against the data files."""
    
    def __init__(self):
        self.data_path = Path(DATA_PROCESSED)
        self.cache = {}
        self._load_data()
    
    def _load_data(self):
        self.tables = {}
        
        table_files = {
            'patients': 'upr/unified_patient_record.parquet',
            'issues': 'analytics/patient_issues.parquet',
            'dqi': 'analytics/patient_dqi_enhanced.parquet',
            'clean_status': 'analytics/patient_clean_status.parquet',
            'dblock': 'analytics/patient_dblock_status.parquet',
            'sites': 'analytics/site_benchmarks.parquet'
        }
        
        for table_name, file_path in table_files.items():
            full_path = self.data_path / file_path
            if full_path.exists():
                try:
                    self.tables[table_name] = pd.read_parquet(full_path)
                except Exception as e:
                    print("Warning: Could not load {}: {}".format(table_name, e))
    
    def execute(self, sql_query: SQLQuery) -> pd.DataFrame:
        table_name = sql_query.parameters.get('table', 'patients')
        
        if table_name not in self.tables:
            return pd.DataFrame()
        
        df = self.tables[table_name].copy()
        
        sql = sql_query.sql
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            df = self._apply_where(df, where_clause)
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP BY\s+([^\s]+(?:\s*,\s*[^\s]+)*)', sql, re.IGNORECASE)
        if group_match:
            group_cols = [c.strip() for c in group_match.group(1).split(',')]
            valid_group_cols = [c for c in group_cols if c in df.columns]
            if valid_group_cols:
                agg_funcs = self._extract_aggregations(sql)
                if agg_funcs:
                    df = df.groupby(valid_group_cols, as_index=False).agg(agg_funcs)
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER BY\s+([^\s]+)\s*(DESC|ASC)?', sql, re.IGNORECASE)
        if order_match:
            order_col = order_match.group(1).strip()
            ascending = order_match.group(2) is None or order_match.group(2).upper() == 'ASC'
            if order_col in df.columns:
                df = df.sort_values(order_col, ascending=ascending)
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))
            df = df.head(limit)
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            if select_clause != '*':
                if 'COUNT(*)' in select_clause.upper():
                    return pd.DataFrame({'count': [len(df)]})
        
        return df
    
    def _apply_where(self, df: pd.DataFrame, where_clause: str) -> pd.DataFrame:
        conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for condition in conditions:
            condition = condition.strip()
            
            # Handle = comparison
            eq_match = re.match(r"(\w+)\s*=\s*'([^']+)'", condition)
            if eq_match:
                col, val = eq_match.groups()
                if col in df.columns:
                    df = df[df[col] == val]
                continue
            
            # Handle IN comparison
            in_match = re.match(r"(\w+)\s+IN\s*\(([^)]+)\)", condition, re.IGNORECASE)
            if in_match:
                col = in_match.group(1)
                values = [v.strip().strip("'\"") for v in in_match.group(2).split(',')]
                if col in df.columns:
                    df = df[df[col].isin(values)]
                continue
            
            # Handle = true/false for boolean
            bool_match = re.match(r"(\w+)\s*=\s*(true|false)", condition, re.IGNORECASE)
            if bool_match:
                col, val = bool_match.groups()
                if col in df.columns:
                    df = df[df[col] == (val.lower() == 'true')]
                continue
        
        return df
    
    def _extract_aggregations(self, sql: str) -> Dict[str, str]:
        agg_funcs = {}
        
        if 'COUNT(*)' in sql.upper():
            agg_funcs['patient_key'] = 'count'
        
        for match in re.finditer(r'AVG\((\w+)\)', sql, re.IGNORECASE):
            col = match.group(1)
            agg_funcs[col] = 'mean'
        
        for match in re.finditer(r'SUM\(([^)]+)\)', sql, re.IGNORECASE):
            expr = match.group(1)
            if 'CASE' not in expr.upper():
                agg_funcs[expr] = 'sum'
        
        for match in re.finditer(r'MIN\((\w+)\)', sql, re.IGNORECASE):
            agg_funcs[match.group(1)] = 'min'
        for match in re.finditer(r'MAX\((\w+)\)', sql, re.IGNORECASE):
            agg_funcs[match.group(1)] = 'max'
        
        return agg_funcs if agg_funcs else {'patient_key': 'count'}
    
    def get_quick_stats(self) -> Dict[str, Any]:
        stats = {}
        
        if 'patients' in self.tables:
            df = self.tables['patients']
            stats['total_patients'] = len(df)
            stats['total_studies'] = df['study_id'].nunique() if 'study_id' in df.columns else 0
            stats['total_sites'] = df['site_id'].nunique() if 'site_id' in df.columns else 0
        
        if 'dqi' in self.tables:
            df = self.tables['dqi']
            stats['mean_dqi'] = df['dqi_score'].mean() if 'dqi_score' in df.columns else 0
        
        if 'clean_status' in self.tables:
            df = self.tables['clean_status']
            if 'tier2_clean' in df.columns:
                stats['clean_rate'] = (df['tier2_clean'].sum() / len(df) * 100) if len(df) > 0 else 0
        
        if 'issues' in self.tables:
            df = self.tables['issues']
            if 'total_issues' in df.columns:
                stats['total_issues'] = df['total_issues'].sum()
                stats['patients_with_issues'] = (df['total_issues'] > 0).sum()
        
        return stats


# =============================================================================
# RESPONSE FORMATTER
# =============================================================================

class ResponseFormatter:
    """Formats query results into human-readable responses."""
    
    def format(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        sql_query: Optional[SQLQuery],
        context: Dict[str, Any] = None
    ) -> Tuple[str, str, List[Dict]]:
        context = context or {}
        
        if data is None or len(data) == 0:
            return self._format_no_results(parsed), "No results found.", []
        
        if parsed.intent == QueryIntent.COUNT:
            return self._format_count(parsed, data, context)
        elif parsed.intent == QueryIntent.LIST:
            return self._format_list(parsed, data, context)
        elif parsed.intent == QueryIntent.STATUS:
            return self._format_status(parsed, data, context)
        elif parsed.intent == QueryIntent.COMPARE:
            return self._format_compare(parsed, data, context)
        elif parsed.intent == QueryIntent.HELP:
            return self._format_help()
        else:
            return self._format_general(parsed, data, context)
    
    def _format_no_results(self, parsed: ParsedQuery) -> str:
        return """
**No Results Found**

Your query "{}" did not return any results.

**Possible reasons:**
- The specified entities (study, site, patient) may not exist
- The filter criteria may be too restrictive
- The data may not be available for the requested timeframe

**Suggestions:**
- Try broadening your search criteria
- Check the entity IDs for typos
- Use "help" to see available queries
""".format(parsed.original_query)
    
    def _format_count(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        context: Dict
    ) -> Tuple[str, str, List[Dict]]:
        
        if 'count' in data.columns and len(data) == 1:
            count = int(data['count'].iloc[0])
            summary = "Found {:,} records".format(count)
            
            response = """
**Count Result**

📊 **{:,}** records found

""".format(count)
            if 'total_patients' in context:
                pct = (count / context['total_patients'] * 100) if context['total_patients'] > 0 else 0
                response += "This represents **{:.1f}%** of total patients ({:,})\n".format(pct, context['total_patients'])
            
            return response, summary, []
        
        elif 'count' in data.columns and len(data) > 1:
            total = int(data['count'].sum())
            summary = "Found {:,} records across {} groups".format(total, len(data))
            
            group_col = [c for c in data.columns if c != 'count'][0] if len(data.columns) > 1 else 'group'
            
            response = """
**Count by {}**

📊 **Total: {:,}** records

| {} | Count | % of Total |
|---|---:|---:|
""".format(group_col.replace('_', ' ').title(), total, group_col.replace('_', ' ').title())
            
            for _, row in data.head(20).iterrows():
                pct = (row['count'] / total * 100) if total > 0 else 0
                response += "| {} | {:,} | {:.1f}% |\n".format(row[group_col], int(row['count']), pct)
            
            if len(data) > 20:
                response += "\n*Showing top 20 of {} groups*\n".format(len(data))
            
            viz = [{
                'type': 'bar',
                'title': 'Count by {}'.format(group_col),
                'x': group_col,
                'y': 'count',
                'data': data.head(10).to_dict('records')
            }]
            
            return response, summary, viz
        
        return self._format_general(parsed, data, context)
    
    def _format_list(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        context: Dict
    ) -> Tuple[str, str, List[Dict]]:
        
        summary = "Found {:,} records".format(len(data))
        
        display_cols = data.columns[:8].tolist()
        display_data = data[display_cols].head(20)
        
        response = """
**Results: {:,} records**

""".format(len(data))
        
        response += "| " + " | ".join(display_cols) + " |\n"
        response += "|" + "|".join(["---"] * len(display_cols)) + "|\n"
        
        for _, row in display_data.iterrows():
            values = [str(row[col])[:30] for col in display_cols]
            response += "| " + " | ".join(values) + " |\n"
        
        if len(data) > 20:
            response += "\n*Showing first 20 of {:,} records*\n".format(len(data))
        
        return response, summary, []
    
    def _format_status(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        context: Dict
    ) -> Tuple[str, str, List[Dict]]:
        
        metrics = {}
        for col in data.columns:
            if len(data) == 1:
                metrics[col] = data[col].iloc[0]
        
        summary = "Status retrieved with {} metrics".format(len(metrics))
        
        response = """
**Status Summary**

"""
        for key, value in metrics.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, float):
                formatted_value = "{:,.2f}".format(value)
            elif isinstance(value, (int, np.integer)):
                formatted_value = "{:,}".format(value)
            else:
                formatted_value = str(value)
            
            response += "- **{}**: {}\n".format(formatted_key, formatted_value)
        
        return response, summary, []
    
    def _format_compare(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        context: Dict
    ) -> Tuple[str, str, List[Dict]]:
        
        summary = "Comparison across {} groups".format(len(data))
        
        group_col = data.columns[0]
        metric_cols = data.columns[1:].tolist()
        
        response = """
**Comparison Results**

Comparing by **{}**:

| {} | """.format(group_col.replace('_', ' ').title(), group_col.replace('_', ' ').title())
        response += " | ".join([c.replace('_', ' ').title() for c in metric_cols]) + " |\n"
        response += "|" + "|".join(["---"] * (len(metric_cols) + 1)) + "|\n"
        
        for _, row in data.iterrows():
            values = [str(row[group_col])]
            for col in metric_cols:
                val = row[col]
                if isinstance(val, float):
                    values.append("{:,.2f}".format(val))
                elif isinstance(val, (int, np.integer)):
                    values.append("{:,}".format(val))
                else:
                    values.append(str(val))
            response += "| " + " | ".join(values) + " |\n"
        
        viz = [{
            'type': 'bar',
            'title': 'Comparison by {}'.format(group_col),
            'x': group_col,
            'y': metric_cols[0] if metric_cols else 'count',
            'data': data.to_dict('records')
        }]
        
        return response, summary, viz
    
    def _format_help(self) -> Tuple[str, str, List[Dict]]:
        response = """
**a6on-i - Natural Language Interface**

I can help you query clinical trial data using natural language. Here are some examples:

**📊 Counts & Statistics**
- "How many patients are in Study_21?"
- "Count patients by study"
- "How many sites have open queries?"

**📋 Lists & Details**
- "Show me all patients at Site_1"
- "List sites with DQI below 90"
- "Find patients with SAE pending"

**📈 Comparisons**
- "Compare Study_21 vs Study_22"
- "Compare site performance across regions"

**❓ Status Queries**
- "What is the current status of Study_21?"
- "Show me the DB lock readiness"
- "What's the clean rate?"

**🔍 Issue Queries**
- "Which sites have the most open queries?"
- "Show me patients with SDV incomplete"
- "What are the top issues?"

**📝 Reports**
- "Generate executive brief for Study_21"
- "Create site performance report"

**💡 Tips**
- Include study or site IDs for specific results (e.g., Study_21, Site_101)
- Use keywords like "by study" or "by site" for grouping
- Ask "why" for root cause analysis
- Ask "when" for timeline predictions
"""
        
        summary = "Help information displayed"
        return response, summary, []
    
    def _format_general(
        self,
        parsed: ParsedQuery,
        data: pd.DataFrame,
        context: Dict
    ) -> Tuple[str, str, List[Dict]]:
        
        summary = "Found {:,} records with {} columns".format(len(data), len(data.columns))
        
        cols_display = ', '.join(data.columns[:10].tolist())
        if len(data.columns) > 10:
            cols_display += "..."
        
        response = """
**Query Results**

Found **{:,}** records.

**Data Overview:**
- Columns: {}
- Rows: {:,}

""".format(len(data), cols_display, len(data))
        
        if len(data) <= 10:
            response += "**Sample Data:**\n\n"
            response += data.to_markdown(index=False) + "\n"
        else:
            response += "**First 5 Records:**\n\n"
            response += data.head(5).to_markdown(index=False) + "\n"
        
        return response, summary, []


# =============================================================================
# CONTEXT INJECTOR
# =============================================================================

class ContextInjector:
    """Injects context from RAG and other sources."""
    
    def __init__(self):
        self.rag_available = False
        self._init_rag()
    
    def _init_rag(self):
        try:
            from src.knowledge.rag_knowledge_base import get_rag_knowledge_base
            self.rag = get_rag_knowledge_base()
            self.rag_available = True
        except Exception:
            self.rag = None
            self.rag_available = False
    
    def get_context(self, parsed: ParsedQuery) -> Dict[str, Any]:
        context = {}
        
        if self.rag_available and self.rag:
            try:
                results = self.rag.search(parsed.original_query, n_results=3)
                if results:
                    context['knowledge'] = [
                        {
                            'text': r.get('text', ''),
                            'source': r.get('metadata', {}).get('source', 'unknown'),
                            'score': r.get('score', 0)
                        }
                        for r in results
                    ]
            except Exception:
                pass
        
        issue_types = parsed.get_entities_by_type(EntityType.ISSUE_TYPE)
        if issue_types:
            context['issue_descriptions'] = {
                'sdv_incomplete': 'Source Data Verification not completed',
                'open_queries': 'Data queries requiring resolution',
                'signature_gaps': 'Missing or overdue electronic signatures',
                'sae_dm_pending': 'SAE reconciliation pending with Data Management',
                'missing_visits': 'Scheduled visits not completed',
                'missing_pages': 'CRF pages not completed'
            }
        
        return context


# =============================================================================
# CONVERSATION MANAGER
# =============================================================================

class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    def add_turn(self, query: str, response: QueryResponse):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': response.parsed.intent.value,
            'entities': [
                {'type': e.entity_type.value, 'value': e.value}
                for e in response.parsed.entities
            ],
            'summary': response.summary
        })
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_from_history(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        
        recent_entities = {}
        for turn in reversed(self.history[-3:]):
            for entity in turn.get('entities', []):
                if entity['type'] not in recent_entities:
                    recent_entities[entity['type']] = entity['value']
        
        return {
            'session_id': self.session_id,
            'turn_count': len(self.history),
            'recent_entities': recent_entities,
            'last_intent': self.history[-1]['intent'] if self.history else None
        }
    
    def clear(self):
        self.history = []


# =============================================================================
# MAIN NL INTERFACE
# =============================================================================

class NaturalLanguageInterface:
    """Main natural language interface for querying clinical trial data."""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sql_generator = SQLGenerator()
        self.data_executor = DataExecutor()
        self.response_formatter = ResponseFormatter()
        self.context_injector = ContextInjector()
        self.conversation = ConversationManager()
        
        self.stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
    
    def query(self, user_query: str) -> QueryResponse:
        import time

        # PostgreSQL integration
        from src.database.pg_data_service import get_data_service
        from src.database.pg_writer import get_pg_writer

        start_time = time.time()
        
        query_id = "Q-{}-{:04d}".format(
            datetime.now().strftime('%Y%m%d%H%M%S'),
            self.stats['queries_processed']
        )
        self.stats['queries_processed'] += 1
        
        try:
            parsed = self._parse_query(user_query)
            
            sql_query = None
            if parsed.intent not in [QueryIntent.HELP, QueryIntent.UNKNOWN]:
                sql_query = self.sql_generator.generate(parsed)
            
            data = None
            if sql_query:
                data = self.data_executor.execute(sql_query)
            
            context = self.context_injector.get_context(parsed)
            context.update(self.data_executor.get_quick_stats())
            context.update(self.conversation.get_context_from_history())
            
            formatted, summary, visualizations = self.response_formatter.format(
                parsed, data, sql_query, context
            )
            
            suggestions = self._generate_suggestions(parsed, data)
            
            confidence = self._calculate_confidence(parsed, sql_query, data)
            
            elapsed = (time.time() - start_time) * 1000
            
            response = QueryResponse(
                query_id=query_id,
                original_query=user_query,
                parsed=parsed,
                sql_query=sql_query,
                data=data,
                formatted_response=formatted,
                summary=summary,
                visualizations=visualizations,
                suggestions=suggestions,
                confidence=confidence,
                processing_time_ms=elapsed
            )
            
            self.conversation.add_turn(user_query, response)
            self.stats['successful_queries'] += 1
            
            return response
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.stats['failed_queries'] += 1
            
            return QueryResponse(
                query_id=query_id,
                original_query=user_query,
                parsed=ParsedQuery(
                    original_query=user_query,
                    normalized_query=user_query.lower(),
                    intent=QueryIntent.UNKNOWN,
                    intent_confidence=0,
                    entities=[],
                    keywords=[]
                ),
                sql_query=None,
                data=None,
                formatted_response="Error processing query: {}".format(str(e)),
                summary="Query failed",
                confidence=0,
                processing_time_ms=elapsed,
                error=str(e)
            )
    
    def _parse_query(self, query: str) -> ParsedQuery:
        normalized = query.lower().strip()
        
        intent, confidence, sub_intents = self.intent_classifier.classify(query)
        
        entities = self.entity_extractor.extract(query)
        
        keywords = self._extract_keywords(normalized)
        
        return ParsedQuery(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            keywords=keywords,
            is_compound=len(sub_intents) > 0,
            sub_intents=sub_intents
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if',
            'because', 'until', 'while', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my',
            'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
            'them', 'their', 'theirs', 'themselves', 'show', 'tell',
            'give', 'get', 'find', 'make'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))
    
    def _generate_suggestions(self, parsed: ParsedQuery, data: Optional[pd.DataFrame]) -> List[str]:
        suggestions = []
        
        if parsed.intent == QueryIntent.COUNT:
            suggestions.append("Try grouping by study or site for more detail")
            suggestions.append("Ask 'why' to understand the root causes")
        elif parsed.intent == QueryIntent.LIST:
            suggestions.append("Use 'compare' to see differences between groups")
            suggestions.append("Ask about specific issue types for filtered results")
        elif parsed.intent == QueryIntent.STATUS:
            suggestions.append("Ask 'when will we be ready for DB lock?' for timeline")
            suggestions.append("Generate a report for stakeholder updates")
        
        if parsed.get_entities_by_type(EntityType.STUDY):
            study = parsed.get_first_entity(EntityType.STUDY).value
            suggestions.append("Compare {} with other studies".format(study))
            suggestions.append("Show sites in {}".format(study))
        
        if parsed.get_entities_by_type(EntityType.SITE):
            site = parsed.get_first_entity(EntityType.SITE).value
            suggestions.append("List patients at {}".format(site))
            suggestions.append("Generate site performance report for {}".format(site))
        
        if not suggestions:
            suggestions = [
                "Try asking about specific studies or sites",
                "Ask 'help' to see available commands",
                "Use 'compare' to analyze differences"
            ]
        
        return suggestions[:3]
    
    def _calculate_confidence(
        self,
        parsed: ParsedQuery,
        sql_query: Optional[SQLQuery],
        data: Optional[pd.DataFrame]
    ) -> float:
        scores = []
        
        scores.append(parsed.intent_confidence)
        
        if parsed.entities:
            entity_conf = sum(e.confidence for e in parsed.entities) / len(parsed.entities)
            scores.append(entity_conf)
        
        if sql_query:
            scores.append(sql_query.confidence)
        
        if data is not None and len(data) > 0:
            scores.append(1.0)
        elif data is not None:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'session_id': self.conversation.session_id,
            'conversation_turns': len(self.conversation.history),
            'rag_available': self.context_injector.rag_available
        }
    
    def clear_conversation(self):
        self.conversation.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_nl_interface = None

def get_nl_interface() -> NaturalLanguageInterface:
    global _nl_interface
    if _nl_interface is None:
        _nl_interface = NaturalLanguageInterface()
    return _nl_interface


def ask(query: str) -> QueryResponse:
    return get_nl_interface().query(query)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("a6on-i - Natural Language Interface v1.0")
    print("=" * 60)
    
    nl = get_nl_interface()
    
    print("\n📊 Data loaded:")
    stats = nl.data_executor.get_quick_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print("   {}: {:.2f}".format(key, value))
        else:
            print("   {}: {:,}".format(key, value))
    
    print("\n🔍 RAG available: {}".format(nl.context_injector.rag_available))
    
    print("\n✅ Natural Language Interface Ready")
    print("\nTry: ask('How many patients are in Study_21?')")
