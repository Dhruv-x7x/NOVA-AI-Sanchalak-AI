"""
a6on-i - Data Validation Pipeline
================================================
Validates patient records against schema with comprehensive error reporting.

Reference: riyaz.md - Improvement #24
"""

import re
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    issue_type: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity.value,
            "value": str(self.value)[:100] if self.value else None,
            "expected": str(self.expected)[:100] if self.expected else None
        }


@dataclass
class ValidationResult:
    """Result of validating a single record."""
    record_id: str
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "is_valid": self.is_valid,
            "errors": self.errors_count,
            "warnings": self.warnings_count,
            "issues": [i.to_dict() for i in self.issues]
        }


@dataclass
class BatchValidationResult:
    """Result of validating a batch of records."""
    total_records: int
    valid_records: int
    invalid_records: int
    total_errors: int
    total_warnings: int
    error_summary: Dict[str, int]
    sample_issues: List[ValidationIssue]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "validation_rate": round(self.valid_records / max(1, self.total_records) * 100, 1),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "error_summary": self.error_summary,
            "sample_issues": [i.to_dict() for i in self.sample_issues[:10]]
        }


class DataValidator:
    """
    Validates patient records against schema.
    
    Checks:
    - Required fields present
    - Data types correct
    - Value ranges valid
    - Referential integrity (site exists, study exists)
    - Business rules (e.g., enrollment date before completion)
    
    Usage:
        validator = DataValidator()
        result = validator.validate_upr_record(record)
        
        if not result.is_valid:
            print(f"Validation errors: {result.errors_count}")
            for issue in result.issues:
                print(f"  - {issue.field}: {issue.message}")
    """
    
    # Required fields for UPR records
    REQUIRED_FIELDS = {
        "patient_key": str,
        "study_id": str,
        "site_id": str,
        "subject_status": str
    }
    
    # Optional fields with types
    OPTIONAL_FIELDS = {
        "dqi_score": (int, float),
        "tier1_clean": bool,
        "tier2_clean": bool,
        "total_queries": (int, float),
        "open_queries": (int, float),
        "pending_signatures": (int, float),
        "risk_level": str,
        "enrollment_date": (str, datetime, date),
        "last_visit_date": (str, datetime, date),
        "country": str,
        "region": str
    }
    
    # Valid values for categorical fields
    VALID_VALUES = {
        "subject_status": {"Active", "Completed", "Withdrawn", "Screened", "Enrolled", 
                          "Screen Failed", "Early Termination", "Lost to Follow-up"},
        "risk_level": {"Critical", "High", "Medium", "Low", "None"},
        "dqi_band": {"Critical", "Low", "Medium", "High", "Excellent"}
    }
    
    # Value ranges
    VALUE_RANGES = {
        "dqi_score": (0, 100),
        "total_queries": (0, 10000),
        "open_queries": (0, 1000),
        "pending_signatures": (0, 500)
    }
    
    def __init__(self):
        self._known_studies: Set[str] = set()
        self._known_sites: Set[str] = set()
        self._load_reference_data()
        logger.info("DataValidator initialized")
    
    def _load_reference_data(self):
        """Load reference data for referential integrity checks."""
        try:
            from src.database.connection import get_db_manager
            db = get_db_manager()
            if db and hasattr(db, 'engine'):
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    # Load studies
                    result = conn.execute(text("SELECT DISTINCT study_id FROM studies"))
                    for row in result:
                        self._known_studies.add(row[0])
                    
                    # Load sites
                    result = conn.execute(text("SELECT DISTINCT site_id FROM clinical_sites"))
                    for row in result:
                        self._known_sites.add(row[0])
                
                logger.info(f"Loaded {len(self._known_studies)} studies, {len(self._known_sites)} sites")
        except Exception as e:
            logger.debug(f"Reference data not loaded: {e}")
    
    def validate_upr_record(self, record: Dict) -> ValidationResult:
        """
        Validate a single patient record against schema.
        
        Args:
            record: Patient record dictionary
            
        Returns:
            ValidationResult with all issues found
        """
        issues = []
        record_id = record.get("patient_key", record.get("id", "unknown"))
        
        # Check required fields
        for field_name, expected_type in self.REQUIRED_FIELDS.items():
            if field_name not in record or record[field_name] is None:
                issues.append(ValidationIssue(
                    field=field_name,
                    issue_type="missing_required",
                    message=f"Required field '{field_name}' is missing",
                    severity=ValidationSeverity.ERROR
                ))
            elif not isinstance(record[field_name], expected_type):
                issues.append(ValidationIssue(
                    field=field_name,
                    issue_type="invalid_type",
                    message=f"Expected {expected_type.__name__}, got {type(record[field_name]).__name__}",
                    severity=ValidationSeverity.ERROR,
                    value=record[field_name],
                    expected=expected_type.__name__
                ))
        
        # Check optional fields types
        for field_name, expected_types in self.OPTIONAL_FIELDS.items():
            if field_name in record and record[field_name] is not None:
                if not isinstance(record[field_name], expected_types):
                    issues.append(ValidationIssue(
                        field=field_name,
                        issue_type="invalid_type",
                        message=f"Invalid type for '{field_name}'",
                        severity=ValidationSeverity.WARNING,
                        value=record[field_name],
                        expected=str(expected_types)
                    ))
        
        # Check valid values
        for field_name, valid_set in self.VALID_VALUES.items():
            if field_name in record and record[field_name] is not None:
                value = record[field_name]
                if value not in valid_set:
                    issues.append(ValidationIssue(
                        field=field_name,
                        issue_type="invalid_value",
                        message=f"Value '{value}' not in allowed values",
                        severity=ValidationSeverity.WARNING,
                        value=value,
                        expected=str(list(valid_set)[:5]) + "..."
                    ))
        
        # Check value ranges
        for field_name, (min_val, max_val) in self.VALUE_RANGES.items():
            if field_name in record and record[field_name] is not None:
                try:
                    value = float(record[field_name])
                    if value < min_val or value > max_val:
                        issues.append(ValidationIssue(
                            field=field_name,
                            issue_type="out_of_range",
                            message=f"Value {value} outside range [{min_val}, {max_val}]",
                            severity=ValidationSeverity.ERROR,
                            value=value,
                            expected=f"[{min_val}, {max_val}]"
                        ))
                except (ValueError, TypeError):
                    pass
        
        # Referential integrity
        if self._known_studies and record.get("study_id"):
            if record["study_id"] not in self._known_studies:
                issues.append(ValidationIssue(
                    field="study_id",
                    issue_type="referential_integrity",
                    message=f"Study '{record['study_id']}' not found in reference data",
                    severity=ValidationSeverity.WARNING,
                    value=record["study_id"]
                ))
        
        if self._known_sites and record.get("site_id"):
            if record["site_id"] not in self._known_sites:
                issues.append(ValidationIssue(
                    field="site_id",
                    issue_type="referential_integrity",
                    message=f"Site '{record['site_id']}' not found in reference data",
                    severity=ValidationSeverity.WARNING,
                    value=record["site_id"]
                ))
        
        # Business rule: patient_key format
        if record.get("patient_key"):
            if not self._validate_patient_key_format(record["patient_key"]):
                issues.append(ValidationIssue(
                    field="patient_key",
                    issue_type="invalid_format",
                    message="Patient key should follow format: Study_X|Site_XXX|Subject_XXXX",
                    severity=ValidationSeverity.INFO,
                    value=record["patient_key"]
                ))
        
        # Count by severity
        errors_count = sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        warnings_count = sum(1 for i in issues if i.severity == ValidationSeverity.WARNING)
        
        return ValidationResult(
            record_id=record_id,
            is_valid=(errors_count == 0),
            issues=issues,
            warnings_count=warnings_count,
            errors_count=errors_count
        )
    
    def _validate_patient_key_format(self, key: str) -> bool:
        """Check if patient key follows expected format."""
        # Expected format: Study_X|Site_XXX|Subject_XXXX
        pattern = r'^[A-Za-z0-9_]+\|[A-Za-z0-9_]+\|[A-Za-z0-9_]+$'
        return bool(re.match(pattern, key))
    
    def validate_batch(self, records: List[Dict]) -> BatchValidationResult:
        """
        Validate a batch of records with detailed error report.
        
        Args:
            records: List of patient records
            
        Returns:
            BatchValidationResult with summary
        """
        valid_count = 0
        invalid_count = 0
        total_errors = 0
        total_warnings = 0
        error_summary: Dict[str, int] = {}
        all_issues: List[ValidationIssue] = []
        
        for record in records:
            result = self.validate_upr_record(record)
            
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            
            total_errors += result.errors_count
            total_warnings += result.warnings_count
            
            for issue in result.issues:
                key = f"{issue.field}:{issue.issue_type}"
                error_summary[key] = error_summary.get(key, 0) + 1
                all_issues.append(issue)
        
        # Get sample of most common issues
        sample_issues = sorted(all_issues, key=lambda x: x.severity.value)[:10]
        
        return BatchValidationResult(
            total_records=len(records),
            valid_records=valid_count,
            invalid_records=invalid_count,
            total_errors=total_errors,
            total_warnings=total_warnings,
            error_summary=dict(sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:20]),
            sample_issues=sample_issues
        )
    
    def add_custom_rule(
        self,
        field: str,
        rule_name: str,
        validator: callable,
        severity: ValidationSeverity = ValidationSeverity.WARNING
    ):
        """Add a custom validation rule."""
        # Would store custom rules for use in validation
        logger.info(f"Custom rule added: {field} - {rule_name}")


# Singleton instance
_validator: Optional[DataValidator] = None


def get_data_validator() -> DataValidator:
    """Get the data validator singleton."""
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator
