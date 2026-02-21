# src/generation/__init__.py
"""
SANCHALAK AI - Generation Module
Phase 6: Generative AI Document Engine

Components:
- 6.1 Template Engine (12 templates)
- 6.2 Report Generators (8 generators)
- 6.3 Export Engine (PDF/DOCX/PPTX)
- 6.4 Natural Language Interface
- 6.5 Auto-Summarization
"""

# =========================
# Template Engine
# =========================
from .template_engine import (
    get_template_engine,
    TemplateEngine,
    ReportTemplate,
    GeneratedReport
)

# =========================
# Report Generators
# =========================
from .report_generators import (
    generate_report,
    ReportGeneratorFactory,
    OutputFormat,
    ReportOutput
)

# =========================
# Export Engine
# =========================
from .export_engine import (
    get_export_engine,
    export_to_pdf,
    export_to_docx,
    export_to_pptx,
    ExportEngine,
    ExportResult,
    StyleConfig
)

# =========================
# Natural Language Interface
# =========================
from .nl_interface import (
    get_nl_interface,
    ask,
    NaturalLanguageInterface,
    QueryResponse,
    QueryIntent,
    EntityType,
    ParsedQuery
)

# =========================
# Auto-Summarization Engine
# =========================
from .auto_summarizer import (
    get_auto_summarizer,
    summarize_patient,
    summarize_site,
    daily_digest,
    executive_summary,
    AutoSummarizer,
    Summary,
    SummaryType,
    SummaryFormat,
    Finding,
    ActionItem,
    Severity
)

# =========================
# Public API
# =========================
__all__ = [
    # Template Engine
    'get_template_engine',
    'TemplateEngine',
    'ReportTemplate',
    'GeneratedReport',

    # Report Generators
    'generate_report',
    'ReportGeneratorFactory',
    'OutputFormat',
    'ReportOutput',

    # Export Engine
    'get_export_engine',
    'export_to_pdf',
    'export_to_docx',
    'export_to_pptx',
    'ExportEngine',
    'ExportResult',
    'StyleConfig',

    # Natural Language Interface
    'get_nl_interface',
    'ask',
    'NaturalLanguageInterface',
    'QueryResponse',
    'QueryIntent',
    'EntityType',
    'ParsedQuery',

    # Auto-Summarizer
    'get_auto_summarizer',
    'summarize_patient',
    'summarize_site',
    'daily_digest',
    'executive_summary',
    'AutoSummarizer',
    'Summary',
    'SummaryType',
    'SummaryFormat',
    'Finding',
    'ActionItem',
    'Severity'
]
