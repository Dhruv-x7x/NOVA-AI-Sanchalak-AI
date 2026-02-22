# File: src/knowledge/documents/__init__.py
"""
Documents module for RAG Knowledge Base.
Contains ICH-GCP guidelines, protocol knowledge, and SOPs.
"""

from .ich_gcp_guidelines import (
    ICH_GCP_GUIDELINES,
    get_ich_gcp_guidelines,
    get_section,
    get_all_sections,
    search_guidelines
)

from .protocol_knowledge import (
    PROTOCOL_KNOWLEDGE_BASE,
    get_protocol_knowledge,
    get_protocol_section,
    search_protocol_knowledge
)

from .sop_documents import (
    SOP_DOCUMENTS,
    get_all_sops,
    get_sop_by_id,
    search_sops,
    get_sops_by_category,
    get_sops_by_department,
    get_sop_summary
)

__all__ = [
    # ICH-GCP
    'ICH_GCP_GUIDELINES',
    'get_ich_gcp_guidelines',
    'get_section',
    'get_all_sections',
    'search_guidelines',
    # Protocol Knowledge
    'PROTOCOL_KNOWLEDGE_BASE',
    'get_protocol_knowledge',
    'get_protocol_section',
    'search_protocol_knowledge',
    # SOPs
    'SOP_DOCUMENTS',
    'get_all_sops',
    'get_sop_by_id',
    'search_sops',
    'get_sops_by_category',
    'get_sops_by_department',
    'get_sop_summary'
]
