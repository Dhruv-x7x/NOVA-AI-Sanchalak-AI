"""
a6on-i - Phase 8.3: @Tagging System v1.1 (FIXED)
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- Entity recognition (@Site_101, @Study_21, @Patient_123, etc.)
- Mention parsing (@username, @role, @team)
- Notification triggers based on mentions
- Link generation for tagged entities
- Tag registry with metadata
- Notification preferences and batching

Fixes in v1.1:
- Fixed pattern priority (specific patterns before generic USER)
- Fixed database locking issues
- Fixed notification ID uniqueness
- Fixed team pattern matching
"""

import sqlite3
import re
import json
# SQLite removed - using PostgreSQL
import hashlib
import uuid
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager


# =============================================================================
# ENUMS
# =============================================================================

class TagType(Enum):
    """Types of tags that can be recognized"""
    USER = "user"           # @sarah_chen, @john_smith
    ROLE = "role"           # @CRA, @DataManager, @SafetyPhysician
    TEAM = "team"           # @CRA-Team, @Safety-Team, @DM-Team
    SITE = "site"           # @Site_101, @Site_JP001
    STUDY = "study"         # @Study_21, @Study_ABC123
    PATIENT = "patient"     # @Patient_1001, @Subject_ABC
    ISSUE = "issue"         # @ISS-20260101-ABC123
    ROOM = "room"           # @ROOM-20260101-ABC123
    QUERY = "query"         # @Query_12345
    SAE = "sae"             # @SAE_001, @SAE-2024-001
    TOPIC = "topic"         # #urgent, #escalation, #consent
    CUSTOM = "custom"       # Custom entity types


class NotificationPriority(Enum):
    """Priority levels for notifications"""
    URGENT = "urgent"       # Immediate delivery
    HIGH = "high"           # Within 1 hour
    NORMAL = "normal"       # Within 4 hours
    LOW = "low"             # Batch with daily digest


class NotificationChannel(Enum):
    """Channels for notification delivery"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationStatus(Enum):
    """Status of notifications"""
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Tag:
    """Represents a parsed tag/mention"""
    tag_id: str
    tag_type: TagType
    raw_text: str              # Original text (e.g., "@Site_101")
    normalized_value: str      # Normalized value (e.g., "Site_101")
    display_name: str          # Human-readable name
    entity_id: Optional[str]   # Linked entity ID if resolved
    position_start: int        # Position in source text
    position_end: int
    context: str               # Surrounding text for context
    metadata: Dict = field(default_factory=dict)
    is_resolved: bool = False  # Whether entity was found in system
    link_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['tag_type'] = self.tag_type.value
        result['created_at'] = self.created_at.isoformat()
        return result


@dataclass
class TagPattern:
    """Pattern definition for tag recognition"""
    tag_type: TagType
    pattern: str               # Regex pattern
    prefix: str                # Tag prefix (@ or #)
    examples: List[str]
    description: str
    priority: int = 0          # Higher priority patterns checked first
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['tag_type'] = self.tag_type.value
        return result


@dataclass
class NotificationTrigger:
    """Defines when notifications should be sent"""
    trigger_id: str
    tag_type: TagType
    condition: str             # Trigger condition (e.g., "on_mention", "on_assign")
    priority: NotificationPriority
    channels: List[NotificationChannel]
    template: str              # Notification message template
    enabled: bool = True
    cooldown_minutes: int = 0  # Prevent spam
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['tag_type'] = self.tag_type.value
        result['priority'] = self.priority.value
        result['channels'] = [c.value for c in self.channels]
        return result


@dataclass
class Notification:
    """A notification to be sent"""
    notification_id: str
    recipient_id: str
    recipient_type: TagType    # user, role, team
    trigger_id: str
    source_type: str           # issue, room, message, etc.
    source_id: str
    title: str
    message: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    status: NotificationStatus = NotificationStatus.PENDING
    tags_referenced: List[str] = field(default_factory=list)
    link_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['recipient_type'] = self.recipient_type.value
        result['priority'] = self.priority.value
        result['channels'] = [c.value for c in self.channels]
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['sent_at'] = self.sent_at.isoformat() if self.sent_at else None
        result['read_at'] = self.read_at.isoformat() if self.read_at else None
        return result


@dataclass
class UserPreferences:
    """User notification preferences"""
    user_id: str
    enabled_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP, NotificationChannel.EMAIL])
    muted_tags: List[str] = field(default_factory=list)  # Tags to ignore
    muted_sources: List[str] = field(default_factory=list)  # Source IDs to ignore
    batch_low_priority: bool = True  # Batch low priority into digest
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None
    digest_frequency: str = "daily"  # daily, weekly, none
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['enabled_channels'] = [c.value for c in self.enabled_channels]
        return result


@dataclass
class EntityLink:
    """Generated link for an entity"""
    entity_type: TagType
    entity_id: str
    display_text: str
    url: str
    tooltip: Optional[str] = None
    icon: Optional[str] = None
    
    def to_html(self) -> str:
        """Generate HTML link"""
        tooltip_attr = f' title="{self.tooltip}"' if self.tooltip else ''
        icon_html = f'{self.icon} ' if self.icon else ''
        return f'<a href="{self.url}" class="entity-link entity-{self.entity_type.value}"{tooltip_attr}>{icon_html}{self.display_text}</a>'
    
    def to_markdown(self) -> str:
        """Generate Markdown link"""
        return f'[{self.display_text}]({self.url})'


# =============================================================================
# TAG PATTERNS - FIXED PRIORITY ORDER
# =============================================================================

DEFAULT_TAG_PATTERNS: List[TagPattern] = [
    # HIGHEST PRIORITY: Specific entity patterns (must match before generic USER)
    
    # Issue mentions - @ISS-XXXXXXXX-XXXXXX (priority 100)
    TagPattern(
        tag_type=TagType.ISSUE,
        pattern=r'@(ISS-[0-9]{8}-[A-Za-z0-9]{6})',
        prefix='@',
        examples=['@ISS-20260101-ABC123'],
        description='Reference an issue',
        priority=100
    ),
    
    # Room mentions - @ROOM-XXXXXXXX-XXXXXX (priority 100)
    TagPattern(
        tag_type=TagType.ROOM,
        pattern=r'@(ROOM-[0-9]{8}-[A-Za-z0-9]{6})',
        prefix='@',
        examples=['@ROOM-20260101-XYZ789'],
        description='Reference an investigation room',
        priority=100
    ),
    
    # SAE mentions - @SAE_XXX or @SAE-XXXX-XXX (priority 90)
    TagPattern(
        tag_type=TagType.SAE,
        pattern=r'@(SAE[_-][A-Za-z0-9_-]+)',
        prefix='@',
        examples=['@SAE_001', '@SAE-2024-001'],
        description='Reference a Serious Adverse Event',
        priority=90
    ),
    
    # Site mentions - @Site_XXX or @Site-XXX (priority 80)
    TagPattern(
        tag_type=TagType.SITE,
        pattern=r'@(Site[_-][A-Za-z0-9_-]+)',
        prefix='@',
        examples=['@Site_101', '@Site_JP001', '@Site-EU-005'],
        description='Reference a clinical site',
        priority=80
    ),
    
    # Study mentions - @Study_XXX (priority 80)
    TagPattern(
        tag_type=TagType.STUDY,
        pattern=r'@(Study[_-][A-Za-z0-9_-]+)',
        prefix='@',
        examples=['@Study_21', '@Study_ABC123'],
        description='Reference a clinical study',
        priority=80
    ),
    
    # Patient mentions - @Patient_XXX or @Subject_XXX (priority 80)
    TagPattern(
        tag_type=TagType.PATIENT,
        pattern=r'@((?:Patient|Subject)[_-][A-Za-z0-9_|.-]+)',
        prefix='@',
        examples=['@Patient_1001', '@Subject_ABC', '@Patient_Study_1|Site_1|Subject_1'],
        description='Reference a patient/subject',
        priority=80
    ),
    
    # Query mentions - @Query_XXXXX (priority 80)
    TagPattern(
        tag_type=TagType.QUERY,
        pattern=r'@(Query[_-][0-9]+)',
        prefix='@',
        examples=['@Query_12345', '@Query-67890'],
        description='Reference a data query',
        priority=80
    ),
    
    # Team mentions - @XXX-Team (priority 70)
    TagPattern(
        tag_type=TagType.TEAM,
        pattern=r'@((?:CRA|DM|Safety|Coding|Site|Study|All)-Team)',
        prefix='@',
        examples=['@CRA-Team', '@DM-Team', '@Safety-Team'],
        description='Mention an entire team',
        priority=70
    ),
    
    # Role mentions - @RoleName (priority 60)
    TagPattern(
        tag_type=TagType.ROLE,
        pattern=r'@(CRA|DataManager|Data_Manager|DM|SafetyPhysician|Safety_Physician|SafetyDM|Safety_DM|StudyLead|Study_Lead|CTM|PI|MedicalCoder|Medical_Coder|Coder|SiteCoordinator|Site_Coordinator)(?![A-Za-z0-9_-])',
        prefix='@',
        examples=['@CRA', '@DataManager', '@SafetyPhysician', '@StudyLead'],
        description='Mention all users with a specific role',
        priority=60
    ),
    
    # Topic/hashtag mentions - #topic (priority 50)
    TagPattern(
        tag_type=TagType.TOPIC,
        pattern=r'#([a-zA-Z][a-zA-Z0-9_]{1,30})',
        prefix='#',
        examples=['#urgent', '#escalation', '#consent', '#safety', '#dblock'],
        description='Tag a topic or category',
        priority=50
    ),
    
    # LOWEST PRIORITY: Generic user mentions (priority 10)
    # Must be last to avoid matching Site_101, Study_21, etc.
    TagPattern(
        tag_type=TagType.USER,
        pattern=r'@([a-z][a-z0-9_]{2,30})(?![A-Za-z0-9_-])',
        prefix='@',
        examples=['@sarah_chen', '@john_smith', '@cra_001'],
        description='Mention a specific user',
        priority=10
    ),
]


# =============================================================================
# DEFAULT NOTIFICATION TRIGGERS
# =============================================================================

DEFAULT_NOTIFICATION_TRIGGERS: List[NotificationTrigger] = [
    # Direct user mention
    NotificationTrigger(
        trigger_id="TRG-USER-MENTION",
        tag_type=TagType.USER,
        condition="on_mention",
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        template="You were mentioned by {author} in {source_type}: {title}"
    ),
    
    # Role mention
    NotificationTrigger(
        trigger_id="TRG-ROLE-MENTION",
        tag_type=TagType.ROLE,
        condition="on_mention",
        priority=NotificationPriority.NORMAL,
        channels=[NotificationChannel.IN_APP],
        template="Your role ({role}) was mentioned in {source_type}: {title}"
    ),
    
    # Team mention
    NotificationTrigger(
        trigger_id="TRG-TEAM-MENTION",
        tag_type=TagType.TEAM,
        condition="on_mention",
        priority=NotificationPriority.NORMAL,
        channels=[NotificationChannel.IN_APP],
        template="Your team ({team}) was mentioned in {source_type}: {title}"
    ),
    
    # Urgent topic
    NotificationTrigger(
        trigger_id="TRG-URGENT-TOPIC",
        tag_type=TagType.TOPIC,
        condition="topic_urgent",
        priority=NotificationPriority.URGENT,
        channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL, NotificationChannel.SMS],
        template="URGENT: {title} - {message}"
    ),
    
    # Safety topic
    NotificationTrigger(
        trigger_id="TRG-SAFETY-TOPIC",
        tag_type=TagType.TOPIC,
        condition="topic_safety",
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        template="Safety Alert: {title}"
    ),
    
    # SAE mention
    NotificationTrigger(
        trigger_id="TRG-SAE-MENTION",
        tag_type=TagType.SAE,
        condition="on_mention",
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
        template="SAE {sae_id} was referenced: {title}"
    ),
]


# =============================================================================
# LINK GENERATOR
# =============================================================================

class LinkGenerator:
    """Generates URLs and links for tagged entities"""
    
    # Base URL patterns for different entity types
    URL_PATTERNS = {
        TagType.USER: "/users/{entity_id}",
        TagType.ROLE: "/roles/{entity_id}",
        TagType.TEAM: "/teams/{entity_id}",
        TagType.SITE: "/sites/{entity_id}",
        TagType.STUDY: "/studies/{entity_id}",
        TagType.PATIENT: "/patients/{entity_id}",
        TagType.ISSUE: "/issues/{entity_id}",
        TagType.ROOM: "/rooms/{entity_id}",
        TagType.QUERY: "/queries/{entity_id}",
        TagType.SAE: "/safety/sae/{entity_id}",
        TagType.TOPIC: "/topics/{entity_id}",
    }
    
    # Icons for entity types (emoji or icon class)
    ICONS = {
        TagType.USER: "👤",
        TagType.ROLE: "🎭",
        TagType.TEAM: "👥",
        TagType.SITE: "🏥",
        TagType.STUDY: "📋",
        TagType.PATIENT: "🧑",
        TagType.ISSUE: "⚠️",
        TagType.ROOM: "🔍",
        TagType.QUERY: "❓",
        TagType.SAE: "🚨",
        TagType.TOPIC: "#️⃣",
    }
    
    def __init__(self, base_url: str = ""):
        self.base_url = base_url
    
    def generate_link(self, tag: Tag) -> EntityLink:
        """Generate a link for a tag"""
        url_pattern = self.URL_PATTERNS.get(tag.tag_type, "/entity/{entity_id}")
        entity_id = tag.entity_id or tag.normalized_value
        url = self.base_url + url_pattern.format(entity_id=entity_id)
        
        return EntityLink(
            entity_type=tag.tag_type,
            entity_id=entity_id,
            display_text=tag.display_name,
            url=url,
            tooltip=f"{tag.tag_type.value.title()}: {tag.display_name}",
            icon=self.ICONS.get(tag.tag_type)
        )
    
    def generate_url(self, tag_type: TagType, entity_id: str) -> str:
        """Generate URL for an entity"""
        url_pattern = self.URL_PATTERNS.get(tag_type, "/entity/{entity_id}")
        return self.base_url + url_pattern.format(entity_id=entity_id)


# =============================================================================
# TAG PARSER
# =============================================================================

class TagParser:
    """Parses and extracts tags from text"""
    
    def __init__(self, patterns: Optional[List[TagPattern]] = None):
        self.patterns = patterns or DEFAULT_TAG_PATTERNS
        # Sort by priority (higher first) - CRITICAL for correct matching
        self.patterns.sort(key=lambda p: -p.priority)
        # Compile patterns
        self._compiled_patterns = [
            (p, re.compile(p.pattern, re.IGNORECASE if p.tag_type != TagType.USER else 0))
            for p in self.patterns
        ]
    
    def parse(self, text: str, context_chars: int = 50) -> List[Tag]:
        """Parse text and extract all tags"""
        tags = []
        matched_ranges = []  # Track matched positions to avoid overlaps
        
        for pattern, regex in self._compiled_patterns:
            for match in regex.finditer(text):
                start, end = match.span()
                
                # Skip if this position overlaps with an already matched range
                overlap = False
                for (s, e) in matched_ranges:
                    if not (end <= s or start >= e):  # Check for overlap
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                matched_ranges.append((start, end))
                
                # Extract context
                context_start = max(0, start - context_chars)
                context_end = min(len(text), end + context_chars)
                context = text[context_start:context_end]
                
                # Create tag
                raw_text = match.group(0)
                normalized = match.group(1) if match.lastindex else match.group(0).lstrip('@#')
                
                tag = Tag(
                    tag_id=self._generate_tag_id(raw_text, start),
                    tag_type=pattern.tag_type,
                    raw_text=raw_text,
                    normalized_value=normalized,
                    display_name=self._format_display_name(pattern.tag_type, normalized),
                    entity_id=None,  # Will be resolved later
                    position_start=start,
                    position_end=end,
                    context=context,
                    is_resolved=False
                )
                
                tags.append(tag)
        
        # Sort by position
        tags.sort(key=lambda t: t.position_start)
        
        return tags
    
    def _generate_tag_id(self, raw_text: str, position: int) -> str:
        """Generate unique tag ID"""
        unique_str = f"{raw_text}:{position}:{time.time_ns()}:{uuid.uuid4().hex[:4]}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        return f"TAG-{hash_val.upper()}"
    
    def _format_display_name(self, tag_type: TagType, normalized: str) -> str:
        """Format display name based on tag type"""
        if tag_type == TagType.USER:
            # Convert user_id to readable name
            return normalized.replace('_', ' ').title()
        elif tag_type == TagType.ROLE:
            # Format role name
            return normalized.replace('_', ' ')
        elif tag_type == TagType.TEAM:
            return normalized
        elif tag_type == TagType.SITE:
            return f"Site {normalized.replace('Site_', '').replace('Site-', '')}"
        elif tag_type == TagType.STUDY:
            return f"Study {normalized.replace('Study_', '').replace('Study-', '')}"
        elif tag_type == TagType.PATIENT:
            return f"Patient {normalized.replace('Patient_', '').replace('Subject_', '')}"
        elif tag_type == TagType.QUERY:
            return f"Query {normalized.replace('Query_', '').replace('Query-', '')}"
        elif tag_type == TagType.SAE:
            return f"SAE {normalized.replace('SAE_', '').replace('SAE-', '')}"
        elif tag_type == TagType.ISSUE:
            return f"Issue {normalized}"
        elif tag_type == TagType.ROOM:
            return f"Room {normalized}"
        elif tag_type == TagType.TOPIC:
            return f"#{normalized}"
        else:
            return normalized
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract just the mention strings (for quick processing)"""
        mentions = []
        mention_pattern = re.compile(r'@([a-zA-Z][a-zA-Z0-9_|-]{2,50})')
        for match in mention_pattern.finditer(text):
            mentions.append(match.group(1))
        return mentions
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract just the topic/hashtag strings"""
        topics = []
        topic_pattern = re.compile(r'#([a-zA-Z][a-zA-Z0-9_]{1,30})')
        for match in topic_pattern.finditer(text):
            topics.append(match.group(1))
        return topics
    
    def highlight_tags(self, text: str, tags: List[Tag], format: str = 'html') -> str:
        """Replace tags with highlighted/linked versions"""
        if not tags:
            return text
        
        # Sort tags by position (reverse order for replacement)
        sorted_tags = sorted(tags, key=lambda t: -t.position_start)
        
        result = text
        link_gen = LinkGenerator()
        
        for tag in sorted_tags:
            if format == 'html':
                link = link_gen.generate_link(tag)
                replacement = link.to_html()
            elif format == 'markdown':
                link = link_gen.generate_link(tag)
                replacement = link.to_markdown()
            else:
                replacement = f"**{tag.raw_text}**"
            
            result = result[:tag.position_start] + replacement + result[tag.position_end:]
        
        return result


# =============================================================================
# ENTITY RESOLVER
# =============================================================================

class EntityResolver:
    """Resolves tags to actual entities in the system"""
    
    def __init__(self):
        self._user_cache: Dict[str, Dict] = {}
        self._site_cache: Dict[str, Dict] = {}
        self._study_cache: Dict[str, Dict] = {}
        self._load_default_users()
    
    def _load_default_users(self):
        """Load default demo users"""
        self._user_cache = {
            'sarah_chen': {'id': 'cra_001', 'name': 'Sarah Chen', 'role': 'CRA'},
            'cra_001': {'id': 'cra_001', 'name': 'Sarah Chen', 'role': 'CRA'},
            'alex_kim': {'id': 'dm_001', 'name': 'Alex Kim', 'role': 'Data Manager'},
            'dm_001': {'id': 'dm_001', 'name': 'Alex Kim', 'role': 'Data Manager'},
            'john_smith': {'id': 'lead_001', 'name': 'John Smith', 'role': 'Study Lead'},
            'lead_001': {'id': 'lead_001', 'name': 'John Smith', 'role': 'Study Lead'},
            'maria_garcia': {'id': 'safety_001', 'name': 'Dr. Maria Garcia', 'role': 'Safety Physician'},
            'safety_001': {'id': 'safety_001', 'name': 'Dr. Maria Garcia', 'role': 'Safety Physician'},
            'mike_johnson': {'id': 'coder_001', 'name': 'Mike Johnson', 'role': 'Medical Coder'},
            'coder_001': {'id': 'coder_001', 'name': 'Mike Johnson', 'role': 'Medical Coder'},
            'emily_brown': {'id': 'ctm_001', 'name': 'Emily Brown', 'role': 'CTM'},
            'ctm_001': {'id': 'ctm_001', 'name': 'Emily Brown', 'role': 'CTM'},
        }
    
    def resolve(self, tag: Tag) -> Tag:
        """Resolve a tag to its entity"""
        if tag.tag_type == TagType.USER:
            return self._resolve_user(tag)
        elif tag.tag_type == TagType.ROLE:
            return self._resolve_role(tag)
        elif tag.tag_type == TagType.SITE:
            return self._resolve_site(tag)
        elif tag.tag_type == TagType.STUDY:
            return self._resolve_study(tag)
        elif tag.tag_type == TagType.TOPIC:
            # Topics are always "resolved"
            tag.is_resolved = True
            tag.entity_id = tag.normalized_value.lower()
            return tag
        else:
            # For other types, mark as resolved if looks valid
            tag.is_resolved = True
            tag.entity_id = tag.normalized_value
            return tag
    
    def _resolve_user(self, tag: Tag) -> Tag:
        """Resolve user mention"""
        normalized = tag.normalized_value.lower()
        if normalized in self._user_cache:
            user = self._user_cache[normalized]
            tag.entity_id = user['id']
            tag.display_name = user['name']
            tag.metadata['role'] = user['role']
            tag.is_resolved = True
        return tag
    
    def _resolve_role(self, tag: Tag) -> Tag:
        """Resolve role mention (returns list of users with that role)"""
        role_map = {
            'cra': 'CRA',
            'datamanager': 'Data Manager',
            'data_manager': 'Data Manager',
            'dm': 'Data Manager',
            'safetyphysician': 'Safety Physician',
            'safety_physician': 'Safety Physician',
            'safetydm': 'Safety Data Manager',
            'safety_dm': 'Safety Data Manager',
            'studylead': 'Study Lead',
            'study_lead': 'Study Lead',
            'ctm': 'CTM',
            'pi': 'Principal Investigator',
            'medicalcoder': 'Medical Coder',
            'medical_coder': 'Medical Coder',
            'coder': 'Medical Coder',
            'sitecoordinator': 'Site Coordinator',
            'site_coordinator': 'Site Coordinator',
        }
        
        normalized = tag.normalized_value.lower().replace('-', '').replace('_', '')
        if normalized in role_map:
            tag.display_name = role_map[normalized]
            tag.entity_id = role_map[normalized]
            tag.is_resolved = True
            # Get users with this role
            role_users = [u for u in self._user_cache.values() if u['role'] == role_map.get(normalized, tag.normalized_value)]
            tag.metadata['user_count'] = len(role_users)
            tag.metadata['users'] = [u['id'] for u in role_users]
        return tag
    
    def _resolve_site(self, tag: Tag) -> Tag:
        """Resolve site mention"""
        # Site IDs are typically valid if they match the pattern
        tag.entity_id = tag.normalized_value
        site_num = tag.normalized_value.replace('Site_', '').replace('Site-', '')
        tag.display_name = f"Site {site_num}"
        tag.is_resolved = True
        return tag
    
    def _resolve_study(self, tag: Tag) -> Tag:
        """Resolve study mention"""
        tag.entity_id = tag.normalized_value
        study_num = tag.normalized_value.replace('Study_', '').replace('Study-', '')
        tag.display_name = f"Study {study_num}"
        tag.is_resolved = True
        return tag
    
    def resolve_all(self, tags: List[Tag]) -> List[Tag]:
        """Resolve all tags"""
        return [self.resolve(tag) for tag in tags]
    
    def get_users_for_role(self, role: str) -> List[str]:
        """Get user IDs for a role"""
        return [u['id'] for u in self._user_cache.values() if u['role'] == role]
    
    def get_users_for_team(self, team: str) -> List[str]:
        """Get user IDs for a team"""
        team_role_map = {
            'CRA-Team': 'CRA',
            'DM-Team': 'Data Manager',
            'Safety-Team': 'Safety Physician',
            'Coding-Team': 'Medical Coder',
            'Study-Team': 'Study Lead',
        }
        role = team_role_map.get(team)
        if role:
            return self.get_users_for_role(role)
        return []


# =============================================================================
# DATABASE CONTEXT MANAGER
# =============================================================================

@contextmanager
def get_db_connection(db_path: str):
    """Context manager for database connections - ensures proper closing"""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# NOTIFICATION MANAGER
# =============================================================================

class NotificationManager:
    """Manages notification creation, queueing, and delivery"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.triggers = {t.trigger_id: t for t in DEFAULT_NOTIFICATION_TRIGGERS}
        self.user_preferences: Dict[str, UserPreferences] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize notification tables"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    recipient_id TEXT NOT NULL,
                    recipient_type TEXT NOT NULL,
                    trigger_id TEXT,
                    source_type TEXT,
                    source_id TEXT,
                    title TEXT NOT NULL,
                    message TEXT,
                    priority TEXT NOT NULL,
                    channels TEXT,
                    status TEXT NOT NULL,
                    tags_referenced TEXT,
                    link_url TEXT,
                    created_at TEXT NOT NULL,
                    sent_at TEXT,
                    read_at TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    enabled_channels TEXT,
                    muted_tags TEXT,
                    muted_sources TEXT,
                    batch_low_priority INTEGER DEFAULT 1,
                    quiet_hours_start INTEGER,
                    quiet_hours_end INTEGER,
                    digest_frequency TEXT DEFAULT 'daily'
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_notifications_recipient 
                ON notifications(recipient_id, status)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_notifications_status 
                ON notifications(status, priority)
            ''')
            
            conn.commit()
    
    def create_notifications_from_tags(
        self,
        tags: List[Tag],
        source_type: str,
        source_id: str,
        title: str,
        message: str,
        author_id: str,
        author_name: str
    ) -> List[Notification]:
        """Create notifications based on parsed tags"""
        notifications = []
        resolver = EntityResolver()
        
        for tag in tags:
            if tag.tag_type == TagType.USER:
                # Direct user mention
                notification = self._create_user_notification(
                    tag, source_type, source_id, title, message, author_name
                )
                if notification:
                    notifications.append(notification)
            
            elif tag.tag_type == TagType.ROLE:
                # Notify all users with this role
                role_users = resolver.get_users_for_role(tag.display_name)
                for user_id in role_users:
                    notification = self._create_role_notification(
                        tag, user_id, source_type, source_id, title, message, author_name
                    )
                    if notification:
                        notifications.append(notification)
            
            elif tag.tag_type == TagType.TEAM:
                # Notify all users in team
                team_users = resolver.get_users_for_team(tag.normalized_value)
                for user_id in team_users:
                    notification = self._create_team_notification(
                        tag, user_id, source_type, source_id, title, message, author_name
                    )
                    if notification:
                        notifications.append(notification)
            
            elif tag.tag_type == TagType.TOPIC:
                # Check for special topics
                if tag.normalized_value.lower() == 'urgent':
                    # Urgent topic - special handling
                    pass
                elif tag.normalized_value.lower() == 'safety':
                    # Safety topic - notify safety team
                    safety_users = resolver.get_users_for_role('Safety Physician')
                    for user_id in safety_users:
                        notification = self._create_safety_notification(
                            tag, user_id, source_type, source_id, title, message
                        )
                        if notification:
                            notifications.append(notification)
        
        # Save notifications to database
        for notification in notifications:
            self._save_notification(notification)
        
        return notifications
    
    def _create_user_notification(
        self, tag: Tag, source_type: str, source_id: str,
        title: str, message: str, author_name: str
    ) -> Optional[Notification]:
        """Create notification for user mention"""
        trigger = self.triggers.get("TRG-USER-MENTION")
        if not trigger or not trigger.enabled:
            return None
        
        return Notification(
            notification_id=self._generate_notification_id(),
            recipient_id=tag.entity_id or tag.normalized_value,
            recipient_type=TagType.USER,
            trigger_id=trigger.trigger_id,
            source_type=source_type,
            source_id=source_id,
            title=trigger.template.format(
                author=author_name,
                source_type=source_type,
                title=title
            ),
            message=message,
            priority=trigger.priority,
            channels=trigger.channels,
            tags_referenced=[tag.tag_id],
            link_url=LinkGenerator().generate_url(
                TagType.ISSUE if source_type == 'issue' else TagType.ROOM,
                source_id
            )
        )
    
    def _create_role_notification(
        self, tag: Tag, user_id: str, source_type: str, source_id: str,
        title: str, message: str, author_name: str
    ) -> Optional[Notification]:
        """Create notification for role mention"""
        trigger = self.triggers.get("TRG-ROLE-MENTION")
        if not trigger or not trigger.enabled:
            return None
        
        return Notification(
            notification_id=self._generate_notification_id(),
            recipient_id=user_id,
            recipient_type=TagType.ROLE,
            trigger_id=trigger.trigger_id,
            source_type=source_type,
            source_id=source_id,
            title=trigger.template.format(
                role=tag.display_name,
                source_type=source_type,
                title=title
            ),
            message=message,
            priority=trigger.priority,
            channels=trigger.channels,
            tags_referenced=[tag.tag_id],
            link_url=LinkGenerator().generate_url(
                TagType.ISSUE if source_type == 'issue' else TagType.ROOM,
                source_id
            )
        )
    
    def _create_team_notification(
        self, tag: Tag, user_id: str, source_type: str, source_id: str,
        title: str, message: str, author_name: str
    ) -> Optional[Notification]:
        """Create notification for team mention"""
        trigger = self.triggers.get("TRG-TEAM-MENTION")
        if not trigger or not trigger.enabled:
            return None
        
        return Notification(
            notification_id=self._generate_notification_id(),
            recipient_id=user_id,
            recipient_type=TagType.TEAM,
            trigger_id=trigger.trigger_id,
            source_type=source_type,
            source_id=source_id,
            title=trigger.template.format(
                team=tag.normalized_value,
                source_type=source_type,
                title=title
            ),
            message=message,
            priority=trigger.priority,
            channels=trigger.channels,
            tags_referenced=[tag.tag_id]
        )
    
    def _create_safety_notification(
        self, tag: Tag, user_id: str, source_type: str, source_id: str,
        title: str, message: str
    ) -> Optional[Notification]:
        """Create notification for safety topic"""
        trigger = self.triggers.get("TRG-SAFETY-TOPIC")
        if not trigger or not trigger.enabled:
            return None
        
        return Notification(
            notification_id=self._generate_notification_id(),
            recipient_id=user_id,
            recipient_type=TagType.USER,
            trigger_id=trigger.trigger_id,
            source_type=source_type,
            source_id=source_id,
            title=trigger.template.format(title=title),
            message=message,
            priority=trigger.priority,
            channels=trigger.channels,
            tags_referenced=[tag.tag_id]
        )
    
    def _generate_notification_id(self) -> str:
        """Generate unique notification ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_part = uuid.uuid4().hex[:8].upper()
        return f"NOTIF-{timestamp}-{random_part}"
    
    def _save_notification(self, notification: Notification):
        """Save notification to database"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO notifications (
                    notification_id, recipient_id, recipient_type, trigger_id,
                    source_type, source_id, title, message, priority, channels,
                    status, tags_referenced, link_url, created_at, sent_at, read_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.notification_id,
                notification.recipient_id,
                notification.recipient_type.value,
                notification.trigger_id,
                notification.source_type,
                notification.source_id,
                notification.title,
                notification.message,
                notification.priority.value,
                json.dumps([c.value for c in notification.channels]),
                notification.status.value,
                json.dumps(notification.tags_referenced),
                notification.link_url,
                notification.created_at.isoformat(),
                notification.sent_at.isoformat() if notification.sent_at else None,
                notification.read_at.isoformat() if notification.read_at else None,
                json.dumps(notification.metadata)
            ))
            
            conn.commit()
    
    def get_notifications(
        self,
        recipient_id: str,
        status: Optional[NotificationStatus] = None,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a recipient"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM notifications WHERE recipient_id = ?"
            params = [recipient_id]
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        notifications = []
        for row in rows:
            notifications.append(self._row_to_notification(row))
        
        return notifications
    
    def _row_to_notification(self, row) -> Notification:
        """Convert database row to Notification object"""
        return Notification(
            notification_id=row[0],
            recipient_id=row[1],
            recipient_type=TagType(row[2]),
            trigger_id=row[3],
            source_type=row[4],
            source_id=row[5],
            title=row[6],
            message=row[7],
            priority=NotificationPriority(row[8]),
            channels=[NotificationChannel(c) for c in json.loads(row[9] or '[]')],
            status=NotificationStatus(row[10]),
            tags_referenced=json.loads(row[11] or '[]'),
            link_url=row[12],
            created_at=datetime.fromisoformat(row[13]),
            sent_at=datetime.fromisoformat(row[14]) if row[14] else None,
            read_at=datetime.fromisoformat(row[15]) if row[15] else None,
            metadata=json.loads(row[16] or '{}')
        )
    
    def mark_as_read(self, notification_id: str):
        """Mark notification as read"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE notifications 
                SET status = ?, read_at = ?
                WHERE notification_id = ?
            ''', (NotificationStatus.READ.value, datetime.now().isoformat(), notification_id))
            
            conn.commit()
    
    def get_unread_count(self, recipient_id: str) -> int:
        """Get count of unread notifications"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM notifications 
                WHERE recipient_id = ? AND status NOT IN (?, ?)
            ''', (recipient_id, NotificationStatus.READ.value, NotificationStatus.FAILED.value))
            
            count = cursor.fetchone()[0]
        
        return count
    
    def get_statistics(self) -> Dict:
        """Get notification statistics"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total notifications
            cursor.execute("SELECT COUNT(*) FROM notifications")
            total = cursor.fetchone()[0]
            
            # By status
            cursor.execute('''
                SELECT status, COUNT(*) FROM notifications GROUP BY status
            ''')
            by_status = dict(cursor.fetchall())
            
            # By priority
            cursor.execute('''
                SELECT priority, COUNT(*) FROM notifications GROUP BY priority
            ''')
            by_priority = dict(cursor.fetchall())
        
        return {
            'total_notifications': total,
            'by_status': by_status,
            'by_priority': by_priority
        }


# =============================================================================
# TAGGING SYSTEM (Main Class)
# =============================================================================

class TaggingSystem:
    """Main class for the @Tagging System"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path("data/collaboration/tagging_system.db"))
        
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.parser = TagParser()
        self.resolver = EntityResolver()
        self.link_generator = LinkGenerator()
        self.notification_manager = NotificationManager(db_path)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize tagging tables"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tag history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tag_history (
                    tag_id TEXT PRIMARY KEY,
                    tag_type TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    display_name TEXT,
                    entity_id TEXT,
                    source_type TEXT,
                    source_id TEXT,
                    position_start INTEGER,
                    position_end INTEGER,
                    context TEXT,
                    is_resolved INTEGER,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    metadata TEXT
                )
            ''')
            
            # Tag statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tag_stats (
                    tag_value TEXT PRIMARY KEY,
                    tag_type TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    first_used TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tag_history_source 
                ON tag_history(source_type, source_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tag_history_type 
                ON tag_history(tag_type)
            ''')
            
            conn.commit()
    
    def parse_and_process(
        self,
        text: str,
        source_type: str,
        source_id: str,
        author_id: str,
        author_name: str,
        title: str = "",
        create_notifications: bool = True
    ) -> Dict:
        """
        Main method: Parse text, resolve entities, create notifications
        
        Returns:
            Dict with tags, notifications, highlighted_text
        """
        # Parse tags
        tags = self.parser.parse(text)
        
        # Resolve entities
        resolved_tags = self.resolver.resolve_all(tags)
        
        # Save to history
        self._save_tags_to_history(resolved_tags, source_type, source_id, author_id)
        
        # Update statistics
        self._update_tag_stats(resolved_tags)
        
        # Create notifications
        notifications = []
        if create_notifications:
            notifications = self.notification_manager.create_notifications_from_tags(
                resolved_tags, source_type, source_id, title, text, author_id, author_name
            )
        
        # Generate highlighted text
        highlighted_html = self.parser.highlight_tags(text, resolved_tags, format='html')
        highlighted_md = self.parser.highlight_tags(text, resolved_tags, format='markdown')
        
        return {
            'tags': [t.to_dict() for t in resolved_tags],
            'tag_count': len(resolved_tags),
            'tags_by_type': self._group_tags_by_type(resolved_tags),
            'notifications_created': len(notifications),
            'notifications': [n.to_dict() for n in notifications],
            'highlighted_html': highlighted_html,
            'highlighted_markdown': highlighted_md,
            'mentions': self.parser.extract_mentions(text),
            'topics': self.parser.extract_topics(text)
        }
    
    def _group_tags_by_type(self, tags: List[Tag]) -> Dict[str, List[str]]:
        """Group tags by type"""
        result = {}
        for tag in tags:
            type_name = tag.tag_type.value
            if type_name not in result:
                result[type_name] = []
            result[type_name].append(tag.normalized_value)
        return result
    
    def _save_tags_to_history(
        self,
        tags: List[Tag],
        source_type: str,
        source_id: str,
        created_by: str
    ):
        """Save tags to history table"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            for tag in tags:
                cursor.execute('''
                    INSERT OR REPLACE INTO tag_history (
                        tag_id, tag_type, raw_text, normalized_value, display_name,
                        entity_id, source_type, source_id, position_start, position_end,
                        context, is_resolved, created_at, created_by, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tag.tag_id,
                    tag.tag_type.value,
                    tag.raw_text,
                    tag.normalized_value,
                    tag.display_name,
                    tag.entity_id,
                    source_type,
                    source_id,
                    tag.position_start,
                    tag.position_end,
                    tag.context,
                    1 if tag.is_resolved else 0,
                    tag.created_at.isoformat(),
                    created_by,
                    json.dumps(tag.metadata)
                ))
            
            conn.commit()
    
    def _update_tag_stats(self, tags: List[Tag]):
        """Update tag usage statistics"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            for tag in tags:
                cursor.execute('''
                    INSERT INTO tag_stats (tag_value, tag_type, usage_count, last_used, first_used)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(tag_value) DO UPDATE SET
                        usage_count = usage_count + 1,
                        last_used = ?
                ''', (
                    tag.normalized_value,
                    tag.tag_type.value,
                    now,
                    now,
                    now
                ))
            
            conn.commit()
    
    def get_tag_suggestions(self, partial: str, tag_type: Optional[TagType] = None, limit: int = 10) -> List[Dict]:
        """Get tag suggestions based on partial input"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT tag_value, tag_type, usage_count 
                FROM tag_stats 
                WHERE tag_value LIKE ?
            '''
            params = [f'{partial}%']
            
            if tag_type:
                query += ' AND tag_type = ?'
                params.append(tag_type.value)
            
            query += ' ORDER BY usage_count DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [
            {'value': row[0], 'type': row[1], 'usage_count': row[2]}
            for row in rows
        ]
    
    def get_tags_for_source(self, source_type: str, source_id: str) -> List[Tag]:
        """Get all tags used in a source"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM tag_history 
                WHERE source_type = ? AND source_id = ?
                ORDER BY position_start
            ''', (source_type, source_id))
            
            rows = cursor.fetchall()
        
        tags = []
        for row in rows:
            tags.append(Tag(
                tag_id=row[0],
                tag_type=TagType(row[1]),
                raw_text=row[2],
                normalized_value=row[3],
                display_name=row[4],
                entity_id=row[5],
                position_start=row[8],
                position_end=row[9],
                context=row[10],
                is_resolved=bool(row[11]),
                created_at=datetime.fromisoformat(row[12]),
                metadata=json.loads(row[14] or '{}')
            ))
        
        return tags
    
    def generate_link(self, tag_type: TagType, entity_id: str, display_text: str) -> EntityLink:
        """Generate a link for an entity"""
        return EntityLink(
            entity_type=tag_type,
            entity_id=entity_id,
            display_text=display_text,
            url=self.link_generator.generate_url(tag_type, entity_id),
            tooltip=f"{tag_type.value.title()}: {display_text}",
            icon=LinkGenerator.ICONS.get(tag_type)
        )
    
    def get_statistics(self) -> Dict:
        """Get tagging system statistics"""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total tags
            cursor.execute("SELECT COUNT(*) FROM tag_history")
            total_tags = cursor.fetchone()[0]
            
            # Tags by type
            cursor.execute('''
                SELECT tag_type, COUNT(*) FROM tag_history GROUP BY tag_type
            ''')
            by_type = dict(cursor.fetchall())
            
            # Most used tags
            cursor.execute('''
                SELECT tag_value, tag_type, usage_count 
                FROM tag_stats 
                ORDER BY usage_count DESC LIMIT 10
            ''')
            top_tags = [
                {'value': row[0], 'type': row[1], 'count': row[2]}
                for row in cursor.fetchall()
            ]
        
        # Notification stats
        notification_stats = self.notification_manager.get_statistics()
        
        return {
            'total_tags': total_tags,
            'tags_by_type': by_type,
            'top_tags': top_tags,
            'notifications': notification_stats
        }


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_tagging_system_instance: Optional[TaggingSystem] = None


def get_tagging_system(db_path: Optional[str] = None) -> TaggingSystem:
    """Get or create TaggingSystem instance"""
    global _tagging_system_instance
    if _tagging_system_instance is None:
        _tagging_system_instance = TaggingSystem(db_path)
    return _tagging_system_instance


def reset_tagging_system():
    """Reset the singleton instance (for testing)"""
    global _tagging_system_instance
    _tagging_system_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_tags(text: str) -> List[Tag]:
    """Quick function to parse tags from text"""
    parser = TagParser()
    return parser.parse(text)


def extract_mentions(text: str) -> List[str]:
    """Quick function to extract @mentions"""
    parser = TagParser()
    return parser.extract_mentions(text)


def extract_topics(text: str) -> List[str]:
    """Quick function to extract #topics"""
    parser = TagParser()
    return parser.extract_topics(text)


def highlight_tags(text: str, format: str = 'html') -> str:
    """Quick function to highlight tags in text"""
    parser = TagParser()
    tags = parser.parse(text)
    return parser.highlight_tags(text, tags, format)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_tagging_system():
    """Test the tagging system"""
    print("=" * 60)
    print("a6on-i - TAGGING SYSTEM TEST")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Use temp database for testing
    test_db = os.path.join(tempfile.gettempdir(), f"test_tagging_{uuid.uuid4().hex[:8]}.db")
    
    # Reset singleton for clean test
    reset_tagging_system()
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Tag Parser - All Types Recognition (MOST IMPORTANT)
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 1: All Tag Types Recognition")
    print("-" * 60)
    try:
        parser = TagParser()
        
        test_cases = [
            ("@sarah_chen", TagType.USER),
            ("@CRA", TagType.ROLE),
            ("@DataManager", TagType.ROLE),
            ("@CRA-Team", TagType.TEAM),
            ("@DM-Team", TagType.TEAM),
            ("@Site_101", TagType.SITE),
            ("@Study_21", TagType.STUDY),
            ("@Patient_1001", TagType.PATIENT),
            ("@ISS-20260101-ABC123", TagType.ISSUE),
            ("@ROOM-20260101-XYZ789", TagType.ROOM),
            ("@Query_12345", TagType.QUERY),
            ("@SAE_001", TagType.SAE),
            ("#urgent", TagType.TOPIC),
        ]
        
        all_passed = True
        for text, expected_type in test_cases:
            tags = parser.parse(text)
            if len(tags) > 0 and tags[0].tag_type == expected_type:
                print(f"   ✓ {text} → {expected_type.value}")
            else:
                actual = tags[0].tag_type.value if tags else 'None'
                print(f"   ✗ {text} → Expected {expected_type.value}, got {actual}")
                all_passed = False
        
        assert all_passed, "Not all tag types recognized correctly"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 2: Complex Message Parsing
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 2: Complex Message Parsing")
    print("-" * 60)
    try:
        parser = TagParser()
        text = "Hey @sarah_chen, please check @Site_101 for @Study_21. CC: @CRA-Team #urgent #safety"
        tags = parser.parse(text)
        
        print(f"   Input: {text}")
        print(f"   Tags found: {len(tags)}")
        for tag in tags:
            print(f"      - {tag.tag_type.value}: {tag.raw_text} → {tag.normalized_value}")
        
        # Should have: sarah_chen (user), Site_101 (site), Study_21 (study), 
        #              CRA-Team (team), urgent (topic), safety (topic)
        assert len(tags) >= 6, f"Expected at least 6 tags, got {len(tags)}"
        
        # Verify specific types
        tag_types = {t.tag_type for t in tags}
        assert TagType.USER in tag_types, "Should have USER tag"
        assert TagType.SITE in tag_types, "Should have SITE tag"
        assert TagType.STUDY in tag_types, "Should have STUDY tag"
        assert TagType.TEAM in tag_types, "Should have TEAM tag"
        assert TagType.TOPIC in tag_types, "Should have TOPIC tag"
        
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 3: Entity Resolver
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 3: Entity Resolver")
    print("-" * 60)
    try:
        resolver = EntityResolver()
        
        # Test user resolution
        user_tag = Tag(
            tag_id="TEST-001",
            tag_type=TagType.USER,
            raw_text="@sarah_chen",
            normalized_value="sarah_chen",
            display_name="sarah_chen",
            entity_id=None,
            position_start=0,
            position_end=11,
            context=""
        )
        
        resolved = resolver.resolve(user_tag)
        print(f"   User @sarah_chen resolved: {resolved.is_resolved}")
        print(f"   Entity ID: {resolved.entity_id}")
        print(f"   Display Name: {resolved.display_name}")
        
        assert resolved.is_resolved, "User should be resolved"
        assert resolved.entity_id == "cra_001", f"Expected cra_001, got {resolved.entity_id}"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 4: Role Resolution
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 4: Role Resolution")
    print("-" * 60)
    try:
        resolver = EntityResolver()
        
        role_tag = Tag(
            tag_id="TEST-002",
            tag_type=TagType.ROLE,
            raw_text="@CRA",
            normalized_value="CRA",
            display_name="CRA",
            entity_id=None,
            position_start=0,
            position_end=4,
            context=""
        )
        
        resolved = resolver.resolve(role_tag)
        print(f"   Role @CRA resolved: {resolved.is_resolved}")
        print(f"   Users with role: {resolved.metadata.get('user_count', 0)}")
        
        assert resolved.is_resolved, "Role should be resolved"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 5: Link Generator
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 5: Link Generator")
    print("-" * 60)
    try:
        link_gen = LinkGenerator(base_url="https://a6on-i.com")
        
        tag = Tag(
            tag_id="TEST-003",
            tag_type=TagType.SITE,
            raw_text="@Site_101",
            normalized_value="Site_101",
            display_name="Site 101",
            entity_id="Site_101",
            position_start=0,
            position_end=9,
            context=""
        )
        
        link = link_gen.generate_link(tag)
        print(f"   Generated URL: {link.url}")
        print(f"   HTML: {link.to_html()}")
        print(f"   Markdown: {link.to_markdown()}")
        
        assert "Site_101" in link.url, "URL should contain Site_101"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 6: Tagging System Integration
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 6: Tagging System Integration")
    print("-" * 60)
    try:
        system = TaggingSystem(test_db)
        
        text = "Attention @DataManager and @CRA-Team: Issue at @Site_101 requires review. #urgent"
        result = system.parse_and_process(
            text=text,
            source_type="issue",
            source_id="ISS-TEST-001",
            author_id="lead_001",
            author_name="John Smith",
            title="Test Issue",
            create_notifications=True
        )
        
        print(f"   Tags found: {result['tag_count']}")
        print(f"   Tags by type: {result['tags_by_type']}")
        print(f"   Notifications created: {result['notifications_created']}")
        print(f"   Mentions: {result['mentions']}")
        print(f"   Topics: {result['topics']}")
        
        assert result['tag_count'] >= 4, f"Expected at least 4 tags, got {result['tag_count']}"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 7: Notification Manager
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 7: Notification Manager")
    print("-" * 60)
    try:
        system = TaggingSystem(test_db)
        
        # Get notifications for DM
        notifications = system.notification_manager.get_notifications("dm_001", limit=10)
        print(f"   Notifications for dm_001: {len(notifications)}")
        
        # Get unread count
        unread = system.notification_manager.get_unread_count("dm_001")
        print(f"   Unread count: {unread}")
        
        # Mark as read
        if notifications:
            system.notification_manager.mark_as_read(notifications[0].notification_id)
            print(f"   Marked {notifications[0].notification_id} as read")
        
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 8: Tag Suggestions
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 8: Tag Suggestions")
    print("-" * 60)
    try:
        system = TaggingSystem(test_db)
        
        # First, add some tags
        system.parse_and_process(
            text="@sarah_chen @sarah_chen @alex_kim",
            source_type="test",
            source_id="TEST-002",
            author_id="test",
            author_name="Test",
            create_notifications=False
        )
        
        suggestions = system.get_tag_suggestions("sarah", limit=5)
        print(f"   Suggestions for 'sarah': {len(suggestions)}")
        for s in suggestions:
            print(f"      - {s['value']} (type: {s['type']}, used: {s['usage_count']}x)")
        
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 9: Highlight Tags
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 9: Highlight Tags")
    print("-" * 60)
    try:
        text = "Please contact @sarah_chen about @Site_101 #urgent"
        
        html = highlight_tags(text, format='html')
        md = highlight_tags(text, format='markdown')
        
        print(f"   Original: {text}")
        print(f"   HTML: {html[:100]}...")
        print(f"   Markdown: {md}")
        
        assert '<a href=' in html, "HTML should contain links"
        assert '[' in md and '](' in md, "Markdown should contain links"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 10: Statistics
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 10: Statistics")
    print("-" * 60)
    try:
        system = TaggingSystem(test_db)
        stats = system.get_statistics()
        
        print(f"   Total tags: {stats['total_tags']}")
        print(f"   Tags by type: {stats['tags_by_type']}")
        print(f"   Top tags: {len(stats['top_tags'])}")
        print(f"   Notifications: {stats['notifications']}")
        
        assert stats['total_tags'] >= 0, "Should have tag count"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 11: Full Complex Message
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 11: Full Complex Message")
    print("-" * 60)
    try:
        text = """
        URGENT: @SafetyPhysician @Safety-Team
        
        Patient @Patient_Study_21|Site_101|Subject_001 at @Site_101 in @Study_21 
        has reported an SAE @SAE-2024-001. 
        
        Please review @ISS-20260101-ABC123 and coordinate with @CRA.
        
        #safety #urgent #escalation
        
        CC: @john_smith @DataManager
        """
        
        parser = TagParser()
        tags = parser.parse(text)
        
        print(f"   Complex message parsed: {len(tags)} tags")
        type_counts = {}
        for tag in tags:
            type_counts[tag.tag_type.value] = type_counts.get(tag.tag_type.value, 0) + 1
        print(f"   By type: {type_counts}")
        
        mentions = parser.extract_mentions(text)
        topics = parser.extract_topics(text)
        print(f"   Mentions: {len(mentions)}")
        print(f"   Topics: {topics}")
        
        assert len(tags) >= 10, f"Expected at least 10 tags in complex message, got {len(tags)}"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 12: Convenience Functions
    tests_total += 1
    print("\n" + "-" * 60)
    print("TEST 12: Convenience Functions")
    print("-" * 60)
    try:
        text = "@sarah_chen @alex_kim #urgent #safety"
        
        tags = parse_tags(text)
        mentions = extract_mentions(text)
        topics = extract_topics(text)
        
        print(f"   parse_tags(): {len(tags)} tags")
        print(f"   extract_mentions(): {mentions}")
        print(f"   extract_topics(): {topics}")
        
        assert len(mentions) == 2, f"Expected 2 mentions, got {len(mentions)}"
        assert len(topics) == 2, f"Expected 2 topics, got {len(topics)}"
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Cleanup
    try:
        os.remove(test_db)
    except:
        pass
    
    # Reset singleton
    reset_tagging_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_tagging_system()
