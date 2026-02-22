# src/agents/communicator_enhanced.py
"""
a6on-i - Enhanced COMMUNICATOR Agent v1.0

Purpose: Draft communications, manage notifications, select appropriate channels,
         and handle message delivery with batching and approval workflows.

Features:
- Recipient profiling and preferences
- Intelligent channel selection
- Template-based message drafting
- Notification batching and timing optimization
- Multi-channel delivery support
- Approval workflow for sensitive communications
- Communication history tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import hashlib
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages."""
    EMAIL = "email"
    NOTIFICATION = "notification"
    ALERT = "alert"
    REPORT = "report"
    REMINDER = "reminder"
    ESCALATION = "escalation"
    SUMMARY = "summary"
    NEWSLETTER = "newsletter"


class MessagePriority(Enum):
    """Priority levels for messages."""
    URGENT = "urgent"        # Immediate delivery
    HIGH = "high"            # Within 1 hour
    NORMAL = "normal"        # Within 4 hours
    LOW = "low"              # Batch with daily digest


class Channel(Enum):
    """Communication channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"
    DASHBOARD = "dashboard"


class RecipientRole(Enum):
    """Recipient roles for targeting."""
    CRA = "CRA"
    DATA_MANAGER = "Data Manager"
    SITE_COORDINATOR = "Site Coordinator"
    SAFETY_DATA_MANAGER = "Safety Data Manager"
    SAFETY_PHYSICIAN = "Safety Physician"
    MEDICAL_CODER = "Medical Coder"
    STUDY_LEAD = "Study Lead"
    CTM = "Clinical Trial Manager"
    SPONSOR = "Sponsor"
    PI = "Principal Investigator"
    SITE = "Site"


class DeliveryStatus(Enum):
    """Message delivery status."""
    DRAFT = "draft"
    QUEUED = "queued"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    READ = "read"


@dataclass
class RecipientProfile:
    """Profile for a message recipient."""
    recipient_id: str
    name: str
    email: str
    role: RecipientRole
    phone: Optional[str] = None
    preferred_channel: Channel = Channel.EMAIL
    preferred_language: str = "en"
    timezone: str = "UTC"
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    do_not_disturb_hours: List[Tuple[int, int]] = field(default_factory=list)
    batch_notifications: bool = True
    max_daily_messages: int = 20
    sites: List[str] = field(default_factory=list)
    studies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "recipient_id": self.recipient_id,
            "name": self.name,
            "email": self.email,
            "role": self.role.value,
            "phone": self.phone,
            "preferred_channel": self.preferred_channel.value,
            "preferred_language": self.preferred_language,
            "timezone": self.timezone,
            "batch_notifications": self.batch_notifications,
            "sites": self.sites,
            "studies": self.studies
        }


@dataclass
class MessageTemplate:
    """Template for message generation."""
    template_id: str
    name: str
    message_type: MessageType
    subject_template: str
    body_template: str
    priority: MessagePriority = MessagePriority.NORMAL
    target_roles: List[RecipientRole] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)
    requires_approval: bool = False
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> Tuple[str, str]:
        """Render template with context."""
        subject = self.subject_template
        body = self.body_template
        
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))
        
        return subject, body
    
    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "message_type": self.message_type.value,
            "subject_template": self.subject_template,
            "priority": self.priority.value,
            "target_roles": [r.value for r in self.target_roles],
            "channels": [c.value for c in self.channels],
            "requires_approval": self.requires_approval,
            "variables": self.variables
        }


@dataclass
class Message:
    """A message to be sent."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    subject: str
    body: str
    sender: str
    recipients: List[RecipientProfile]
    channels: List[Channel]
    status: DeliveryStatus = DeliveryStatus.DRAFT
    template_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    scheduled_time: Optional[datetime] = None
    sent_time: Optional[datetime] = None
    delivered_time: Optional[datetime] = None
    read_time: Optional[datetime] = None
    requires_approval: bool = False
    approval_status: Optional[str] = None
    approved_by: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "subject": self.subject,
            "body": self.body[:200] + "..." if len(self.body) > 200 else self.body,
            "sender": self.sender,
            "recipients": [r.name for r in self.recipients],
            "recipient_count": len(self.recipients),
            "channels": [c.value for c in self.channels],
            "status": self.status.value,
            "template_id": self.template_id,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "sent_time": self.sent_time.isoformat() if self.sent_time else None,
            "requires_approval": self.requires_approval,
            "approval_status": self.approval_status,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class NotificationBatch:
    """A batch of notifications to be sent together."""
    batch_id: str
    recipient: RecipientProfile
    messages: List[Message] = field(default_factory=list)
    batch_type: str = "daily_digest"  # daily_digest, urgent_batch, weekly_summary
    scheduled_time: datetime = field(default_factory=datetime.now)
    status: DeliveryStatus = DeliveryStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, message: Message):
        self.messages.append(message)
    
    @property
    def total_messages(self) -> int:
        return len(self.messages)
    
    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "recipient": self.recipient.name,
            "recipient_email": self.recipient.email,
            "batch_type": self.batch_type,
            "total_messages": self.total_messages,
            "scheduled_time": self.scheduled_time.isoformat(),
            "status": self.status.value,
            "message_types": [m.message_type.value for m in self.messages]
        }


@dataclass
class CommunicationResult:
    """Result of communication operations."""
    result_id: str
    query: str
    messages_drafted: int = 0
    messages_queued: int = 0
    messages_sent: int = 0
    messages_pending_approval: int = 0
    messages_failed: int = 0
    batches_created: int = 0
    messages: List[Message] = field(default_factory=list)
    batches: List[NotificationBatch] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "query": self.query,
            "messages_drafted": self.messages_drafted,
            "messages_queued": self.messages_queued,
            "messages_sent": self.messages_sent,
            "messages_pending_approval": self.messages_pending_approval,
            "messages_failed": self.messages_failed,
            "batches_created": self.batches_created,
            "messages": [m.to_dict() for m in self.messages],
            "batches": [b.to_dict() for b in self.batches],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


class RecipientManager:
    """Manages recipient profiles and preferences."""
    
    # Default profiles for testing
    DEFAULT_PROFILES = {
        "cra_001": RecipientProfile(
            recipient_id="cra_001",
            name="Sarah Chen",
            email="sarah.chen@example.com",
            role=RecipientRole.CRA,
            preferred_channel=Channel.EMAIL,
            sites=["Site_1", "Site_2", "Site_3"],
            studies=["Study_21", "Study_22"]
        ),
        "dm_001": RecipientProfile(
            recipient_id="dm_001",
            name="Alex Kim",
            email="alex.kim@example.com",
            role=RecipientRole.DATA_MANAGER,
            preferred_channel=Channel.EMAIL,
            batch_notifications=True,
            studies=["Study_21", "Study_22"]
        ),
        "safety_001": RecipientProfile(
            recipient_id="safety_001",
            name="Dr. Maria Garcia",
            email="maria.garcia@example.com",
            role=RecipientRole.SAFETY_PHYSICIAN,
            preferred_channel=Channel.EMAIL,
            batch_notifications=False
        ),
        "lead_001": RecipientProfile(
            recipient_id="lead_001",
            name="John Smith",
            email="john.smith@example.com",
            role=RecipientRole.STUDY_LEAD,
            preferred_channel=Channel.EMAIL
        ),
        "site_001": RecipientProfile(
            recipient_id="site_001",
            name="Site Coordinator - Site 1",
            email="coordinator@site1.example.com",
            role=RecipientRole.SITE_COORDINATOR,
            preferred_channel=Channel.EMAIL,
            sites=["Site_1"]
        ),
        "ctm_001": RecipientProfile(
            recipient_id="ctm_001",
            name="Emily Brown",
            email="emily.brown@example.com",
            role=RecipientRole.CTM,
            preferred_channel=Channel.EMAIL
        ),
        "coder_001": RecipientProfile(
            recipient_id="coder_001",
            name="Mike Johnson",
            email="mike.johnson@example.com",
            role=RecipientRole.MEDICAL_CODER,
            preferred_channel=Channel.EMAIL,
            batch_notifications=True
        )
    }
    
    def __init__(self):
        self.profiles: Dict[str, RecipientProfile] = dict(self.DEFAULT_PROFILES)
    
    def get_profile(self, recipient_id: str) -> Optional[RecipientProfile]:
        """Get a recipient profile by ID."""
        return self.profiles.get(recipient_id)
    
    def get_profiles_by_role(self, role: RecipientRole) -> List[RecipientProfile]:
        """Get all profiles with a specific role."""
        return [p for p in self.profiles.values() if p.role == role]
    
    def get_profiles_for_site(self, site_id: str) -> List[RecipientProfile]:
        """Get all profiles associated with a site."""
        return [p for p in self.profiles.values() if site_id in p.sites]
    
    def get_profiles_for_study(self, study_id: str) -> List[RecipientProfile]:
        """Get all profiles associated with a study."""
        return [p for p in self.profiles.values() if study_id in p.studies]
    
    def add_profile(self, profile: RecipientProfile):
        """Add or update a recipient profile."""
        self.profiles[profile.recipient_id] = profile
    
    def get_preferred_channel(self, recipient_id: str) -> Channel:
        """Get preferred channel for a recipient."""
        profile = self.get_profile(recipient_id)
        return profile.preferred_channel if profile else Channel.EMAIL


class TemplateLibrary:
    """Library of message templates."""
    
    TEMPLATES = {
        # Query-related templates
        "query_reminder": MessageTemplate(
            template_id="TPL-QRY-001",
            name="Query Resolution Reminder",
            message_type=MessageType.REMINDER,
            subject_template="Action Required: {{query_count}} Open Queries at {{site_id}}",
            body_template="""
Dear {{recipient_name}},

This is a reminder that there are {{query_count}} open queries requiring your attention at {{site_id}}.

Summary:
- Queries aged > 14 days: {{aged_queries}}
- Queries aged > 7 days: {{recent_queries}}

Please log into the EDC system to review and respond to these queries.

If you have any questions, please contact your CRA.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.NORMAL,
            target_roles=[RecipientRole.SITE_COORDINATOR, RecipientRole.PI],
            channels=[Channel.EMAIL],
            variables=["recipient_name", "query_count", "site_id", "aged_queries", "recent_queries"]
        ),
        
        # Signature reminders
        "signature_reminder": MessageTemplate(
            template_id="TPL-SIG-001",
            name="Signature Reminder",
            message_type=MessageType.REMINDER,
            subject_template="Signature Required: {{signature_count}} CRFs Awaiting PI Signature",
            body_template="""
Dear {{recipient_name}},

{{signature_count}} CRFs are awaiting your signature at {{site_id}}.

Please log into the EDC system at your earliest convenience to complete the signatures.

Overdue signatures may impact study timelines and data lock readiness.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.HIGH,
            target_roles=[RecipientRole.PI],
            channels=[Channel.EMAIL],
            variables=["recipient_name", "signature_count", "site_id"]
        ),
        
        # SAE alerts
        "sae_alert": MessageTemplate(
            template_id="TPL-SAE-001",
            name="SAE Alert",
            message_type=MessageType.ALERT,
            subject_template="URGENT: SAE Requires Immediate Attention - {{subject_id}}",
            body_template="""
URGENT SAFETY ALERT

Subject: {{subject_id}}
Site: {{site_id}}
Study: {{study_id}}

An SAE has been reported and requires your immediate attention.

SAE Details:
- Onset Date: {{onset_date}}
- Event: {{event_description}}
- Status: {{status}}

Please review this case immediately in the safety database.

This is an automated alert. Do not reply to this email.
            """.strip(),
            priority=MessagePriority.URGENT,
            target_roles=[RecipientRole.SAFETY_PHYSICIAN, RecipientRole.SAFETY_DATA_MANAGER],
            channels=[Channel.EMAIL, Channel.SMS],
            requires_approval=False,
            variables=["subject_id", "site_id", "study_id", "onset_date", "event_description", "status"]
        ),
        
        # Site performance report
        "site_performance": MessageTemplate(
            template_id="TPL-RPT-001",
            name="Site Performance Report",
            message_type=MessageType.REPORT,
            subject_template="Weekly Site Performance Report: {{site_id}}",
            body_template="""
Site Performance Report
=======================

Site: {{site_id}}
Report Period: {{report_period}}
Generated: {{generated_date}}

KEY METRICS:
- DQI Score: {{dqi_score}}
- Clean Patient Rate: {{clean_rate}}%
- Open Queries: {{open_queries}}
- Pending Signatures: {{pending_signatures}}

TREND:
{{trend_summary}}

RECOMMENDATIONS:
{{recommendations}}

For detailed analysis, please visit the a6on-i dashboard.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.LOW,
            target_roles=[RecipientRole.CRA, RecipientRole.CTM],
            channels=[Channel.EMAIL],
            variables=["site_id", "report_period", "generated_date", "dqi_score", 
                      "clean_rate", "open_queries", "pending_signatures", 
                      "trend_summary", "recommendations"]
        ),
        
        # Escalation notice
        "escalation": MessageTemplate(
            template_id="TPL-ESC-001",
            name="Issue Escalation",
            message_type=MessageType.ESCALATION,
            subject_template="ESCALATION: {{issue_type}} at {{site_id}} - Action Required",
            body_template="""
ESCALATION NOTICE

This issue has been escalated due to: {{escalation_reason}}

Issue Details:
- Site: {{site_id}}
- Issue Type: {{issue_type}}
- Duration: {{days_outstanding}} days outstanding
- Impact: {{impact_description}}

Previous Actions:
{{previous_actions}}

Required Action:
{{required_action}}

Please respond within 24 hours.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.HIGH,
            target_roles=[RecipientRole.STUDY_LEAD, RecipientRole.CTM],
            channels=[Channel.EMAIL],
            requires_approval=True,
            variables=["site_id", "issue_type", "escalation_reason", "days_outstanding",
                      "impact_description", "previous_actions", "required_action"]
        ),
        
        # Daily digest
        "daily_digest": MessageTemplate(
            template_id="TPL-DIG-001",
            name="Daily Digest",
            message_type=MessageType.SUMMARY,
            subject_template="Your Daily a6on-i Digest - {{date}}",
            body_template="""
Good morning, {{recipient_name}}!

Here's your daily summary for {{date}}:

PRIORITY ITEMS:
{{priority_items}}

YOUR SITES SUMMARY:
{{sites_summary}}

ACTIONS COMPLETED YESTERDAY:
{{completed_actions}}

UPCOMING:
{{upcoming_items}}

Have a productive day!
a6on-i System
            """.strip(),
            priority=MessagePriority.LOW,
            target_roles=[RecipientRole.CRA, RecipientRole.DATA_MANAGER],
            channels=[Channel.EMAIL],
            variables=["recipient_name", "date", "priority_items", "sites_summary",
                      "completed_actions", "upcoming_items"]
        ),
        
        # DB Lock readiness
        "dblock_update": MessageTemplate(
            template_id="TPL-DBL-001",
            name="DB Lock Readiness Update",
            message_type=MessageType.REPORT,
            subject_template="DB Lock Status Update: {{ready_rate}}% Ready",
            body_template="""
Database Lock Readiness Update
==============================

Study: {{study_id}}
As of: {{report_date}}

CURRENT STATUS:
- Patients Ready: {{patients_ready}} ({{ready_rate}}%)
- Patients Pending: {{patients_pending}}
- Patients Blocked: {{patients_blocked}}

PROJECTED TIMELINE:
- Estimated Completion: {{estimated_date}}
- Confidence: {{confidence}}%

TOP BLOCKERS:
{{blockers}}

RECOMMENDED ACTIONS:
{{recommendations}}

For detailed analysis, visit the DB Lock Dashboard.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.NORMAL,
            target_roles=[RecipientRole.STUDY_LEAD, RecipientRole.DATA_MANAGER],
            channels=[Channel.EMAIL],
            variables=["study_id", "report_date", "patients_ready", "ready_rate",
                      "patients_pending", "patients_blocked", "estimated_date",
                      "confidence", "blockers", "recommendations"]
        ),
        
        # Action assignment
        "action_assignment": MessageTemplate(
            template_id="TPL-ACT-001",
            name="Action Assignment",
            message_type=MessageType.NOTIFICATION,
            subject_template="New Action Assigned: {{action_title}}",
            body_template="""
New Action Assignment
====================

You have been assigned a new action:

Action: {{action_title}}
Priority: {{priority}}
Due Date: {{due_date}}
Entity: {{entity_id}}

Description:
{{description}}

Steps:
{{steps}}

Please complete this action by the due date.

Best regards,
a6on-i System
            """.strip(),
            priority=MessagePriority.NORMAL,
            target_roles=[RecipientRole.CRA, RecipientRole.DATA_MANAGER, RecipientRole.SITE_COORDINATOR],
            channels=[Channel.EMAIL, Channel.IN_APP],
            variables=["action_title", "priority", "due_date", "entity_id", "description", "steps"]
        )
    }
    
    def __init__(self):
        self.templates = dict(self.TEMPLATES)
    
    def get_template(self, template_id: str) -> Optional[MessageTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, message_type: MessageType) -> List[MessageTemplate]:
        """Get all templates of a specific type."""
        return [t for t in self.templates.values() if t.message_type == message_type]
    
    def get_templates_for_role(self, role: RecipientRole) -> List[MessageTemplate]:
        """Get all templates targeting a specific role."""
        return [t for t in self.templates.values() if role in t.target_roles]
    
    def search_templates(self, query: str) -> List[MessageTemplate]:
        """Search templates by keyword."""
        query_lower = query.lower()
        return [
            t for t in self.templates.values()
            if query_lower in t.name.lower() or any(query_lower in tag for tag in t.tags)
        ]


class ChannelSelector:
    """Selects appropriate communication channels."""
    
    # Channel priority by message type and priority
    CHANNEL_PRIORITY = {
        (MessageType.ALERT, MessagePriority.URGENT): [Channel.SMS, Channel.EMAIL, Channel.IN_APP],
        (MessageType.ALERT, MessagePriority.HIGH): [Channel.EMAIL, Channel.IN_APP],
        (MessageType.ESCALATION, MessagePriority.HIGH): [Channel.EMAIL],
        (MessageType.REMINDER, MessagePriority.NORMAL): [Channel.EMAIL, Channel.IN_APP],
        (MessageType.REPORT, MessagePriority.LOW): [Channel.EMAIL],
        (MessageType.SUMMARY, MessagePriority.LOW): [Channel.EMAIL],
        (MessageType.NOTIFICATION, MessagePriority.NORMAL): [Channel.IN_APP, Channel.EMAIL],
    }
    
    def select_channels(
        self,
        message_type: MessageType,
        priority: MessagePriority,
        recipient: RecipientProfile
    ) -> List[Channel]:
        """Select appropriate channels for a message."""
        # Get default channels for message type and priority
        key = (message_type, priority)
        default_channels = self.CHANNEL_PRIORITY.get(key, [Channel.EMAIL])
        
        # Consider recipient preferences
        preferred = recipient.preferred_channel
        
        # Urgent messages override preferences
        if priority == MessagePriority.URGENT:
            return default_channels
        
        # For normal messages, prefer recipient's channel if in defaults
        if preferred in default_channels:
            return [preferred]
        
        # Otherwise use first default
        return [default_channels[0]] if default_channels else [Channel.EMAIL]
    
    def is_channel_available(self, channel: Channel, recipient: RecipientProfile) -> bool:
        """Check if a channel is available for a recipient."""
        if channel == Channel.SMS and not recipient.phone:
            return False
        if channel == Channel.EMAIL and not recipient.email:
            return False
        return True


class NotificationBatcher:
    """Batches notifications for efficient delivery."""
    
    def __init__(self):
        self._batch_counter = 0
        self.batches: Dict[str, NotificationBatch] = {}
        self.recipient_queues: Dict[str, List[Message]] = defaultdict(list)
    
    def _generate_batch_id(self) -> str:
        self._batch_counter += 1
        return f"BATCH-{self._batch_counter:04d}"
    
    def should_batch(self, message: Message, recipient: RecipientProfile) -> bool:
        """Determine if a message should be batched."""
        # Never batch urgent messages
        if message.priority == MessagePriority.URGENT:
            return False
        
        # Never batch alerts
        if message.message_type == MessageType.ALERT:
            return False
        
        # Check recipient preferences
        if not recipient.batch_notifications:
            return False
        
        return True
    
    def add_to_batch(self, message: Message, recipient: RecipientProfile) -> Optional[str]:
        """Add a message to the batch queue."""
        if not self.should_batch(message, recipient):
            return None
        
        self.recipient_queues[recipient.recipient_id].append(message)
        return recipient.recipient_id
    
    def create_batch(
        self,
        recipient: RecipientProfile,
        batch_type: str = "daily_digest",
        scheduled_time: Optional[datetime] = None
    ) -> NotificationBatch:
        """Create a batch from queued messages."""
        messages = self.recipient_queues.get(recipient.recipient_id, [])
        
        batch = NotificationBatch(
            batch_id=self._generate_batch_id(),
            recipient=recipient,
            messages=messages,
            batch_type=batch_type,
            scheduled_time=scheduled_time or datetime.now() + timedelta(hours=1)
        )
        
        self.batches[batch.batch_id] = batch
        
        # Clear the queue
        self.recipient_queues[recipient.recipient_id] = []
        
        return batch
    
    def get_pending_batches(self) -> List[NotificationBatch]:
        """Get all pending batches ready for delivery."""
        now = datetime.now()
        return [
            b for b in self.batches.values()
            if b.status == DeliveryStatus.QUEUED and b.scheduled_time <= now
        ]
    
    def get_queue_size(self, recipient_id: str) -> int:
        """Get the number of messages in queue for a recipient."""
        return len(self.recipient_queues.get(recipient_id, []))


class EnhancedCommunicatorAgent:
    """
    Enhanced COMMUNICATOR Agent for message drafting and delivery.
    
    Capabilities:
    - Recipient profiling and management
    - Intelligent channel selection
    - Template-based message drafting
    - Notification batching
    - Approval workflow for sensitive communications
    - Multi-channel delivery
    """
    
    def __init__(self, llm_wrapper=None):
        self.recipient_manager = RecipientManager()
        self.template_library = TemplateLibrary()
        self.channel_selector = ChannelSelector()
        self.batcher = NotificationBatcher()
        self.llm = llm_wrapper
        self._message_counter = 0
        self._result_counter = 0
        self.message_history: Dict[str, Message] = {}
        
        logger.info("EnhancedCommunicatorAgent initialized")
    
    def _generate_message_id(self) -> str:
        self._message_counter += 1
        return f"MSG-{self._message_counter:04d}"
    
    def _generate_result_id(self) -> str:
        self._result_counter += 1
        return f"COM-{self._result_counter:04d}"
    
    def draft_message(
        self,
        template_id: str,
        recipients: List[str],
        context: Dict[str, Any],
        sender: str = "system"
    ) -> Message:
        """Draft a message using a template."""
        template = self.template_library.get_template(template_id)
        
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Resolve recipients
        recipient_profiles = []
        for r_id in recipients:
            profile = self.recipient_manager.get_profile(r_id)
            if profile:
                recipient_profiles.append(profile)
        
        if not recipient_profiles:
            raise ValueError("No valid recipients found")
        
        # Render template
        # Add recipient name to context for first recipient
        context['recipient_name'] = recipient_profiles[0].name
        subject, body = template.render(context)
        
        # Select channels based on first recipient (could be enhanced for multi-recipient)
        channels = self.channel_selector.select_channels(
            template.message_type,
            template.priority,
            recipient_profiles[0]
        )
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=template.message_type,
            priority=template.priority,
            subject=subject,
            body=body,
            sender=sender,
            recipients=recipient_profiles,
            channels=channels,
            status=DeliveryStatus.DRAFT,
            template_id=template_id,
            context=context,
            requires_approval=template.requires_approval
        )
        
        self.message_history[message.message_id] = message
        
        return message
    
    def draft_custom_message(
        self,
        message_type: MessageType,
        priority: MessagePriority,
        subject: str,
        body: str,
        recipients: List[str],
        sender: str = "system",
        channels: Optional[List[Channel]] = None
    ) -> Message:
        """Draft a custom message without a template."""
        # Resolve recipients
        recipient_profiles = []
        for r_id in recipients:
            profile = self.recipient_manager.get_profile(r_id)
            if profile:
                recipient_profiles.append(profile)
        
        if not recipient_profiles:
            raise ValueError("No valid recipients found")
        
        # Select channels if not provided
        if not channels:
            channels = self.channel_selector.select_channels(
                message_type,
                priority,
                recipient_profiles[0]
            )
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=message_type,
            priority=priority,
            subject=subject,
            body=body,
            sender=sender,
            recipients=recipient_profiles,
            channels=channels,
            status=DeliveryStatus.DRAFT
        )
        
        self.message_history[message.message_id] = message
        
        return message
    
    def draft_for_role(
        self,
        template_id: str,
        role: RecipientRole,
        context: Dict[str, Any],
        sender: str = "system"
    ) -> List[Message]:
        """Draft messages for all recipients with a specific role."""
        profiles = self.recipient_manager.get_profiles_by_role(role)
        messages = []
        
        for profile in profiles:
            try:
                msg = self.draft_message(
                    template_id=template_id,
                    recipients=[profile.recipient_id],
                    context=context,
                    sender=sender
                )
                messages.append(msg)
            except Exception as e:
                logger.warning(f"Failed to draft for {profile.recipient_id}: {e}")
        
        return messages
    
    def draft_for_site(
        self,
        template_id: str,
        site_id: str,
        context: Dict[str, Any],
        sender: str = "system"
    ) -> List[Message]:
        """Draft messages for all recipients associated with a site."""
        profiles = self.recipient_manager.get_profiles_for_site(site_id)
        
        # Add site_id to context
        context['site_id'] = site_id
        
        messages = []
        for profile in profiles:
            try:
                msg = self.draft_message(
                    template_id=template_id,
                    recipients=[profile.recipient_id],
                    context=context,
                    sender=sender
                )
                messages.append(msg)
            except Exception as e:
                logger.warning(f"Failed to draft for {profile.recipient_id}: {e}")
        
        return messages
    
    def queue_message(self, message: Message) -> str:
        """Queue a message for delivery."""
        if message.requires_approval and message.approval_status != "approved":
            message.status = DeliveryStatus.PENDING_APPROVAL
            return "pending_approval"
        
        # Check if should batch
        for recipient in message.recipients:
            if self.batcher.should_batch(message, recipient):
                self.batcher.add_to_batch(message, recipient)
                message.status = DeliveryStatus.QUEUED
                return "batched"
        
        # Queue for immediate delivery
        message.status = DeliveryStatus.QUEUED
        message.scheduled_time = datetime.now()
        return "queued"
    
    def send_message(self, message: Message) -> bool:
        """Send a message (simulated)."""
        if message.status not in [DeliveryStatus.QUEUED, DeliveryStatus.APPROVED]:
            return False
        
        try:
            message.status = DeliveryStatus.SENDING
            
            # Simulate sending
            # In production, would integrate with email/SMS/notification services
            
            message.status = DeliveryStatus.SENT
            message.sent_time = datetime.now()
            
            # Simulate delivery confirmation
            message.status = DeliveryStatus.DELIVERED
            message.delivered_time = datetime.now()
            
            return True
            
        except Exception as e:
            message.status = DeliveryStatus.FAILED
            message.error_message = str(e)
            message.retry_count += 1
            return False
    
    def approve_message(self, message_id: str, approver: str) -> bool:
        """Approve a message for sending."""
        message = self.message_history.get(message_id)
        
        if not message:
            return False
        
        if message.status != DeliveryStatus.PENDING_APPROVAL:
            return False
        
        message.approval_status = "approved"
        message.approved_by = approver
        message.status = DeliveryStatus.APPROVED
        
        return True
    
    def reject_message(self, message_id: str, rejector: str, reason: str = "") -> bool:
        """Reject a message."""
        message = self.message_history.get(message_id)
        
        if not message:
            return False
        
        message.approval_status = "rejected"
        message.approved_by = rejector
        message.status = DeliveryStatus.REJECTED
        message.error_message = reason
        
        return True
    
    def create_digest(self, recipient_id: str) -> Optional[NotificationBatch]:
        """Create a digest batch for a recipient."""
        profile = self.recipient_manager.get_profile(recipient_id)
        
        if not profile:
            return None
        
        queue_size = self.batcher.get_queue_size(recipient_id)
        
        if queue_size == 0:
            return None
        
        return self.batcher.create_batch(
            recipient=profile,
            batch_type="daily_digest",
            scheduled_time=datetime.now()
        )
    
    def get_pending_approvals(self) -> List[Message]:
        """Get all messages pending approval."""
        return [
            m for m in self.message_history.values()
            if m.status == DeliveryStatus.PENDING_APPROVAL
        ]
    
    def get_message_stats(self) -> Dict[str, Any]:
        """Get statistics on messages."""
        messages = list(self.message_history.values())
        
        stats = {
            "total": len(messages),
            "by_status": {},
            "by_type": {},
            "by_priority": {}
        }
        
        for msg in messages:
            # By status
            status = msg.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # By type
            msg_type = msg.message_type.value
            stats["by_type"][msg_type] = stats["by_type"].get(msg_type, 0) + 1
            
            # By priority
            priority = msg.priority.value
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1
        
        return stats
    
    def communicate_from_query(self, query: str) -> CommunicationResult:
        """Handle communication based on natural language query."""
        start_time = datetime.now()
        query_lower = query.lower()
        
        result = CommunicationResult(
            result_id=self._generate_result_id(),
            query=query
        )
        
        # Determine intent
        if 'reminder' in query_lower and 'query' in query_lower:
            # Query reminder
            result = self._handle_query_reminder(query, result)
        
        elif 'reminder' in query_lower and 'signature' in query_lower:
            # Signature reminder
            result = self._handle_signature_reminder(query, result)
        
        elif 'sae' in query_lower or 'safety' in query_lower:
            # SAE alert
            result = self._handle_sae_alert(query, result)
        
        elif 'escalat' in query_lower:
            # Escalation
            result = self._handle_escalation(query, result)
        
        elif 'digest' in query_lower or 'summary' in query_lower:
            # Daily digest
            result = self._handle_digest(query, result)
        
        elif 'status' in query_lower:
            # Show message stats
            stats = self.get_message_stats()
            result.summary = f"Message Statistics: {stats['total']} total messages"
            result.recommendations = [
                f"By status: {stats['by_status']}",
                f"By type: {stats['by_type']}",
                f"By priority: {stats['by_priority']}"
            ]
        
        elif 'pending' in query_lower and 'approval' in query_lower:
            # Show pending approvals
            pending = self.get_pending_approvals()
            result.messages = pending
            result.messages_pending_approval = len(pending)
            result.summary = f"{len(pending)} messages pending approval"
        
        else:
            result.summary = "Communication intent not recognized"
            result.recommendations = [
                "Try: 'Send query reminder to Site_1'",
                "Try: 'Draft signature reminder'",
                "Try: 'Create daily digest for CRA team'",
                "Try: 'Show message status'"
            ]
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _handle_query_reminder(self, query: str, result: CommunicationResult) -> CommunicationResult:
        """Handle query reminder request."""
        # Extract site if mentioned
        site_id = self._extract_site(query) or "Site_1"
        
        context = {
            "query_count": 15,
            "site_id": site_id,
            "aged_queries": 5,
            "recent_queries": 10
        }
        
        try:
            messages = self.draft_for_site("query_reminder", site_id, context)
            
            for msg in messages:
                self.queue_message(msg)
            
            result.messages = messages
            result.messages_drafted = len(messages)
            result.messages_queued = len([m for m in messages if m.status == DeliveryStatus.QUEUED])
            result.summary = f"Drafted {len(messages)} query reminder(s) for {site_id}"
            
        except Exception as e:
            result.summary = f"Failed to create reminders: {e}"
        
        return result
    
    def _handle_signature_reminder(self, query: str, result: CommunicationResult) -> CommunicationResult:
        """Handle signature reminder request."""
        site_id = self._extract_site(query) or "Site_1"
        
        context = {
            "signature_count": 8,
            "site_id": site_id
        }
        
        try:
            # Target PI role
            messages = self.draft_for_role("signature_reminder", RecipientRole.PI, context)
            
            for msg in messages:
                self.queue_message(msg)
            
            result.messages = messages
            result.messages_drafted = len(messages)
            result.messages_queued = len(messages)
            result.summary = f"Drafted {len(messages)} signature reminder(s)"
            
        except Exception as e:
            result.summary = f"Failed to create reminders: {e}"
        
        return result
    
    def _handle_sae_alert(self, query: str, result: CommunicationResult) -> CommunicationResult:
        """Handle SAE alert request."""
        context = {
            "subject_id": "Subject_001",
            "site_id": "Site_1",
            "study_id": "Study_21",
            "onset_date": datetime.now().strftime("%Y-%m-%d"),
            "event_description": "Serious adverse event reported",
            "status": "Pending Review"
        }
        
        try:
            messages = self.draft_for_role("sae_alert", RecipientRole.SAFETY_PHYSICIAN, context)
            
            # SAE alerts are urgent - send immediately
            for msg in messages:
                msg.status = DeliveryStatus.QUEUED
                self.send_message(msg)
            
            result.messages = messages
            result.messages_drafted = len(messages)
            result.messages_sent = len(messages)
            result.summary = f"Sent {len(messages)} SAE alert(s)"
            
        except Exception as e:
            result.summary = f"Failed to send alerts: {e}"
        
        return result
    
    def _handle_escalation(self, query: str, result: CommunicationResult) -> CommunicationResult:
        """Handle escalation request."""
        site_id = self._extract_site(query) or "Site_1"
        
        context = {
            "site_id": site_id,
            "issue_type": "Open Queries",
            "escalation_reason": "No response for 14+ days",
            "days_outstanding": 21,
            "impact_description": "Blocking DB Lock progress",
            "previous_actions": "- Reminder sent on Day 7\n- Follow-up on Day 14",
            "required_action": "Contact site immediately to resolve queries"
        }
        
        try:
            messages = self.draft_for_role("escalation", RecipientRole.STUDY_LEAD, context)
            
            for msg in messages:
                self.queue_message(msg)
            
            result.messages = messages
            result.messages_drafted = len(messages)
            result.messages_pending_approval = len([m for m in messages if m.requires_approval])
            result.summary = f"Drafted {len(messages)} escalation notice(s) - requires approval"
            
        except Exception as e:
            result.summary = f"Failed to create escalation: {e}"
        
        return result
    
    def _handle_digest(self, query: str, result: CommunicationResult) -> CommunicationResult:
        """Handle digest request."""
        # Create digests for CRAs
        profiles = self.recipient_manager.get_profiles_by_role(RecipientRole.CRA)
        
        batches = []
        for profile in profiles:
            # Add some sample messages to queue
            sample_msg = self.draft_custom_message(
                message_type=MessageType.NOTIFICATION,
                priority=MessagePriority.LOW,
                subject="Sample notification",
                body="This is a sample notification for the digest",
                recipients=[profile.recipient_id]
            )
            self.batcher.add_to_batch(sample_msg, profile)
            
            batch = self.create_digest(profile.recipient_id)
            if batch:
                batches.append(batch)
        
        result.batches = batches
        result.batches_created = len(batches)
        result.summary = f"Created {len(batches)} digest batch(es)"
        
        return result
    
    def _extract_site(self, query: str) -> Optional[str]:
        """Extract site ID from query."""
        import re
        match = re.search(r'Site[_\s]?(\d+)', query, re.IGNORECASE)
        if match:
            return f"Site_{match.group(1)}"
        return None
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Main processing method for orchestrator integration."""
        result = self.communicate_from_query(query)
        return result.to_dict()


def get_communicator_agent(llm_wrapper=None) -> EnhancedCommunicatorAgent:
    """Factory function to get communicator agent instance."""
    return EnhancedCommunicatorAgent(llm_wrapper=llm_wrapper)
