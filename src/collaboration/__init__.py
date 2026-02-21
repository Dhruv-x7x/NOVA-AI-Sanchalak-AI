"""
SANCHALAK AI - Collaboration Module
Phase 8: Collaboration Hub

Includes:
- 8.1 Issue Registry
- 8.2 Investigation Rooms
- 8.3 @Tagging System
- 8.4 5-Level Escalation Engine
- 8.5 Team Workspaces
- 8.6 Alert System
"""

# =========================
# Phase 8.1: Issue Registry
# =========================
from .issue_registry import (
    IssueRegistry,
    Issue,
    IssueComment,
    IssueAuditEntry,
    IssueFilter,
    IssueStatus,
    IssuePriority,
    IssueSeverity,
    IssueCategory,
    AssigneeRole,
    AuditAction,
    get_issue_registry,
)

# ==============================
# Phase 8.2: Investigation Rooms
# ==============================
from .investigation_rooms import (
    InvestigationRoomsManager,
    InvestigationRoom,
    RoomParticipant,
    RoomContext,
    TimelineEvent,
    Evidence,
    DiscussionThread,
    ThreadMessage,
    RootCauseVote,
    Resolution,
    RoomStatus,
    RoomType,
    ParticipantRole,
    EvidenceType,
    EvidenceStrength,
    ThreadType,
    VoteType,
    ResolutionType,
    get_investigation_rooms_manager,
)

# =========================
# Phase 8.3: @Tagging System
# =========================
from .tagging_system import (
    TaggingSystem,
    Tag,
    TagType,
    Notification,
    NotificationPriority as TagNotificationPriority,
    EntityLink,
    TagParser,
    EntityResolver,
    NotificationManager,
    LinkGenerator,
    get_tagging_system,
    parse_tags,
    extract_mentions,
    extract_topics,
    highlight_tags,
)

# =================================
# Phase 8.4: 5-Level Escalation Engine
# =================================
from .escalation_engine import (
    EscalationEngine,
    Escalation,
    EscalationRule,
    EscalationImpact,
    SponsorPackage,
    EscalationFilter,
    EscalationLevel,
    EscalationStatus,
    EscalationTrigger,
    NotificationPriority as EscalationNotificationPriority,
    get_escalation_engine,
    reset_escalation_engine,
    check_escalations,
    get_active_escalations,
    escalate_issue,
    get_escalation_stats,
)

# =========================
# Phase 8.5: Team Workspaces
# =========================
from .team_workspaces import (
    TeamWorkspacesManager,
    Workspace,
    WorkspaceMember,
    WorkspaceGoal,
    ActivityFeedItem,
    Announcement,
    SharedResource,
    WorkspaceMetrics,
    WorkspaceType,
    WorkspaceStatus,
    MemberRole,
    GoalStatus,
    GoalPriority,
    ActivityType,
    ResourceType,
    get_team_workspaces_manager,
    reset_team_workspaces_manager,
)

# ======================
# Phase 8.6: Alert System
# ======================
from .alert_system import (
    AlertSystem,
    Alert,
    AlertBatch,
    UserAlertPreferences,
    AlertStats,
    FatigueStatus,
    AlertPriority,
    AlertChannel,
    AlertStatus,
    AlertCategory,
    BatchType,
    SuppressionReason,
    PriorityClassifier,
    ChannelRouter,
    BatchingEngine,
    FatiguePreventor,
    DeduplicationEngine,
    get_alert_system,
    reset_alert_system,
    send_alert,
    get_user_alerts,
    get_alert_stats,
)

# =========
# Public API
# =========
__all__ = [
    # Issue Registry
    'IssueRegistry', 'Issue', 'IssueComment', 'IssueAuditEntry', 'IssueFilter',
    'IssueStatus', 'IssuePriority', 'IssueSeverity', 'IssueCategory',
    'AssigneeRole', 'AuditAction', 'get_issue_registry',

    # Investigation Rooms
    'InvestigationRoomsManager', 'InvestigationRoom', 'RoomParticipant',
    'RoomContext', 'TimelineEvent', 'Evidence', 'DiscussionThread',
    'ThreadMessage', 'RootCauseVote', 'Resolution',
    'RoomStatus', 'RoomType', 'ParticipantRole', 'EvidenceType',
    'EvidenceStrength', 'ThreadType', 'VoteType', 'ResolutionType',
    'get_investigation_rooms_manager',

    # Tagging System
    'TaggingSystem', 'Tag', 'TagType', 'Notification', 'TagNotificationPriority',
    'EntityLink', 'TagParser', 'EntityResolver', 'NotificationManager',
    'LinkGenerator', 'get_tagging_system',
    'parse_tags', 'extract_mentions', 'extract_topics', 'highlight_tags',

    # Escalation Engine
    'EscalationEngine', 'Escalation', 'EscalationRule', 'EscalationImpact',
    'SponsorPackage', 'EscalationFilter', 'EscalationLevel', 'EscalationStatus',
    'EscalationTrigger', 'EscalationNotificationPriority',
    'get_escalation_engine', 'reset_escalation_engine',
    'check_escalations', 'get_active_escalations',
    'escalate_issue', 'get_escalation_stats',

    # Team Workspaces
    'TeamWorkspacesManager', 'Workspace', 'WorkspaceMember', 'WorkspaceGoal',
    'ActivityFeedItem', 'Announcement', 'SharedResource', 'WorkspaceMetrics',
    'WorkspaceType', 'WorkspaceStatus', 'MemberRole',
    'GoalStatus', 'GoalPriority', 'ActivityType', 'ResourceType',
    'get_team_workspaces_manager', 'reset_team_workspaces_manager',

    # Alert System
    'AlertSystem', 'Alert', 'AlertBatch', 'UserAlertPreferences',
    'AlertStats', 'FatigueStatus', 'AlertPriority', 'AlertChannel',
    'AlertStatus', 'AlertCategory', 'BatchType', 'SuppressionReason',
    'PriorityClassifier', 'ChannelRouter', 'BatchingEngine',
    'FatiguePreventor', 'DeduplicationEngine',
    'get_alert_system', 'reset_alert_system',
    'send_alert', 'get_user_alerts', 'get_alert_stats',
]
