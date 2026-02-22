"""
a6on-i - PostgreSQL Collaboration Service
Provides direct PostgreSQL access for collaboration data,
replacing SQLite-based collaboration modules for production use.
"""

from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

class PGCollaborationService:
    """PostgreSQL-based collaboration service for production use."""
    
    def __init__(self):
        from src.database.connection import get_db_manager
        self.db = get_db_manager()
        logger.info("PGCollaborationService initialized")
    
    # =========================================================================
    # INVESTIGATION ROOMS
    # =========================================================================
    
    def list_rooms(self, status: str = None) -> List[Dict]:
        """List all investigation rooms."""
        try:
            with self.db.engine.connect() as conn:
                if status:
                    query = text("""
                        SELECT r.*, 
                            (SELECT COUNT(*) FROM room_participants p WHERE p.room_id = r.room_id) as participant_count,
                            (SELECT COUNT(*) FROM room_evidence e WHERE e.room_id = r.room_id) as evidence_count,
                            (SELECT COUNT(*) FROM room_threads t WHERE t.room_id = r.room_id) as thread_count
                        FROM investigation_rooms r
                        WHERE r.status = :status
                        ORDER BY r.created_at DESC
                    """)
                    result = conn.execute(query, {"status": status})
                else:
                    query = text("""
                        SELECT r.*, 
                            (SELECT COUNT(*) FROM room_participants p WHERE p.room_id = r.room_id) as participant_count,
                            (SELECT COUNT(*) FROM room_evidence e WHERE e.room_id = r.room_id) as evidence_count,
                            (SELECT COUNT(*) FROM room_threads t WHERE t.room_id = r.room_id) as thread_count
                        FROM investigation_rooms r
                        ORDER BY r.created_at DESC
                    """)
                    result = conn.execute(query)
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning(f"Error listing rooms: {e}")
            return []
    
    def create_room(self, title: str, room_type: str, description: str, 
                    created_by: str) -> Dict:
        """Create a new investigation room."""
        room_id = f"ROOM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            with self.db.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO investigation_rooms 
                    (room_id, title, room_type, description, created_by, status, created_at)
                    VALUES (:room_id, :title, :room_type, :description, :created_by, 'active', NOW())
                """), {
                    "room_id": room_id, "title": title, "room_type": room_type,
                    "description": description, "created_by": created_by
                })
            return {"room_id": room_id, "title": title, "status": "active"}
        except Exception as e:
            logger.error(f"Error creating room: {e}")
            return {}
    
    def get_room(self, room_id: str) -> Optional[Dict]:
        """Get a single room by ID."""
        rooms = self.list_rooms()
        for room in rooms:
            if room.get('room_id') == room_id:
                return room
        return None
    
    # =========================================================================
    # TEAM WORKSPACES
    # =========================================================================
    
    def list_workspaces(self, status: str = None) -> List[Dict]:
        """List all team workspaces."""
        try:
            with self.db.engine.connect() as conn:
                if status:
                    query = text("""
                        SELECT w.*, 
                            (SELECT COUNT(*) FROM workspace_members m WHERE m.workspace_id = w.workspace_id) as member_count,
                            (SELECT COUNT(*) FROM workspace_goals g WHERE g.workspace_id = w.workspace_id) as goal_count,
                            (SELECT COUNT(*) FROM workspace_goals g WHERE g.workspace_id = w.workspace_id AND g.status = 'in_progress') as active_goal_count
                        FROM team_workspaces w
                        WHERE w.status = :status
                        ORDER BY w.created_at DESC
                    """)
                    result = conn.execute(query, {"status": status})
                else:
                    query = text("""
                        SELECT w.*, 
                            (SELECT COUNT(*) FROM workspace_members m WHERE m.workspace_id = w.workspace_id) as member_count,
                            (SELECT COUNT(*) FROM workspace_goals g WHERE g.workspace_id = w.workspace_id) as goal_count,
                            (SELECT COUNT(*) FROM workspace_goals g WHERE g.workspace_id = w.workspace_id AND g.status = 'in_progress') as active_goal_count
                        FROM team_workspaces w
                        ORDER BY w.created_at DESC
                    """)
                    result = conn.execute(query)
                rows = [dict(row._mapping) for row in result]
                # Add TQI placeholder (could be calculated from regional data)
                for row in rows:
                    row['tqi'] = 80  # Placeholder
                return rows
        except Exception as e:
            logger.warning(f"Error listing workspaces: {e}")
            return []
    
    def create_workspace(self, name: str, workspace_type: str, 
                         description: str, created_by: str) -> Dict:
        """Create a new team workspace."""
        ws_id = f"WS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            with self.db.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO team_workspaces 
                    (workspace_id, name, workspace_type, description, created_by, status, created_at)
                    VALUES (:ws_id, :name, :ws_type, :description, :created_by, 'active', NOW())
                """), {
                    "ws_id": ws_id, "name": name, "ws_type": workspace_type,
                    "description": description, "created_by": created_by
                })
            return {"workspace_id": ws_id, "name": name, "status": "active"}
        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            return {}
    
    # =========================================================================
    # ESCALATIONS
    # =========================================================================
    
    def list_escalations(self, status: str = None) -> List[Dict]:
        """List all escalations."""
        try:
            with self.db.engine.connect() as conn:
                if status:
                    query = text("""
                        SELECT * FROM escalations
                        WHERE status = :status
                        ORDER BY level DESC, created_at DESC
                    """)
                    result = conn.execute(query, {"status": status})
                else:
                    query = text("""
                        SELECT * FROM escalations
                        ORDER BY level DESC, created_at DESC
                    """)
                    result = conn.execute(query)
                rows = [dict(row._mapping) for row in result]
                # Calculate SLA remaining hours - handle timezone-aware datetimes
                now = datetime.now()
                for row in rows:
                    if row.get('sla_deadline'):
                        sla = row['sla_deadline']
                        # Make both timezone-naive for comparison
                        if hasattr(sla, 'replace') and sla.tzinfo is not None:
                            sla = sla.replace(tzinfo=None)
                        try:
                            delta = sla - now
                            row['sla_remaining_hours'] = max(0, delta.total_seconds() / 3600)
                        except Exception:
                            row['sla_remaining_hours'] = 24
                    else:
                        row['sla_remaining_hours'] = 24
                return rows
        except Exception as e:
            logger.warning(f"Error listing escalations: {e}")
            return []
    
    def search_escalations(self, filter_obj=None) -> tuple:
        """Search escalations with optional filter. Returns (list, total_count)."""
        escalations = self.list_escalations()
        return escalations, len(escalations)
    
    # =========================================================================
    # USER ALERTS
    # =========================================================================
    
    def get_alerts_for_user(self, user_id: str) -> List[Dict]:
        """Get alerts for a specific user. Returns all alerts if user not found."""
        try:
            with self.db.engine.connect() as conn:
                # First try to get alerts for specific user
                query = text("""
                    SELECT * FROM user_alerts
                    WHERE recipient_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                result = conn.execute(query, {"user_id": user_id})
                alerts = [dict(row._mapping) for row in result]
                
                # If no alerts for this user, return all recent alerts
                if not alerts:
                    query = text("""
                        SELECT * FROM user_alerts
                        ORDER BY created_at DESC
                        LIMIT 50
                    """)
                    result = conn.execute(query)
                    alerts = [dict(row._mapping) for row in result]
                
                return alerts
        except Exception as e:
            logger.warning(f"Error getting user alerts: {e}")
            return []
    
    def send_alert(self, recipient_id: str, title: str, message: str,
                   category: str, priority: str) -> Dict:
        """Send an alert to a user."""
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        try:
            with self.db.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO user_alerts 
                    (alert_id, recipient_id, title, message, category, priority, status, created_at)
                    VALUES (:alert_id, :recipient_id, :title, :message, :category, :priority, 'unread', NOW())
                """), {
                    "alert_id": alert_id, "recipient_id": recipient_id, 
                    "title": title, "message": message, "category": category, 
                    "priority": priority
                })
            return {"alert_id": alert_id, "status": "sent"}
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return {}
    
    def mark_alert_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        try:
            with self.db.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE user_alerts SET status = 'read', read_at = NOW()
                    WHERE alert_id = :alert_id
                """), {"alert_id": alert_id})
            return True
        except Exception as e:
            logger.error(f"Error marking alert read: {e}")
            return False
    
    # =========================================================================
    # ISSUES (from PostgreSQL issues table)
    # =========================================================================
    
    def list_issues(self, status: str = None, limit: int = 100) -> List[Dict]:
        """List issues from the issues table, transforming to UI-expected format."""
        try:
            with self.db.engine.connect() as conn:
                if status:
                    query = text("""
                        SELECT issue_id, patient_key, issue_type, issue_subtype, severity, status, 
                               description, form_name, field_name, assigned_to, 
                               created_date, due_date, created_at, updated_at
                        FROM issues
                        WHERE status = :status
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    result = conn.execute(query, {"status": status, "limit": limit})
                else:
                    query = text("""
                        SELECT issue_id, patient_key, issue_type, issue_subtype, severity, status, 
                               description, form_name, field_name, assigned_to, 
                               created_date, due_date, created_at, updated_at
                        FROM issues
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    result = conn.execute(query, {"limit": limit})
                
                issues = []
                for row in result:
                    row_dict = dict(row._mapping)
                    # Transform to UI-expected format
                    issues.append({
                        'issue_id': f"ISS-{row_dict['issue_id']:03d}",
                        'title': f"{row_dict['issue_type'].title()} Issue - {row_dict['issue_subtype'].replace('_', ' ').title()}",
                        'description': row_dict.get('description', ''),
                        'category': row_dict.get('issue_subtype', row_dict.get('issue_type', 'unknown')),
                        'priority': row_dict.get('severity', 'medium'),
                        'severity': row_dict.get('severity', 'medium'),
                        'status': row_dict.get('status', 'open'),
                        'assigned_to': row_dict.get('assigned_to', 'Unassigned'),
                        'site_id': row_dict.get('patient_key', '').split('|')[1] if '|' in str(row_dict.get('patient_key', '')) else 'Unknown',
                        'patient_key': row_dict.get('patient_key', ''),
                        'form_name': row_dict.get('form_name', ''),
                        'field_name': row_dict.get('field_name', ''),
                        'created_at': row_dict.get('created_at') or row_dict.get('created_date'),
                        'due_date': row_dict.get('due_date'),
                        'age_days': 0  # Will be calculated by caller
                    })
                return issues
        except Exception as e:
            logger.warning(f"Error listing issues: {e}")
            return []
    
    def search_issues(self, filter_obj=None, limit: int = 100) -> tuple:
        """Search issues with optional filter. Returns (list, total_count)."""
        issues = self.list_issues(limit=limit)
        return issues, len(issues)


# Singleton instance
_pg_collab_service = None

def get_pg_collaboration_service() -> PGCollaborationService:
    """Get singleton PostgreSQL collaboration service."""
    global _pg_collab_service
    if _pg_collab_service is None:
        _pg_collab_service = PGCollaborationService()
    return _pg_collab_service
