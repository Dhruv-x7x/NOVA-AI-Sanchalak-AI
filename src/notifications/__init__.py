"""
a6on-i - Notifications Module
========================================
Centralized notification queue system for all platform notifications.
"""

from .notification_service import (
    NotificationService,
    NotificationQueue,
    Notification,
    NotificationType,
    NotificationPriority,
    NotificationChannel,
    NotificationStatus,
    get_notification_service,
    reset_notification_service,
    send_notification,
    get_user_notifications,
    mark_as_read,
)

__all__ = [
    'NotificationService',
    'NotificationQueue',
    'Notification',
    'NotificationType',
    'NotificationPriority',
    'NotificationChannel',
    'NotificationStatus',
    'get_notification_service',
    'reset_notification_service',
    'send_notification',
    'get_user_notifications',
    'mark_as_read',
]
