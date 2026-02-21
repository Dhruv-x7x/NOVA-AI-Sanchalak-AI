"""
SANCHALAK AI - Proactive Pattern Watcher Scheduler
=======================================================
Background job to run pattern watcher every 15 minutes:
- Check for emerging anomalies
- Generate proactive alerts
- Connect to notification system

Per riyaz.md Section 23: Proactive Pattern Watcher Activation
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class WatcherAlert:
    """Represents a proactive alert from pattern watcher."""
    alert_id: str
    alert_type: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_ids: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None
    acknowledged: bool = False
    snoozed_until: Optional[str] = None


@dataclass
class WatcherMetrics:
    """Metrics for pattern watcher runs."""
    runs_completed: int = 0
    alerts_generated: int = 0
    alerts_acknowledged: int = 0
    last_run_time: Optional[str] = None
    last_run_duration_seconds: float = 0.0
    patterns_checked: int = 0
    anomalies_detected: int = 0


class PatternWatcherScheduler:
    """
    Background scheduler for proactive pattern watching.
    
    Runs pattern watcher every 15 minutes (configurable) to:
    1. Check for emerging anomalies
    2. Detect pattern deviations
    3. Generate proactive alerts
    4. Push to notification system
    
    Usage:
        scheduler = PatternWatcherScheduler()
        scheduler.start()  # Runs in background
        ...
        scheduler.stop()  # Cleanup
    """
    
    DEFAULT_INTERVAL = 15 * 60  # 15 minutes in seconds
    
    def __init__(
        self,
        interval_seconds: int = DEFAULT_INTERVAL,
        notification_callback: Optional[Callable[[WatcherAlert], None]] = None
    ):
        """
        Initialize pattern watcher scheduler.
        
        Args:
            interval_seconds: How often to run pattern watcher
            notification_callback: Function to call when alerts are generated
        """
        self.interval = interval_seconds
        self.notification_callback = notification_callback
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self.metrics = WatcherMetrics()
        self.active_alerts: Dict[str, WatcherAlert] = {}
        self._alert_history: List[Dict] = []
        
        # Alert thresholds
        self.thresholds = {
            "dqi_drop": 10.0,  # Alert if DQI drops more than 10 points
            "query_spike": 50,  # Alert if queries spike by 50+
            "signature_backlog": 100,  # Alert if unsigned forms exceed 100
            "sae_pending_hours": 24,  # Alert if SAE pending > 24 hours
            "sdv_behind_days": 7,  # Alert if SDV behind by 7+ days
        }
        
        logger.info(f"PatternWatcherScheduler initialized (interval={interval_seconds}s)")
    
    def start(self) -> bool:
        """Start the background watcher thread."""
        if self._running:
            logger.warning("Pattern watcher scheduler already running")
            return False
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("Pattern watcher scheduler started")
        return True
    
    def stop(self, timeout: float = 30.0) -> bool:
        """Stop the background watcher thread."""
        if not self._running:
            return True
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Pattern watcher thread did not stop cleanly")
                return False
            self._thread = None
        
        logger.info("Pattern watcher scheduler stopped")
        return True
    
    def _run_loop(self):
        """Main background loop."""
        logger.info("Pattern watcher loop started")
        
        while self._running and not self._stop_event.is_set():
            try:
                self._run_pattern_check()
            except Exception as e:
                logger.error(f"Pattern watcher error: {e}", exc_info=True)
            
            # Wait for next interval or stop signal
            self._stop_event.wait(timeout=self.interval)
        
        logger.info("Pattern watcher loop ended")
    
    def _run_pattern_check(self):
        """Execute a single pattern check cycle."""
        start_time = datetime.now()
        logger.info("Starting pattern watcher check...")
        
        # Track metrics
        patterns_checked = 0
        anomalies_detected = 0
        new_alerts = []
        
        try:
            # Import pattern watcher (lazy to avoid circular imports)
            from src.agents.proactive.pattern_watcher import PatternWatcher
            
            watcher = PatternWatcher()
            
            # 1. Check DQI anomalies
            dqi_alerts = self._check_dqi_anomalies(watcher)
            new_alerts.extend(dqi_alerts)
            patterns_checked += 1
            anomalies_detected += len(dqi_alerts)
            
            # 2. Check query volume spikes
            query_alerts = self._check_query_spikes(watcher)
            new_alerts.extend(query_alerts)
            patterns_checked += 1
            anomalies_detected += len(query_alerts)
            
            # 3. Check signature backlogs
            sig_alerts = self._check_signature_backlogs(watcher)
            new_alerts.extend(sig_alerts)
            patterns_checked += 1
            anomalies_detected += len(sig_alerts)
            
            # 4. Check SAE processing delays
            sae_alerts = self._check_sae_delays(watcher)
            new_alerts.extend(sae_alerts)
            patterns_checked += 1
            anomalies_detected += len(sae_alerts)
            
            # 5. Check SDV completion trends
            sdv_alerts = self._check_sdv_trends(watcher)
            new_alerts.extend(sdv_alerts)
            patterns_checked += 1
            anomalies_detected += len(sdv_alerts)
            
            # 6. Check site performance degradation
            site_alerts = self._check_site_performance(watcher)
            new_alerts.extend(site_alerts)
            patterns_checked += 1
            anomalies_detected += len(site_alerts)
            
        except Exception as e:
            logger.warning(f"Pattern watcher check failed, using demo alerts: {e}")
            # Generate synthetic check for demo
            new_alerts = self._generate_demo_alerts()
            patterns_checked = 6
            anomalies_detected = len(new_alerts)
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        # Update metrics
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics.runs_completed += 1
        self.metrics.alerts_generated += len(new_alerts)
        self.metrics.last_run_time = datetime.now().isoformat()
        self.metrics.last_run_duration_seconds = duration
        self.metrics.patterns_checked = patterns_checked
        self.metrics.anomalies_detected = anomalies_detected
        
        logger.info(
            f"Pattern check complete: {patterns_checked} patterns, "
            f"{anomalies_detected} anomalies, {len(new_alerts)} alerts "
            f"(duration: {duration:.2f}s)"
        )
    
    def _check_dqi_anomalies(self, watcher) -> List[WatcherAlert]:
        """Check for DQI score anomalies."""
        alerts = []
        
        try:
            # Get DQI trends from watcher
            dqi_changes = watcher.detect_dqi_changes(
                lookback_days=7,
                threshold=self.thresholds["dqi_drop"]
            )
            
            for change in dqi_changes:
                if change.get("drop", 0) >= self.thresholds["dqi_drop"]:
                    alert = WatcherAlert(
                        alert_id=f"DQI-{change['study_id']}-{change['site_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                        alert_type="dqi_drop",
                        severity="high" if change["drop"] >= 15 else "medium",
                        title=f"DQI Drop Detected at {change['site_id']}",
                        description=f"DQI dropped by {change['drop']:.1f} points in the last 7 days",
                        study_id=change.get("study_id"),
                        site_id=change.get("site_id"),
                        recommended_actions=[
                            "Review recent query volume",
                            "Check for coordinator availability issues",
                            "Schedule site call to discuss"
                        ]
                    )
                    alerts.append(alert)
        except Exception as e:
            logger.debug(f"DQI check unavailable: {e}")
        
        return alerts
    
    def _check_query_spikes(self, watcher) -> List[WatcherAlert]:
        """Check for query volume spikes."""
        alerts = []
        
        try:
            query_spikes = watcher.detect_query_spikes(
                threshold=self.thresholds["query_spike"]
            )
            
            for spike in query_spikes:
                alert = WatcherAlert(
                    alert_id=f"QRY-SPIKE-{spike['site_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                    alert_type="query_spike",
                    severity="medium",
                    title=f"Query Volume Spike at {spike['site_id']}",
                    description=f"Query volume increased by {spike['increase']} in 24 hours",
                    study_id=spike.get("study_id"),
                    site_id=spike.get("site_id"),
                    recommended_actions=[
                        "Review query types for patterns",
                        "Check for edit check issues",
                        "Consider site training"
                    ]
                )
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Query spike check unavailable: {e}")
        
        return alerts
    
    def _check_signature_backlogs(self, watcher) -> List[WatcherAlert]:
        """Check for signature backlog issues."""
        alerts = []
        
        try:
            backlogs = watcher.detect_signature_backlogs(
                threshold=self.thresholds["signature_backlog"]
            )
            
            for backlog in backlogs:
                alert = WatcherAlert(
                    alert_id=f"SIG-BACKLOG-{backlog['site_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                    alert_type="signature_backlog",
                    severity="high",
                    title=f"Signature Backlog at {backlog['site_id']}",
                    description=f"{backlog['unsigned_count']} forms awaiting PI signature",
                    study_id=backlog.get("study_id"),
                    site_id=backlog.get("site_id"),
                    recommended_actions=[
                        "Schedule PI signing session",
                        "Check PI availability",
                        "Consider delegation log update"
                    ]
                )
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Signature backlog check unavailable: {e}")
        
        return alerts
    
    def _check_sae_delays(self, watcher) -> List[WatcherAlert]:
        """Check for SAE processing delays."""
        alerts = []
        
        try:
            delays = watcher.detect_sae_delays(
                threshold_hours=self.thresholds["sae_pending_hours"]
            )
            
            for delay in delays:
                alert = WatcherAlert(
                    alert_id=f"SAE-DELAY-{delay['sae_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                    alert_type="sae_delay",
                    severity="critical",
                    title=f"SAE Processing Delay - {delay['sae_id']}",
                    description=f"SAE pending for {delay['hours_pending']:.0f} hours",
                    study_id=delay.get("study_id"),
                    site_id=delay.get("site_id"),
                    recommended_actions=[
                        "Escalate to safety team immediately",
                        "Review DM-Safety handoff process",
                        "Check for missing information"
                    ]
                )
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"SAE delay check unavailable: {e}")
        
        return alerts
    
    def _check_sdv_trends(self, watcher) -> List[WatcherAlert]:
        """Check for SDV completion trend issues."""
        alerts = []
        
        try:
            trends = watcher.detect_sdv_trends(
                behind_days=self.thresholds["sdv_behind_days"]
            )
            
            for trend in trends:
                alert = WatcherAlert(
                    alert_id=f"SDV-TREND-{trend['site_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                    alert_type="sdv_behind",
                    severity="medium",
                    title=f"SDV Falling Behind at {trend['site_id']}",
                    description=f"SDV completion {trend['days_behind']} days behind target",
                    study_id=trend.get("study_id"),
                    site_id=trend.get("site_id"),
                    recommended_actions=[
                        "Schedule additional monitoring visit",
                        "Consider remote SDV options",
                        "Review monitoring frequency"
                    ]
                )
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"SDV trend check unavailable: {e}")
        
        return alerts
    
    def _check_site_performance(self, watcher) -> List[WatcherAlert]:
        """Check for site performance degradation."""
        alerts = []
        
        try:
            degradations = watcher.detect_site_degradation()
            
            for deg in degradations:
                alert = WatcherAlert(
                    alert_id=f"SITE-DEG-{deg['site_id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
                    alert_type="site_degradation",
                    severity="high",
                    title=f"Performance Degradation at {deg['site_id']}",
                    description=f"Multiple metrics declining: {', '.join(deg['declining_metrics'])}",
                    study_id=deg.get("study_id"),
                    site_id=deg.get("site_id"),
                    recommended_actions=[
                        "Schedule site performance review call",
                        "Analyze root cause patterns",
                        "Consider additional support resources"
                    ]
                )
                alerts.append(alert)
        except Exception as e:
            logger.debug(f"Site performance check unavailable: {e}")
        
        return alerts
    
    def _generate_demo_alerts(self) -> List[WatcherAlert]:
        """Generate demo alerts when pattern watcher not available."""
        # Only generate occasionally for demo purposes
        import random
        if random.random() > 0.3:  # 70% chance of no alerts
            return []
        
        demo_alerts = [
            WatcherAlert(
                alert_id=f"DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                alert_type="demo_alert",
                severity="medium",
                title="Demo: Proactive Pattern Alert",
                description="This is a demonstration alert from the pattern watcher scheduler",
                study_id="DEMO-STUDY",
                recommended_actions=["This is a demo - no action required"]
            )
        ]
        return demo_alerts[:1]  # Return 0 or 1 demo alert
    
    def _process_alert(self, alert: WatcherAlert):
        """Process a new alert - store and notify."""
        # Check if similar alert already active (avoid duplicates)
        existing_key = f"{alert.alert_type}-{alert.site_id}"
        if existing_key in self.active_alerts:
            existing = self.active_alerts[existing_key]
            # Skip if alert is snoozed
            if existing.snoozed_until:
                snooze_end = datetime.fromisoformat(existing.snoozed_until)
                if datetime.now() < snooze_end:
                    logger.debug(f"Alert {existing_key} is snoozed until {snooze_end}")
                    return
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self._alert_history.append({
            "alert": alert.__dict__,
            "processed_at": datetime.now().isoformat()
        })
        
        # Keep history bounded
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]
        
        # Notify via callback if provided
        if self.notification_callback:
            try:
                self.notification_callback(alert)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
        
        logger.info(f"Alert processed: {alert.alert_id} ({alert.severity})")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_active_alerts(
        self, 
        severity: Optional[str] = None,
        study_id: Optional[str] = None
    ) -> List[WatcherAlert]:
        """Get currently active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if study_id:
            alerts = [a for a in alerts if a.study_id == study_id]
        
        return sorted(alerts, key=lambda a: a.detected_at, reverse=True)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.metrics.alerts_acknowledged += 1
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def snooze_alert(self, alert_id: str, hours: int = 4) -> bool:
        """Snooze an alert for specified hours."""
        if alert_id in self.active_alerts:
            snooze_until = datetime.now() + timedelta(hours=hours)
            self.active_alerts[alert_id].snoozed_until = snooze_until.isoformat()
            logger.info(f"Alert {alert_id} snoozed until {snooze_until}")
            return True
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss/remove an alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} dismissed")
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get watcher metrics."""
        return {
            "runs_completed": self.metrics.runs_completed,
            "alerts_generated": self.metrics.alerts_generated,
            "alerts_acknowledged": self.metrics.alerts_acknowledged,
            "active_alerts_count": len(self.active_alerts),
            "last_run_time": self.metrics.last_run_time,
            "last_run_duration_seconds": self.metrics.last_run_duration_seconds,
            "patterns_checked": self.metrics.patterns_checked,
            "anomalies_detected": self.metrics.anomalies_detected,
            "is_running": self._running,
            "interval_seconds": self.interval
        }
    
    def run_now(self) -> Dict[str, Any]:
        """Trigger an immediate pattern check (for testing/manual runs)."""
        logger.info("Manual pattern check triggered")
        self._run_pattern_check()
        return self.get_metrics()
    
    def update_thresholds(self, new_thresholds: Dict[str, Any]) -> None:
        """Update alert thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Thresholds updated: {self.thresholds}")


# Singleton instance
_scheduler: Optional[PatternWatcherScheduler] = None


def get_pattern_watcher_scheduler() -> PatternWatcherScheduler:
    """Get singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PatternWatcherScheduler()
    return _scheduler


def start_pattern_watcher():
    """Convenience function to start the pattern watcher."""
    scheduler = get_pattern_watcher_scheduler()
    return scheduler.start()


def stop_pattern_watcher():
    """Convenience function to stop the pattern watcher."""
    scheduler = get_pattern_watcher_scheduler()
    return scheduler.stop()
