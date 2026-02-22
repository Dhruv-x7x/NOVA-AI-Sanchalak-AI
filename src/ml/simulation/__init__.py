"""
a6on-i - Simulation Module

Phase 9: Digital Twin & Simulation
- 9.1 Trial State Model
- 9.2 Scenario Simulator
- 9.3 Resource Optimizer
- 9.4 Timeline Projector
"""

from .trial_state_model import (
    TrialStateModel,
    TrialStateSnapshot,
    PatientEntity,
    SiteEntity,
    StudyEntity,
    IssueEntity,
    EntityType,
    PatientStatus,
    SiteStatus,
    IssueStatus,
    get_trial_state_model,
    reset_trial_state_model,
    load_trial_state,
    get_current_state,
    check_trial_constraints,
)

from .scenario_simulator import (
    ScenarioSimulator,
    MonteCarloEngine,
    ScenarioResult,
    UncertaintyBand,
    ScenarioOutcome,
    ComparisonResult,
    ResourceType as ScenarioResourceType,
    OutcomeMetric,
    get_scenario_simulator,
    reset_scenario_simulator,
    simulate_close_site,
    simulate_add_resource,
    simulate_deadline,
    simulate_process_change,
)

from .resource_optimizer import (
    ResourceOptimizer,
    Resource,
    Task,
    Constraint,
    Assignment,
    AllocationPlan,
    ImpactProjection,
    Recommendation,
    ResourceType,
    TaskType,
    OptimizationObjective,
    AllocationStatus,
    ConstraintType,
    get_resource_optimizer,
    reset_resource_optimizer,
    optimize_allocation,
    get_recommendations,
    get_resource_stats,
)

from .timeline_projector import (
    TimelineProjector,
    Milestone,
    Trajectory,
    TrajectoryPoint,
    CriticalPath,
    CriticalPathNode,
    TimelineProjection,
    MilestoneType,
    MilestoneStatus,
    TrajectoryType,
    RiskLevel,
    get_timeline_projector,
    reset_timeline_projector,
    project_timeline,
    get_db_lock_projection,
    get_critical_path,
)

from .digital_twin_service import (
    DigitalTwinService,
    get_digital_twin_service,
    reset_digital_twin_service,
)

__all__ = [
    # Trial State Model (9.1)
    'TrialStateModel',
    'TrialStateSnapshot',
    'PatientEntity',
    'SiteEntity',
    'StudyEntity',
    'IssueEntity',
    'EntityType',
    'PatientStatus',
    'SiteStatus',
    'IssueStatus',
    'get_trial_state_model',
    'reset_trial_state_model',
    'load_trial_state',
    'get_current_state',
    'check_trial_constraints',
    
    # Scenario Simulator (9.2)
    'ScenarioSimulator',
    'MonteCarloEngine',
    'ScenarioResult',
    'UncertaintyBand',
    'ScenarioOutcome',
    'ComparisonResult',
    'ScenarioResourceType',
    'OutcomeMetric',
    'get_scenario_simulator',
    'reset_scenario_simulator',
    'simulate_close_site',
    'simulate_add_resource',
    'simulate_deadline',
    'simulate_process_change',
    
    # Resource Optimizer (9.3)
    'ResourceOptimizer',
    'Resource',
    'Task',
    'Constraint',
    'Assignment',
    'AllocationPlan',
    'ImpactProjection',
    'Recommendation',
    'ResourceType',
    'TaskType',
    'OptimizationObjective',
    'AllocationStatus',
    'ConstraintType',
    'get_resource_optimizer',
    'reset_resource_optimizer',
    'optimize_allocation',
    'get_recommendations',
    'get_resource_stats',
    
    # Timeline Projector (9.4)
    'TimelineProjector',
    'Milestone',
    'Trajectory',
    'TrajectoryPoint',
    'CriticalPath',
    'CriticalPathNode',
    'TimelineProjection',
    'MilestoneType',
    'MilestoneStatus',
    'TrajectoryType',
    'RiskLevel',
    'get_timeline_projector',
    'reset_timeline_projector',
    'project_timeline',
    'get_db_lock_projection',
    'get_critical_path',
    
    # Digital Twin Service (unified interface)
    'DigitalTwinService',
    'get_digital_twin_service',
    'reset_digital_twin_service',
]
