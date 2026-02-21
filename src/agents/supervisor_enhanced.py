#!/usr/bin/env python3
"""
SANCHALAK AI - Enhanced SUPERVISOR Agent v2.1

Features:
1. Advanced task decomposition (complex → subtasks)
2. Intelligent agent routing (intent classification + confidence)
3. Conflict resolution between agents
4. Output synthesis and deduplication
5. Parallel agent execution support
6. Confidence-based routing with fallbacks

FIXES in v2.1:
- Lowered COMPOUND threshold from >0.5 to >=0.4
- decompose_task() now creates subtasks for ALL agents in sequence
- Improved confidence scoring
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classified query intents for routing"""
    DIAGNOSTIC = "diagnostic"
    FORECAST = "forecast"
    RESOLUTION = "resolution"
    COMMUNICATION = "communication"
    STATUS = "status"
    COMPARISON = "comparison"
    SAFETY = "safety"
    COMPOUND = "compound"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass
class Subtask:
    """Represents a decomposed subtask"""
    id: str
    description: str
    assigned_agent: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


@dataclass
class RoutingDecision:
    """Represents a routing decision with reasoning"""
    primary_intent: QueryIntent
    secondary_intents: List[QueryIntent]
    confidence: float
    agent_sequence: List[str]
    reasoning: str
    complexity: TaskComplexity
    requires_safety_check: bool = False
    requires_human_approval: bool = False


@dataclass
class ConflictResolution:
    """Resolution of conflicts between agent outputs"""
    conflict_type: str
    conflicting_agents: List[str]
    resolution_strategy: str
    resolved_output: Dict[str, Any]
    confidence: float


class EnhancedSupervisorAgent:
    """
    Enhanced SUPERVISOR Agent v2.1 with advanced orchestration capabilities.
    
    Responsibilities:
    1. Analyze and classify incoming queries
    2. Decompose complex queries into subtasks
    3. Route to appropriate specialist agents
    4. Resolve conflicts between agent outputs
    5. Synthesize final response
    """
    
    INTENT_PATTERNS = {
        QueryIntent.DIAGNOSTIC: {
            'keywords': ['why', 'cause', 'reason', 'investigate', 'root cause',
                        'explain', 'understand', 'happening', 'issue', 'problem'],
            'patterns': [
                r'why\s+(is|are|does|do|has|have)',
                r'what\s+(is|are)\s+(causing|the\s+cause)',
                r'root\s+cause',
                r'investigate\s+',
            ],
            'weight': 1.0
        },
        QueryIntent.FORECAST: {
            'keywords': ['when', 'timeline', 'predict', 'forecast', 'project',
                        'estimate', 'ready', 'complete', 'finish', 'deadline'],
            'patterns': [
                r'when\s+will',
                r'how\s+long',
                r'by\s+when',
                r'timeline\s+for',
                r'predict\s+',
            ],
            'weight': 1.0
        },
        QueryIntent.RESOLUTION: {
            'keywords': ['how', 'fix', 'resolve', 'solve', 'action', 'recommend',
                        'steps', 'plan', 'do', 'improve', 'address'],
            'patterns': [
                r'how\s+(to|do|can|should)',
                r'what\s+(should|can)\s+(we|i)\s+do',
                r'recommend\s+',
                r'action\s+plan',
            ],
            'weight': 1.0
        },
        QueryIntent.COMMUNICATION: {
            'keywords': ['email', 'notify', 'message', 'draft', 'write', 'send',
                        'communicate', 'inform', 'escalate', 'alert'],
            'patterns': [
                r'(draft|write|send)\s+(an?\s+)?(email|message)',
                r'notify\s+',
                r'communicate\s+',
                r'escalat',
            ],
            'weight': 1.0
        },
        QueryIntent.STATUS: {
            'keywords': ['what', 'current', 'status', 'summary', 'overview',
                        'show', 'list', 'display', 'report', 'dashboard'],
            'patterns': [
                r'what\s+(is|are)\s+the\s+(current|status)',
                r'show\s+(me\s+)?the',
                r'give\s+(me\s+)?(a\s+)?summary',
            ],
            'weight': 0.8
        },
        QueryIntent.COMPARISON: {
            'keywords': ['compare', 'benchmark', 'versus', 'vs', 'difference',
                        'better', 'worse', 'ranking', 'performance'],
            'patterns': [
                r'compare\s+',
                r'vs\.?\s+',
                r'benchmark\s+',
                r'how\s+does\s+.+\s+compare',
            ],
            'weight': 1.0
        },
        QueryIntent.SAFETY: {
            'keywords': ['sae', 'adverse', 'safety', 'urgent', 'critical',
                        'serious', 'life-threatening', 'death', 'hospitalization'],
            'patterns': [
                r'sae\s+',
                r'adverse\s+event',
                r'safety\s+(issue|concern|signal)',
                r'urgent\s+',
            ],
            'weight': 1.5
        },
    }
    
    AGENT_CAPABILITIES = {
        'diagnostic': {
            'intents': [QueryIntent.DIAGNOSTIC],
            'tools': ['get_patient', 'get_site_summary', 'search_patterns',
                     'get_cascade_impact', 'search_knowledge'],
            'can_handle': ['root cause analysis', 'issue investigation',
                          'pattern detection', 'hypothesis generation'],
        },
        'forecaster': {
            'intents': [QueryIntent.FORECAST],
            'tools': ['get_dblock_projection', 'get_site_summary',
                     'get_study_summary', 'predict_risk'],
            'can_handle': ['timeline prediction', 'readiness projection',
                          'trend analysis', 'what-if scenarios'],
        },
        'resolver': {
            'intents': [QueryIntent.RESOLUTION],
            'tools': ['search_resolutions', 'get_cascade_impact',
                     'create_task', 'get_site_benchmark'],
            'can_handle': ['action planning', 'resolution recommendations',
                          'priority setting', 'effort estimation'],
        },
        'communicator': {
            'intents': [QueryIntent.COMMUNICATION],
            'tools': ['draft_query_email', 'search_knowledge'],
            'can_handle': ['email drafting', 'notification creation',
                          'escalation messages', 'status updates'],
        },
        'synthesizer': {
            'intents': [QueryIntent.STATUS, QueryIntent.COMPARISON],
            'tools': ['get_overall_summary', 'get_site_benchmark'],
            'can_handle': ['output combination', 'summary generation',
                          'comparison analysis', 'report compilation'],
        },
    }
    
    CONFLICT_STRATEGIES = {
        'confidence_based': "Use output with highest confidence score",
        'recency_based': "Use most recent data/analysis",
        'consensus': "Combine overlapping recommendations",
        'escalate': "Flag for human review when agents disagree",
        'safety_first': "Prioritize safety-related recommendations",
    }
    
    AGENT_SUBTASK_TEMPLATES = {
        'diagnostic': {
            'description': "Analyze root causes and form hypotheses",
            'priority': 2
        },
        'forecaster': {
            'description': "Generate timeline projections and predictions",
            'priority': 2
        },
        'resolver': {
            'description': "Generate action recommendations and resolution plan",
            'priority': 3
        },
        'communicator': {
            'description': "Draft communication based on analysis",
            'priority': 4
        },
        'synthesizer': {
            'description': "Synthesize all findings into final response",
            'priority': 5
        }
    }
    
    def __init__(self, llm_wrapper=None):
        """Initialize Enhanced Supervisor Agent"""
        self.llm = llm_wrapper
        self.name = "supervisor"
        self.task_history: List[Dict] = []
    
    def analyze_query(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Analyze query to determine intent, complexity, and routing."""
        query_lower = query.lower().strip()
        
        intent_scores = self._detect_intents(query_lower)
        
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0] if sorted_intents else QueryIntent.STATUS
        primary_confidence = sorted_intents[0][1] if sorted_intents else 0.5
        
        secondary_intents = [
            intent for intent, score in sorted_intents[1:]
            if score >= 0.3
        ]
        
        high_score_intents = [i for i, s in sorted_intents if s >= 0.4]
        if len(high_score_intents) > 1:
            primary_intent = QueryIntent.COMPOUND
            secondary_intents = high_score_intents
        
        complexity = self._assess_complexity(query_lower, primary_intent, secondary_intents)
        
        agent_sequence = self._build_agent_sequence(
            primary_intent, secondary_intents, complexity
        )
        
        requires_safety = self._check_safety_requirements(query_lower, intent_scores)
        
        requires_approval = self._check_approval_requirements(
            query_lower, complexity, requires_safety
        )
        
        reasoning = self._generate_routing_reasoning(
            query, primary_intent, secondary_intents,
            complexity, agent_sequence
        )
        
        return RoutingDecision(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=primary_confidence,
            agent_sequence=agent_sequence,
            reasoning=reasoning,
            complexity=complexity,
            requires_safety_check=requires_safety,
            requires_human_approval=requires_approval
        )
    
    def _detect_intents(self, query: str) -> Dict[QueryIntent, float]:
        """Detect all intents with confidence scores"""
        scores = {}
        
        for intent, config in self.INTENT_PATTERNS.items():
            score = 0.0
            
            keyword_matches = sum(
                1 for kw in config['keywords']
                if kw in query
            )
            if keyword_matches > 0:
                score += min(keyword_matches * 0.25, 0.6)
            
            pattern_matches = sum(
                1 for pattern in config['patterns']
                if re.search(pattern, query)
            )
            if pattern_matches > 0:
                score += min(pattern_matches * 0.35, 0.6)
            
            score *= config['weight']
            
            if score > 0:
                scores[intent] = min(score, 1.0)
        
        return scores
    
    def _assess_complexity(
        self,
        query: str,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent]
    ) -> TaskComplexity:
        """Assess task complexity based on query analysis"""
        
        if primary_intent == QueryIntent.SAFETY:
            return TaskComplexity.CRITICAL
        
        if primary_intent == QueryIntent.COMPOUND or len(secondary_intents) >= 2:
            return TaskComplexity.COMPLEX
        
        if len(secondary_intents) == 1:
            return TaskComplexity.MODERATE
        
        entity_patterns = [
            r'site[s]?\s*[\d\w_,\s]+',
            r'study\s*[\d\w_,\s]+',
            r'patient[s]?\s*[\d\w_,\s]+',
        ]
        entity_count = sum(1 for p in entity_patterns if re.search(p, query))
        if entity_count >= 2:
            return TaskComplexity.MODERATE
        
        if len(query) > 100 and query.count('.') >= 2:
            return TaskComplexity.MODERATE
        
        return TaskComplexity.SIMPLE
    
    def _build_agent_sequence(
        self,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent],
        complexity: TaskComplexity
    ) -> List[str]:
        """Build optimal agent sequence for the query"""
        
        sequence = ['supervisor']
        
        intent_to_agent = {
            QueryIntent.DIAGNOSTIC: 'diagnostic',
            QueryIntent.FORECAST: 'forecaster',
            QueryIntent.RESOLUTION: 'resolver',
            QueryIntent.COMMUNICATION: 'communicator',
            QueryIntent.STATUS: 'synthesizer',
            QueryIntent.COMPARISON: 'synthesizer',
            QueryIntent.SAFETY: 'diagnostic',
        }
        
        if primary_intent == QueryIntent.COMPOUND:
            for intent in secondary_intents:
                agent = intent_to_agent.get(intent)
                if agent and agent not in sequence:
                    sequence.append(agent)
        else:
            primary_agent = intent_to_agent.get(primary_intent, 'diagnostic')
            if primary_agent not in sequence:
                sequence.append(primary_agent)
        
        if complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            for intent in secondary_intents:
                agent = intent_to_agent.get(intent)
                if agent and agent not in sequence:
                    sequence.append(agent)
        
        if primary_intent == QueryIntent.SAFETY and 'resolver' not in sequence:
            sequence.append('resolver')
        
        if complexity == TaskComplexity.COMPLEX and 'forecaster' not in sequence:
            sequence.append('forecaster')
        
        if 'synthesizer' not in sequence:
            sequence.append('synthesizer')
        
        return sequence
    
    def _check_safety_requirements(
        self,
        query: str,
        intent_scores: Dict[QueryIntent, float]
    ) -> bool:
        """Check if query requires safety protocols"""
        
        if intent_scores.get(QueryIntent.SAFETY, 0) > 0.3:
            return True
        
        safety_keywords = [
            'sae', 'adverse', 'death', 'hospitalization', 'life-threatening',
            'serious', 'urgent', 'critical', 'emergency', 'safety signal'
        ]
        return any(kw in query for kw in safety_keywords)
    
    def _check_approval_requirements(
        self,
        query: str,
        complexity: TaskComplexity,
        requires_safety: bool
    ) -> bool:
        """Determine if human approval is required"""
        
        if requires_safety:
            return True
        
        if complexity == TaskComplexity.CRITICAL:
            return True
        
        approval_keywords = [
            'send', 'submit', 'close', 'lock', 'approve',
            'reject', 'escalate', 'notify sponsor'
        ]
        if any(kw in query for kw in approval_keywords):
            return True
        
        return False
    
    def _generate_routing_reasoning(
        self,
        query: str,
        primary_intent: QueryIntent,
        secondary_intents: List[QueryIntent],
        complexity: TaskComplexity,
        agent_sequence: List[str]
    ) -> str:
        """Generate human-readable routing reasoning"""
        
        reasoning_parts = []
        
        reasoning_parts.append(
            f"Primary intent: {primary_intent.value.upper()} query"
        )
        
        if secondary_intents:
            secondary_str = ', '.join(i.value for i in secondary_intents)
            reasoning_parts.append(f"Secondary intents: {secondary_str}")
        
        reasoning_parts.append(f"Complexity: {complexity.value}")
        
        agent_str = ' → '.join(agent_sequence)
        reasoning_parts.append(f"Agent sequence: {agent_str}")
        
        return " | ".join(reasoning_parts)
    
    def decompose_task(
        self,
        query: str,
        routing: RoutingDecision
    ) -> List[Subtask]:
        """Decompose query into subtasks for ALL agents in sequence"""
        subtasks = []
        task_counter = 0
        
        agents_to_process = [a for a in routing.agent_sequence if a != 'supervisor']
        
        if routing.complexity == TaskComplexity.SIMPLE and len(agents_to_process) <= 2:
            primary_agent = agents_to_process[0] if agents_to_process else 'diagnostic'
            subtasks.append(Subtask(
                id="task_1",
                description=query,
                assigned_agent=primary_agent,
                priority=1
            ))
            return subtasks
        
        task_counter += 1
        data_task = Subtask(
            id=f"task_{task_counter}",
            description=f"Gather relevant data and context for: {query[:80]}...",
            assigned_agent="diagnostic",
            priority=1
        )
        subtasks.append(data_task)
        previous_task_id = data_task.id
        
        for agent in agents_to_process:
            if agent == 'diagnostic' and task_counter == 1:
                task_counter += 1
                template = self.AGENT_SUBTASK_TEMPLATES.get(agent, {})
                analysis_task = Subtask(
                    id=f"task_{task_counter}",
                    description=template.get('description', f"Process with {agent}"),
                    assigned_agent=agent,
                    priority=template.get('priority', 2),
                    dependencies=[previous_task_id]
                )
                subtasks.append(analysis_task)
                previous_task_id = analysis_task.id
                continue
            
            if agent == 'synthesizer':
                continue
            
            task_counter += 1
            template = self.AGENT_SUBTASK_TEMPLATES.get(agent, {})
            
            deps = [previous_task_id]
            
            if agent == 'resolver':
                diagnostic_tasks = [t.id for t in subtasks if t.assigned_agent == 'diagnostic']
                if diagnostic_tasks:
                    deps = [diagnostic_tasks[-1]]
            
            if agent == 'communicator':
                resolver_tasks = [t.id for t in subtasks if t.assigned_agent == 'resolver']
                if resolver_tasks:
                    deps = [resolver_tasks[-1]]
            
            task = Subtask(
                id=f"task_{task_counter}",
                description=template.get('description', f"Process with {agent}"),
                assigned_agent=agent,
                priority=template.get('priority', 3),
                dependencies=deps
            )
            subtasks.append(task)
            previous_task_id = task.id
        
        if 'synthesizer' in agents_to_process:
            task_counter += 1
            prior_task_ids = [t.id for t in subtasks if t.assigned_agent != 'synthesizer'][-3:]
            
            subtasks.append(Subtask(
                id=f"task_{task_counter}",
                description="Synthesize all findings into final response",
                assigned_agent="synthesizer",
                priority=5,
                dependencies=prior_task_ids
            ))
        
        return subtasks
    
    def resolve_conflicts(
        self,
        agent_outputs: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[ConflictResolution]]:
        """Resolve conflicts between agent outputs."""
        conflicts_found = []
        resolved_output = {
            'hypotheses': [],
            'recommendations': [],
            'forecasts': [],
            'communications': [],
            'key_findings': [],
            'confidence': 0.0
        }
        
        all_hypotheses = []
        all_recommendations = []
        all_forecasts = []
        
        for agent_name, output in agent_outputs.items():
            if 'hypotheses' in output:
                for h in output['hypotheses']:
                    h['source_agent'] = agent_name
                    all_hypotheses.append(h)
            if 'recommendations' in output:
                for r in output['recommendations']:
                    r['source_agent'] = agent_name
                    all_recommendations.append(r)
            if 'forecasts' in output:
                for f in output['forecasts']:
                    f['source_agent'] = agent_name
                    all_forecasts.append(f)
        
        if len(all_hypotheses) > 1:
            conflict = self._check_hypothesis_conflicts(all_hypotheses)
            if conflict:
                conflicts_found.append(conflict)
                all_hypotheses = self._apply_conflict_resolution(all_hypotheses, conflict)
        
        if len(all_recommendations) > 1:
            conflict = self._check_recommendation_conflicts(all_recommendations)
            if conflict:
                conflicts_found.append(conflict)
                all_recommendations = self._apply_conflict_resolution(all_recommendations, conflict)
        
        if len(all_forecasts) > 1:
            conflict = self._check_forecast_conflicts(all_forecasts)
            if conflict:
                conflicts_found.append(conflict)
                all_forecasts = self._apply_conflict_resolution(all_forecasts, conflict)
        
        resolved_output['hypotheses'] = self._deduplicate_by_similarity(all_hypotheses, key='description')[:5]
        resolved_output['recommendations'] = self._deduplicate_by_similarity(all_recommendations, key='action')[:10]
        resolved_output['forecasts'] = self._deduplicate_by_similarity(all_forecasts, key='metric')[:5]
        
        confidences = []
        for h in resolved_output['hypotheses']:
            confidences.append(h.get('confidence', 0.5))
        for r in resolved_output['recommendations']:
            confidences.append(r.get('confidence', 0.5))
        
        resolved_output['confidence'] = sum(confidences) / len(confidences) if confidences else 0.5
        
        return resolved_output, conflicts_found
    
    def _check_hypothesis_conflicts(self, hypotheses: List[Dict]) -> Optional[ConflictResolution]:
        if len(hypotheses) < 2:
            return None
        
        confidences = [h.get('confidence', 0.5) for h in hypotheses]
        if max(confidences) - min(confidences) > 0.4:
            agents = list(set(h.get('source_agent', 'unknown') for h in hypotheses))
            return ConflictResolution(
                conflict_type="hypothesis_confidence_mismatch",
                conflicting_agents=agents,
                resolution_strategy="confidence_based",
                resolved_output={'action': 'keep_highest_confidence'},
                confidence=max(confidences)
            )
        return None
    
    def _check_recommendation_conflicts(self, recommendations: List[Dict]) -> Optional[ConflictResolution]:
        if len(recommendations) < 2:
            return None
        
        priorities = [r.get('priority', 'medium') for r in recommendations]
        unique_priorities = set(priorities)
        
        if len(unique_priorities) > 2:
            agents = list(set(r.get('source_agent', 'unknown') for r in recommendations))
            return ConflictResolution(
                conflict_type="priority_disagreement",
                conflicting_agents=agents,
                resolution_strategy="safety_first",
                resolved_output={'action': 'prioritize_safety_and_high'},
                confidence=0.7
            )
        return None
    
    def _check_forecast_conflicts(self, forecasts: List[Dict]) -> Optional[ConflictResolution]:
        if len(forecasts) < 2:
            return None
        
        timelines = [f.get('days', 0) for f in forecasts if 'days' in f]
        
        if len(timelines) >= 2:
            if max(timelines) > min(timelines) * 2:
                agents = list(set(f.get('source_agent', 'unknown') for f in forecasts))
                return ConflictResolution(
                    conflict_type="timeline_disagreement",
                    conflicting_agents=agents,
                    resolution_strategy="consensus",
                    resolved_output={'action': 'use_average_with_range'},
                    confidence=0.6
                )
        return None
    
    def _apply_conflict_resolution(self, items: List[Dict], conflict: ConflictResolution) -> List[Dict]:
        if conflict.resolution_strategy == "confidence_based":
            return sorted(items, key=lambda x: x.get('confidence', 0), reverse=True)
        
        elif conflict.resolution_strategy == "safety_first":
            safety_keywords = ['sae', 'safety', 'adverse', 'urgent', 'critical']
            
            def is_safety_related(item):
                text = str(item).lower()
                return any(kw in text for kw in safety_keywords)
            
            safety_items = [i for i in items if is_safety_related(i)]
            other_items = [i for i in items if not is_safety_related(i)]
            return safety_items + other_items
        
        elif conflict.resolution_strategy == "consensus":
            return items
        
        return items
    
    def _deduplicate_by_similarity(self, items: List[Dict], key: str, threshold: float = 0.8) -> List[Dict]:
        if not items:
            return []
        
        seen_texts = []
        unique_items = []
        
        for item in items:
            text = str(item.get(key, ''))
            
            is_duplicate = False
            for seen in seen_texts:
                words1 = set(text.lower().split())
                words2 = set(seen.lower().split())
                
                if len(words1) == 0 or len(words2) == 0:
                    continue
                
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.append(text)
                unique_items.append(item)
        
        return unique_items
    
    def synthesize_outputs(
        self,
        query: str,
        routing: RoutingDecision,
        agent_outputs: Dict[str, Dict],
        conflicts: List[ConflictResolution]
    ) -> Dict[str, Any]:
        """Synthesize all agent outputs into a coherent final response."""
        
        resolved, new_conflicts = self.resolve_conflicts(agent_outputs)
        conflicts.extend(new_conflicts)
        
        final_response = {
            'query': query,
            'routing': {
                'primary_intent': routing.primary_intent.value,
                'complexity': routing.complexity.value,
                'agent_sequence': routing.agent_sequence,
                'reasoning': routing.reasoning
            },
            'key_findings': [],
            'hypotheses': resolved.get('hypotheses', []),
            'recommendations': resolved.get('recommendations', []),
            'forecasts': resolved.get('forecasts', []),
            'communications': resolved.get('communications', []),
            'confidence': resolved.get('confidence', 0.5),
            'conflicts_resolved': len(conflicts),
            'requires_approval': routing.requires_human_approval,
            'safety_flagged': routing.requires_safety_check,
            'timestamp': datetime.now().isoformat()
        }
        
        key_findings = []
        for agent_name, output in agent_outputs.items():
            if 'summary' in output:
                key_findings.append({'source': agent_name, 'finding': output['summary']})
            if 'key_insight' in output:
                key_findings.append({'source': agent_name, 'finding': output['key_insight']})
        
        final_response['key_findings'] = key_findings[:5]
        final_response['executive_summary'] = self._generate_executive_summary(
            query, routing, resolved, conflicts
        )
        
        return final_response
    
    def _generate_executive_summary(
        self,
        query: str,
        routing: RoutingDecision,
        resolved: Dict,
        conflicts: List[ConflictResolution]
    ) -> str:
        """Generate a concise executive summary"""
        
        parts = []
        parts.append(f"Query analyzed as {routing.complexity.value} {routing.primary_intent.value} request.")
        
        n_hypotheses = len(resolved.get('hypotheses', []))
        n_recommendations = len(resolved.get('recommendations', []))
        n_forecasts = len(resolved.get('forecasts', []))
        
        if n_hypotheses > 0:
            parts.append(f"Identified {n_hypotheses} potential root cause(s).")
        if n_recommendations > 0:
            parts.append(f"Generated {n_recommendations} action recommendation(s).")
        if n_forecasts > 0:
            parts.append(f"Produced {n_forecasts} timeline projection(s).")
        
        confidence = resolved.get('confidence', 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5
            
        conf_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        parts.append(f"Overall confidence: {conf_level} ({confidence:.0%}).")
        
        if conflicts:
            parts.append(f"Resolved {len(conflicts)} conflict(s) between agents.")
        
        if routing.requires_safety_check:
            parts.append("⚠️ SAFETY FLAG: This query involves safety-critical elements.")
        
        if routing.requires_human_approval:
            parts.append("🔒 APPROVAL REQUIRED: Human review needed before execution.")
        
        return " ".join(parts)
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main entry point: Process a query through the enhanced supervisor."""
        logger.info(f"Enhanced Supervisor v2.1 processing: {query[:100]}...")
        
        routing = self.analyze_query(query, context)
        logger.info(f"Routing: {routing.reasoning}")
        
        subtasks = self.decompose_task(query, routing)
        logger.info(f"Decomposed into {len(subtasks)} subtasks")
        
        result = {
            'routing_decision': {
                'primary_intent': routing.primary_intent.value,
                'secondary_intents': [i.value for i in routing.secondary_intents],
                'confidence': routing.confidence,
                'complexity': routing.complexity.value,
                'agent_sequence': routing.agent_sequence,
                'reasoning': routing.reasoning,
                'requires_safety_check': routing.requires_safety_check,
                'requires_human_approval': routing.requires_human_approval
            },
            'subtasks': [
                {
                    'id': st.id,
                    'description': st.description,
                    'assigned_agent': st.assigned_agent,
                    'priority': st.priority,
                    'dependencies': st.dependencies
                }
                for st in subtasks
            ],
            'next_agent': routing.agent_sequence[1] if len(routing.agent_sequence) > 1 else 'synthesizer',
            'message': f"Query classified as {routing.complexity.value} {routing.primary_intent.value}. "
                      f"Routing through: {' → '.join(routing.agent_sequence)}"
        }
        
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query[:200],
            'routing': routing.primary_intent.value,
            'agents': routing.agent_sequence
        })
        
        return result


def get_enhanced_supervisor(llm_wrapper=None) -> EnhancedSupervisorAgent:
    """Get an instance of the Enhanced Supervisor Agent"""
    return EnhancedSupervisorAgent(llm_wrapper)


def test_supervisor():
    """Test the enhanced supervisor with sample queries"""
    supervisor = get_enhanced_supervisor()
    
    test_queries = [
        "Why does Site_101 have so many open queries?",
        "When will we be ready for database lock?",
        "How should we resolve the SDV backlog at JP-205?",
        "Draft an email to notify the site about pending signatures",
        "What is the current status of Study_21?",
        "Compare site performance between ASIA and EU regions",
        "Urgent: We have 3 SAE cases pending - what's the root cause and when will they be resolved?",
        "Why are queries increasing, when will they be resolved, and draft an escalation email",
    ]
    
    print("=" * 70)
    print("ENHANCED SUPERVISOR AGENT v2.1 - TEST RESULTS")
    print("=" * 70)
    
    all_passed = True
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}: {query[:60]}...")
        print("─" * 70)
        
        result = supervisor.process(query)
        routing = result['routing_decision']
        
        print(f"Primary Intent:  {routing['primary_intent']}")
        print(f"Complexity:      {routing['complexity']}")
        print(f"Confidence:      {routing['confidence']:.0%}")
        print(f"Agent Sequence:  {' → '.join(routing['agent_sequence'])}")
        print(f"Subtasks:        {len(result['subtasks'])}")
        print(f"Safety Check:    {'⚠️ YES' if routing['requires_safety_check'] else 'No'}")
        print(f"Needs Approval:  {'🔒 YES' if routing['requires_human_approval'] else 'No'}")
        
        if result['subtasks']:
            print(f"\nSubtasks:")
            for st in result['subtasks']:
                deps = f" (depends: {st['dependencies']})" if st['dependencies'] else ""
                print(f"  [{st['priority']}] {st['assigned_agent']}: {st['description'][:50]}...{deps}")
        
        if routing['complexity'] in ['complex', 'critical']:
            agents_in_subtasks = set(st['assigned_agent'] for st in result['subtasks'])
            expected_agents = set(a for a in routing['agent_sequence'] if a != 'supervisor')
            missing = expected_agents - agents_in_subtasks
            
            if missing:
                print(f"\n  ⚠️ VALIDATION: Missing subtasks for: {missing}")
                all_passed = False
            else:
                print(f"\n  ✅ VALIDATION: All agents have subtasks")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("⚠️ SOME VALIDATIONS FAILED - Review above")
    print("=" * 70)


if __name__ == "__main__":
    test_supervisor()