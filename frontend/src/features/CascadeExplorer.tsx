import { useState, useCallback, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { graphApi, studiesApi, intelligenceApi } from '@/services/api';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  GitBranch,
  AlertTriangle,
  RefreshCw,
  Lock,
  CalendarX,
  FileQuestion,
  Search,
  PenTool,
  AlertCircle,
  Shield,
  Tag,
  Pill,
  FlaskConical,
  Link2,
  FileX,
  BarChart3,
  Zap,
  Target,
  TrendingUp,
  Users,
  Clock,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import {
  TooltipProvider,
} from "@/components/ui/tooltip";

// ============================================================
// ISSUE TYPE CONFIGURATION - Matching system configuration
// ============================================================

interface IssueTypeConfig {
  name: string;
  color: string;
  icon: React.ReactNode;
  weight: number;
  responsible: string;
}

const ISSUE_CONFIG: Record<string, IssueTypeConfig> = {
  missing_visits: { name: "Missing Visits", color: "#e74c3c", icon: <CalendarX className="w-4 h-4" />, weight: 100, responsible: "Site Coordinator" },
  missing_pages: { name: "Missing Pages", color: "#c0392b", icon: <FileX className="w-4 h-4" />, weight: 95, responsible: "CRA" },
  open_queries: { name: "Open Queries", color: "#f39c12", icon: <FileQuestion className="w-4 h-4" />, weight: 80, responsible: "Data Manager" },
  sdv_incomplete: { name: "SDV Incomplete", color: "#e67e22", icon: <Search className="w-4 h-4" />, weight: 75, responsible: "CRA" },
  signature_gaps: { name: "Signature Gaps", color: "#9b59b6", icon: <PenTool className="w-4 h-4" />, weight: 70, responsible: "Site / PI" },
  broken_signatures: { name: "Broken Signatures", color: "#8e44ad", icon: <AlertCircle className="w-4 h-4" />, weight: 65, responsible: "Site / PI" },
  sae_dm_pending: { name: "SAE-DM Pending", color: "#e74c3c", icon: <AlertTriangle className="w-4 h-4" />, weight: 100, responsible: "Safety Data Manager" },
  sae_safety_pending: { name: "SAE-Safety Pending", color: "#c0392b", icon: <Shield className="w-4 h-4" />, weight: 95, responsible: "Safety Physician" },
  meddra_uncoded: { name: "MedDRA Uncoded", color: "#3498db", icon: <Tag className="w-4 h-4" />, weight: 50, responsible: "Medical Coder" },
  whodrug_uncoded: { name: "WHODrug Uncoded", color: "#2980b9", icon: <Pill className="w-4 h-4" />, weight: 50, responsible: "Medical Coder" },
  lab_issues: { name: "Lab Issues", color: "#1abc9c", icon: <FlaskConical className="w-4 h-4" />, weight: 60, responsible: "Data Manager" },
  edrr_issues: { name: "EDRR Issues", color: "#16a085", icon: <Link2 className="w-4 h-4" />, weight: 55, responsible: "Data Manager" },
  inactivated_forms: { name: "Inactivated Forms", color: "#95a5a6", icon: <FileX className="w-4 h-4" />, weight: 40, responsible: "Data Manager" },
  high_query_volume: { name: "High Query Volume", color: "#f1c40f", icon: <BarChart3 className="w-4 h-4" />, weight: 45, responsible: "CRA" },
  db_lock: { name: "DB Lock Ready", color: "#27ae60", icon: <Lock className="w-4 h-4" />, weight: 0, responsible: "Study Lead" },
};

// Issue dependencies - what fixing each issue unlocks
const ISSUE_DEPENDENCIES: Record<string, string[]> = {
  missing_visits: ["missing_pages", "open_queries", "sdv_incomplete", "signature_gaps"],
  missing_pages: ["open_queries", "sdv_incomplete", "signature_gaps"],
  open_queries: ["signature_gaps", "db_lock"],
  sdv_incomplete: ["signature_gaps", "db_lock"],
  signature_gaps: ["db_lock"],
  broken_signatures: ["signature_gaps", "db_lock"],
  sae_dm_pending: ["sae_safety_pending", "db_lock"],
  sae_safety_pending: ["db_lock"],
  meddra_uncoded: ["db_lock"],
  whodrug_uncoded: ["db_lock"],
  lab_issues: ["db_lock"],
  edrr_issues: ["db_lock"],
  inactivated_forms: ["db_lock"],
  high_query_volume: ["open_queries"],
};

// Hierarchical layers for layout
const LAYERS: Record<number, string[]> = {
  0: ["missing_visits", "high_query_volume"],
  1: ["missing_pages", "sae_dm_pending"],
  2: ["open_queries", "sdv_incomplete", "sae_safety_pending"],
  3: ["signature_gaps", "broken_signatures", "meddra_uncoded", "whodrug_uncoded"],
  4: ["lab_issues", "edrr_issues", "inactivated_forms"],
  5: ["db_lock"],
};

// ============================================================
// TYPES
// ============================================================

interface CascadeNode {
  id: string;
  name: string;
  patientCount: number;
  unlockScore: number;
  color: string;
  icon: React.ReactNode;
  responsible: string;
  x: number;
  y: number;
  layer: number;
}

interface CascadeEdge {
  source: string;
  target: string;
  weight: number;
}

// Impact level color helper
const getImpactColor = (score: number) => {
  if (score >= 80) return '#e74c3c';
  if (score >= 60) return '#f39c12';
  if (score >= 40) return '#3498db';
  return '#27ae60';
};

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function CascadeExplorer() {
  const [selectedStudy, setSelectedStudy] = useState<string>('all');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('graph');
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Fetch studies for dropdown
  const { data: studiesData } = useQuery({
    queryKey: ['studies'],
    queryFn: () => studiesApi.list(),
  });

  // Fetch cascade analysis data
  const { data: cascadeData, isLoading: cascadeLoading, refetch } = useQuery({
    queryKey: ['cascade-analysis', selectedStudy],
    queryFn: () => graphApi.getCascadeAnalysis({ limit: 500, study_id: selectedStudy !== 'all' ? selectedStudy : undefined }),
  });

  // Fetch cascade issue type counts from the new endpoint
  const { data: cascadeIssueData } = useQuery({
    queryKey: ['cascade-issue-types', selectedStudy],
    queryFn: () => graphApi.getCascadeIssueTypes({ study_id: selectedStudy !== 'all' ? selectedStudy : undefined }),
  });

  // Fetch cascade topology (dependencies) from the backend
  const { data: topologyData } = useQuery({
    queryKey: ['cascade-topology', selectedStudy],
    queryFn: () => graphApi.getCascadeTopology({ study_id: selectedStudy !== 'all' ? selectedStudy : undefined }),
  });

  const studies = studiesData?.studies || [];

  if (cascadeLoading) {
    return <SanchalakLoader size="lg" label="Loading cascade graph..." fullPage />;
  }

  const issueDependencies = topologyData?.topology || ISSUE_DEPENDENCIES;
  const isLiveGraph = topologyData?.source === 'neo4j';

  // Use real issue counts from API or defaults
  const issueCounts = useMemo(() => {
    if (cascadeIssueData?.issue_counts) {
      return cascadeIssueData.issue_counts;
    }

    // Fallback: return zeros — real data comes from the API
    const counts: Record<string, number> = {};
    Object.keys(ISSUE_CONFIG).forEach(key => {
      counts[key] = 0;
    });
    return counts;
  }, [cascadeIssueData]);

  // Calculate unlock scores — measures what % of downstream issues are blocked
  const unlockScores = useMemo(() => {
    const scores: Record<string, number> = {};
    const vals = Object.values(issueCounts) as number[];
    const totalPatients: number = vals.reduce((sum, c) => sum + (c || 0), 0) || 1;

    Object.keys(ISSUE_CONFIG).forEach(key => {
      // Count downstream patients blocked by this issue type
      const directTargets = (issueDependencies[key] || []) as string[];
      let downstreamPatients = 0;
      const visited = new Set<string>();
      const queue = [...directTargets];
      while (queue.length > 0) {
        const current = queue.shift()!;
        if (visited.has(current)) continue;
        visited.add(current);
        downstreamPatients += (issueCounts[current] as number) || 0;
        const nextTargets = (issueDependencies[current] || []) as string[];
        queue.push(...nextTargets);
      }

      // Score = weighted combination of own patients + downstream impact
      const ownPatients: number = (issueCounts[key] as number) || 0;
      const weight: number = ISSUE_CONFIG[key]?.weight || 50;
      const downstreamRatio: number = downstreamPatients / totalPatients;
      const ownRatio: number = ownPatients / totalPatients;

      // Final score: 0-100 range, weighted by config weight, own patients, and downstream impact
      scores[key] = Math.min(100, Math.round(
        (weight * 0.3) + (ownRatio * 100 * 0.3) + (downstreamRatio * 100 * 0.4)
      ));
    });

    return scores;
  }, [issueCounts, issueDependencies]);

  // Generate nodes with positions
  const nodes: CascadeNode[] = useMemo(() => {
    const result: CascadeNode[] = [];

    Object.entries(LAYERS).forEach(([layerStr, issueTypes]) => {
      const layer = parseInt(layerStr);
      const y = 80 + layer * 90; // Vertical spacing

      issueTypes.forEach((issueType, index) => {
        const config = ISSUE_CONFIG[issueType];
        if (!config) return;

        const x = (index + 1) / (issueTypes.length + 1) * 100; // Horizontal spread as percentage

        result.push({
          id: issueType,
          name: config.name,
          patientCount: issueCounts[issueType] || 0,
          unlockScore: unlockScores[issueType] || 0,
          color: config.color,
          icon: config.icon,
          responsible: config.responsible,
          x: x,
          y: y,
          layer,
        });
      });
    });

    return result;
  }, [issueCounts, unlockScores]);

  // Generate edges
  const edges: CascadeEdge[] = useMemo(() => {
    const result: CascadeEdge[] = [];

    Object.entries(issueDependencies).forEach(([source, targets]) => {
      (targets as string[]).forEach(target => {
        result.push({
          source,
          target,
          weight: issueCounts[source] || 0,
        });
      });
    });

    return result;
  }, [issueCounts]);

  // Get highlight path when a node is selected
  const highlightPath = useMemo(() => {
    if (!selectedNode) return new Set<string>();

    const path = new Set<string>([selectedNode]);
    const queue = [...(ISSUE_DEPENDENCIES[selectedNode] || [])];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (!path.has(current)) {
        path.add(current);
        const downstream = (issueDependencies[current] || []) as string[];
        queue.push(...downstream);
      }
    }

    return path;
  }, [selectedNode]);

  // Get cascade impact for selected node
  const selectedNodeImpact = useMemo(() => {
    if (!selectedNode) return null;

    const directUnlocks = (issueDependencies[selectedNode] || []) as string[];
    const cascadeChain = Array.from(highlightPath).filter(n => n !== selectedNode);
    const patientCount = issueCounts[selectedNode] || 0;

    // Calculate metrics
    let patientsUnblocked = patientCount;
    cascadeChain.forEach(node => {
      patientsUnblocked += Math.floor((issueCounts[node] || 0) * 0.3);
    });

    const weight = ISSUE_CONFIG[selectedNode]?.weight || 50;
    const dqiImprovement = (patientCount / 1000) * (weight / 100) * 5;
    const effortHours = patientCount * 0.1;
    const daysSaved = cascadeChain.length * 2 + patientCount / 500;
    const roiScore = (patientsUnblocked * 10 + dqiImprovement * 100) / Math.max(effortHours, 1);

    return {
      directUnlocks,
      cascadeChain,
      patientsUnblocked,
      dqiImprovement,
      effortHours,
      daysSaved,
      roiScore,
      priority: roiScore >= 50 ? 'Critical' : roiScore >= 30 ? 'High' : roiScore >= 15 ? 'Medium' : 'Low',
    };
  }, [selectedNode, highlightPath, issueCounts]);

  // Get position for a node
  const getNodePosition = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    return node ? { x: node.x, y: node.y } : { x: 50, y: 50 };
  }, [nodes]);

  // Action mutation
  const approveFixMutation = useMutation({
    mutationFn: (issueType: string) => intelligenceApi.autoFix({ issue_id: -1, entity_id: issueType }),
    onSuccess: () => {
      alert('Cascade Fix Sequence Initiated! Downstream blockers are being cleared.');
      refetch();
    }
  });

  const handleApproveFix = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedNode) approveFixMutation.mutate(selectedNode);
  };

  // Sort nodes by unlock score for sidebar
  const sortedNodes = useMemo(() => {
    return [...nodes]
      .filter(n => n.patientCount > 0)
      .sort((a, b) => b.unlockScore - a.unlockScore);
  }, [nodes]);

  // Total issues — use the pre-computed total from the API (total_open_issues from UPR)
  const totalIssues: number = cascadeIssueData?.total_issues ?? (Object.values(issueCounts) as number[]).reduce((sum: number, count: number) => sum + count, 0);
  const criticalIssues: number = (issueCounts.sae_dm_pending || 0) + (issueCounts.sae_safety_pending || 0);

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Header */}
        <div className="glass-card rounded-xl p-6 border border-nexus-border" style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-xl bg-white/20 flex items-center justify-center backdrop-blur">
                <GitBranch className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Cascade Explorer</h1>
                <p className="text-white/80">Interactive dependency graph - Fix upstream issues to unlock downstream DB Lock</p>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant={isLiveGraph ? "success" : "secondary"} className="bg-white/10 text-white border-white/20">
                    {isLiveGraph ? <RefreshCw className="w-3 h-3 mr-1 animate-spin-slow" /> : <Lock className="w-3 h-3 mr-1" />}
                    {isLiveGraph ? "Live Neo4j Graph" : "Standard Model"}
                  </Badge>
                  {topologyData?.source && (
                    <span className="text-[10px] text-white/50 uppercase tracking-widest">Source: {topologyData.source}</span>
                  )}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Select value={selectedStudy} onValueChange={setSelectedStudy}>
                <SelectTrigger className="w-44 bg-white/10 border-white/20 text-white backdrop-blur">
                  <SelectValue placeholder="All Studies" />
                </SelectTrigger>
                <SelectContent className="bg-nexus-card border-nexus-border">
                  <SelectItem value="all">All Studies</SelectItem>
                  {studies.map((study: any) => (
                    <SelectItem key={study.study_id} value={study.study_id}>
                      {study.name || study.study_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetch()}
                className="bg-white/10 border-white/20 text-white hover:bg-white/20"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* KPI Summary */}
        <div className="grid gap-4 md:grid-cols-5">
          <Card className="kpi-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">{totalIssues.toLocaleString()}</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Total Issues</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <GitBranch className="w-5 h-5 text-blue-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="kpi-card kpi-card-error">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">{criticalIssues}</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Safety Critical</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-error-500/20 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-error-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="kpi-card kpi-card-warning">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">{Object.keys(ISSUE_CONFIG).length - 1}</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Issue Types</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-warning-500/20 flex items-center justify-center">
                  <Target className="w-5 h-5 text-warning-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="kpi-card kpi-card-purple">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">{edges.length}</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Dependencies</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <Link2 className="w-5 h-5 text-purple-400" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="kpi-card kpi-card-success">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">6</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Path to DB Lock</p>
                </div>
                <div className="w-10 h-10 rounded-lg bg-success-500/20 flex items-center justify-center">
                  <Lock className="w-5 h-5 text-success-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="bg-nexus-card border border-nexus-border">
            <TabsTrigger value="graph" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-indigo-600">
              Dependency Graph
            </TabsTrigger>
            <TabsTrigger value="impact" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-indigo-600">
              Impact Analysis
            </TabsTrigger>
            <TabsTrigger value="critical" className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-indigo-600">
              Critical Path
            </TabsTrigger>
          </TabsList>

          <TabsContent value="graph">
            <div className="grid gap-6 lg:grid-cols-4">
              {/* Main Graph */}
              <div className="lg:col-span-3">
                <Card className="glass-card border-nexus-border overflow-hidden">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-white text-lg">Cascade Dependency Graph</CardTitle>
                        <CardDescription className="text-nexus-text-secondary">
                          Fix upstream issues to unlock downstream - DB Lock
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 rounded-full border border-red-500/30">
                          <CalendarX className="w-3 h-3 text-red-400" />
                          <span className="text-xs text-red-400 font-medium">Root Causes (Start Here)</span>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0">
                    {/* Graph Container */}
                    <div
                      className="relative bg-gradient-to-b from-nexus-bg to-nexus-card"
                      style={{ height: '580px' }}
                    >
                      <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                        <defs>
                          {/* Arrow marker */}
                          <marker
                            id="arrowhead"
                            markerWidth="6"
                            markerHeight="6"
                            refX="5"
                            refY="3"
                            orient="auto"
                          >
                            <polygon points="0 0, 6 3, 0 6" fill="rgba(148, 163, 184, 0.4)" />
                          </marker>
                          <marker
                            id="arrowhead-highlight"
                            markerWidth="6"
                            markerHeight="6"
                            refX="5"
                            refY="3"
                            orient="auto"
                          >
                            <polygon points="0 0, 6 3, 0 6" fill="#00CED1" />
                          </marker>
                          {/* Glow filter */}
                          <filter id="glow">
                            <feGaussianBlur stdDeviation="0.5" result="coloredBlur" />
                            <feMerge>
                              <feMergeNode in="coloredBlur" />
                              <feMergeNode in="SourceGraphic" />
                            </feMerge>
                          </filter>
                        </defs>

                        {/* Edges */}
                        {edges.map((edge, idx) => {
                          const sourcePos = getNodePosition(edge.source);
                          const targetPos = getNodePosition(edge.target);
                          const isHighlighted = highlightPath.has(edge.source) && highlightPath.has(edge.target);

                          // Convert percentage to viewBox coordinates
                          const x1 = sourcePos.x;
                          const y1 = sourcePos.y / 6;
                          const x2 = targetPos.x;
                          const y2 = targetPos.y / 6;

                          return (
                            <line
                              key={`edge-${idx}`}
                              x1={x1}
                              y1={y1}
                              x2={x2}
                              y2={y2}
                              stroke={isHighlighted ? '#00CED1' : 'rgba(148, 163, 184, 0.25)'}
                              strokeWidth={isHighlighted ? 0.4 : 0.15}
                              markerEnd={isHighlighted ? 'url(#arrowhead-highlight)' : 'url(#arrowhead)'}
                            />
                          );
                        })}

                        {/* Nodes */}
                        {nodes.map((node) => {
                          const x = node.x;
                          const y = node.y / 6;
                          const isSelected = selectedNode === node.id;
                          const isHighlighted = highlightPath.has(node.id);
                          const isHovered = hoveredNode === node.id;
                          const isDbLock = node.id === 'db_lock';

                          // Node size based on patient count
                          const baseSize = isDbLock ? 4 : Math.max(2, Math.min(3.5, 2 + node.patientCount / 300));
                          const size = isSelected ? baseSize * 1.3 : isHovered ? baseSize * 1.15 : baseSize;

                          return (
                            <g
                              key={node.id}
                              className="cursor-pointer transition-all duration-200"
                              onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
                              onMouseEnter={() => setHoveredNode(node.id)}
                              onMouseLeave={() => setHoveredNode(null)}
                            >
                              {/* Glow effect for selected/highlighted */}
                              {(isSelected || isHighlighted) && (
                                <circle
                                  cx={x}
                                  cy={y}
                                  r={size + 1}
                                  fill={isSelected ? '#FFD700' : '#00CED1'}
                                  opacity={0.3}
                                />
                              )}

                              {/* Main node */}
                              {isDbLock ? (
                                // Diamond shape for DB Lock
                                <polygon
                                  points={`${x},${y - size} ${x + size},${y} ${x},${y + size} ${x - size},${y}`}
                                  fill={node.color}
                                  stroke="white"
                                  strokeWidth={0.2}
                                  filter="url(#glow)"
                                />
                              ) : (
                                <circle
                                  cx={x}
                                  cy={y}
                                  r={size}
                                  fill={isSelected ? '#FFD700' : isHighlighted ? '#00CED1' : node.color}
                                  stroke="white"
                                  strokeWidth={0.2}
                                  filter="url(#glow)"
                                />
                              )}

                              {/* Label below node */}
                              <text
                                x={x}
                                y={y + size + 2}
                                textAnchor="middle"
                                className="text-[1.2px] fill-white font-medium select-none pointer-events-none"
                                style={{ fontSize: '1.8px' }}
                              >
                                {node.name}
                              </text>
                            </g>
                          );
                        })}
                      </svg>

                      {/* Hover tooltip */}
                      {hoveredNode && !selectedNode && (
                        <div
                          className="absolute bg-nexus-card/95 backdrop-blur-lg border border-nexus-border rounded-lg p-3 shadow-2xl z-50 pointer-events-none"
                          style={{
                            left: `${getNodePosition(hoveredNode).x}%`,
                            top: `${getNodePosition(hoveredNode).y / 6}%`,
                            transform: 'translate(-50%, -120%)',
                            minWidth: '180px',
                          }}
                        >
                          {(() => {
                            const node = nodes.find(n => n.id === hoveredNode);
                            if (!node) return null;
                            const config = ISSUE_CONFIG[node.id];
                            return (
                              <>
                                <div className="flex items-center gap-2 mb-2">
                                  <div style={{ color: node.color }}>{config?.icon}</div>
                                  <span className="font-semibold text-white">{node.name}</span>
                                </div>
                                <div className="space-y-1 text-xs">
                                  <div className="flex justify-between">
                                    <span className="text-nexus-text-secondary">Patients:</span>
                                    <span className="text-white font-medium">{node.patientCount.toLocaleString()}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-nexus-text-secondary">Unlock Score:</span>
                                    <span className="font-medium" style={{ color: getImpactColor(node.unlockScore) }}>
                                      {node.unlockScore.toFixed(1)}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-nexus-text-secondary">Impact:</span>
                                    <Badge
                                      variant={node.unlockScore >= 80 ? 'error' : node.unlockScore >= 60 ? 'warning' : 'secondary'}
                                      className="text-[10px] px-1.5 py-0"
                                    >
                                      {node.unlockScore >= 80 ? 'Critical' : node.unlockScore >= 60 ? 'High' : node.unlockScore >= 40 ? 'Medium' : 'Low'}
                                    </Badge>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-nexus-text-secondary">Responsible:</span>
                                    <span className="text-white">{node.responsible}</span>
                                  </div>
                                </div>
                              </>
                            );
                          })()}
                        </div>
                      )}

                      {/* Legend */}
                      <div className="absolute bottom-4 left-4 bg-nexus-card/90 backdrop-blur-md rounded-lg border border-nexus-border p-3">
                        <p className="text-[10px] font-bold text-nexus-text-secondary uppercase tracking-wider mb-2">Legend</p>
                        <div className="flex items-center gap-4 text-xs">
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-red-500" />
                            <span className="text-nexus-text-secondary">Critical</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-yellow-500" />
                            <span className="text-nexus-text-secondary">High</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-blue-500" />
                            <span className="text-nexus-text-secondary">Medium</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 rotate-45 bg-green-500" />
                            <span className="text-nexus-text-secondary">DB Lock (Target)</span>
                          </div>
                        </div>
                      </div>

                      {/* DB Lock Target indicator */}
                      <div className="absolute bottom-4 right-4 bg-green-500/20 backdrop-blur-md rounded-lg border border-green-500/30 px-4 py-2">
                        <div className="flex items-center gap-2">
                          <Lock className="w-4 h-4 text-green-400" />
                          <span className="text-sm font-medium text-green-400">DB Lock Ready (Target)</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Issue Summary Sidebar */}
              <div className="space-y-4">
                <Card className="glass-card border-nexus-border">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-white text-base flex items-center gap-2">
                      <Target className="w-4 h-4 text-purple-400" />
                      Issue Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 max-h-[300px] overflow-y-auto">
                    {sortedNodes.slice(0, 10).map((node) => {
                      const config = ISSUE_CONFIG[node.id];
                      return (
                        <div
                          key={node.id}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${selectedNode === node.id
                            ? 'bg-purple-500/20 border-purple-500/50'
                            : 'bg-nexus-card/50 border-nexus-border hover:border-nexus-text-secondary'
                            }`}
                          onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
                          style={{ borderLeftWidth: '4px', borderLeftColor: node.color }}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span style={{ color: node.color }}>{config?.icon}</span>
                              <span className="text-sm font-medium text-white">{node.name}</span>
                            </div>
                            <Badge
                              className="text-xs"
                              style={{ backgroundColor: `${node.color}33`, color: node.color }}
                            >
                              {node.patientCount.toLocaleString()}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between mt-2 text-xs text-nexus-text-secondary">
                            <span>Unlock Score: {node.unlockScore.toFixed(1)}</span>
                            <span>{node.responsible}</span>
                          </div>
                        </div>
                      );
                    })}
                  </CardContent>
                </Card>

                {/* Selected Node Impact */}
                {selectedNode && selectedNodeImpact && (
                  <Card className="glass-card border-nexus-border border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-nexus-card">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-white text-base flex items-center gap-2">
                        <Zap className="w-4 h-4 text-yellow-400" />
                        Cascade Impact
                      </CardTitle>
                      <CardDescription className="text-nexus-text-secondary">
                        Fix {ISSUE_CONFIG[selectedNode]?.name}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="bg-nexus-bg/50 rounded-lg p-2">
                          <p className="text-nexus-text-muted text-xs">Patients Unblocked</p>
                          <p className="text-white font-bold">{selectedNodeImpact.patientsUnblocked.toLocaleString()}</p>
                        </div>
                        <div className="bg-nexus-bg/50 rounded-lg p-2">
                          <p className="text-nexus-text-muted text-xs">DQI Improvement</p>
                          <p className="text-green-400 font-bold">+{selectedNodeImpact.dqiImprovement.toFixed(1)}</p>
                        </div>
                        <div className="bg-nexus-bg/50 rounded-lg p-2">
                          <p className="text-nexus-text-muted text-xs">Days Saved</p>
                          <p className="text-blue-400 font-bold">{selectedNodeImpact.daysSaved.toFixed(0)}</p>
                        </div>
                        <div className="bg-nexus-bg/50 rounded-lg p-2">
                          <p className="text-nexus-text-muted text-xs">Effort (hrs)</p>
                          <p className="text-orange-400 font-bold">{selectedNodeImpact.effortHours.toFixed(0)}</p>
                        </div>
                      </div>

                      <div className="bg-nexus-bg/50 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-nexus-text-secondary">ROI Score</span>
                          <Badge variant={
                            selectedNodeImpact.priority === 'Critical' ? 'error' :
                              selectedNodeImpact.priority === 'High' ? 'warning' : 'secondary'
                          }>
                            {selectedNodeImpact.priority}
                          </Badge>
                        </div>
                        <div className="text-2xl font-bold text-white">{selectedNodeImpact.roiScore.toFixed(1)}</div>
                      </div>

                      {selectedNodeImpact.directUnlocks.length > 0 && (
                        <div>
                          <p className="text-xs text-nexus-text-secondary mb-2">Direct Unlocks:</p>
                          <div className="flex flex-wrap gap-1">
                            {selectedNodeImpact.directUnlocks.map(unlock => (
                              <Badge
                                key={unlock}
                                variant="secondary"
                                className="text-[10px]"
                                style={{ borderColor: ISSUE_CONFIG[unlock]?.color }}
                              >
                                {ISSUE_CONFIG[unlock]?.name}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="flex gap-2">
                        <Button
                          onClick={handleApproveFix}
                          disabled={approveFixMutation.isPending}
                          className="flex-1 bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-500 hover:to-teal-600 text-white font-bold h-11"
                        >
                          {approveFixMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <CheckCircle2 className="w-4 h-4 mr-2" />}
                          APPROVE CASCADE FIX
                        </Button>
                        <Button variant="outline" className="border-nexus-border text-nexus-text-secondary hover:text-white h-11">
                          MODIFY
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="impact">
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white">Cascade Impact Analysis</CardTitle>
                <CardDescription className="text-nexus-text-secondary">
                  Fix these issues first for maximum downstream impact - ranked by ROI score
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Header Row */}
                  <div className="grid grid-cols-12 gap-4 px-4 py-2 text-[10px] font-bold text-nexus-text-muted uppercase tracking-widest border-b border-nexus-border/30">
                    <div className="col-span-1 text-center">#</div>
                    <div className="col-span-5 pr-4">Issue Description & Cascade Path</div>
                    <div className="col-span-1 text-center">Patients</div>
                    <div className="col-span-2 text-center">Unlock Score</div>
                    <div className="col-span-1 text-center">ROI</div>
                    <div className="col-span-2 text-right pr-2">Assignee</div>
                  </div>

                  {sortedNodes.slice(0, 12).map((node, idx) => {
                    const config = ISSUE_CONFIG[node.id];
                    const directUnlocks = ISSUE_DEPENDENCIES[node.id] || [];
                    const roiScore = (node.patientCount * 10 + node.unlockScore * 5) / Math.max(node.patientCount * 0.1, 1);

                    return (
                      <div
                        key={node.id}
                        className="grid grid-cols-12 gap-4 items-center p-4 rounded-xl bg-nexus-card/50 border border-nexus-border hover:border-nexus-text-secondary transition-all"
                        style={{ borderLeftWidth: '4px', borderLeftColor: node.color }}
                      >
                        <div className="col-span-1 flex items-center justify-center">
                          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-nexus-bg text-white font-bold text-sm">
                            {idx + 1}
                          </div>
                        </div>

                        <div className="col-span-5 flex items-center gap-3">
                          <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style={{ backgroundColor: `${node.color}20` }}>
                            <span style={{ color: node.color }}>{config?.icon}</span>
                          </div>
                          <div className="overflow-hidden">
                            <h4 className="text-white font-medium truncate">{node.name}</h4>
                            <p className="text-xs text-nexus-text-secondary truncate">
                              Unlocks: {directUnlocks.map(u => ISSUE_CONFIG[u]?.name).join(', ') || 'None'}
                            </p>
                          </div>
                        </div>

                        <div className="col-span-1 text-center">
                          <p className="text-[10px] text-nexus-text-muted uppercase tracking-wider mb-1">Patients</p>
                          <p className="text-sm font-semibold text-white">{node.patientCount.toLocaleString()}</p>
                        </div>

                        <div className="col-span-2 text-center">
                          <p className="text-[10px] text-nexus-text-muted uppercase tracking-wider mb-1">Unlock Score</p>
                          <p className="text-sm font-semibold" style={{ color: getImpactColor(node.unlockScore) }}>
                            {node.unlockScore.toFixed(1)}
                          </p>
                        </div>

                        <div className="col-span-1 text-center">
                          <p className="text-[10px] text-nexus-text-muted uppercase tracking-wider mb-1">ROI</p>
                          <p className="text-sm font-semibold text-green-400">{roiScore.toFixed(0)}</p>
                        </div>

                        <div className="col-span-2 text-right pr-2">
                          <p className="text-[10px] text-nexus-text-muted uppercase tracking-wider mb-1">Owner</p>
                          <p className="text-xs text-nexus-text-secondary font-medium">{node.responsible}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="critical">
            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="glass-card border-nexus-border">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-orange-400" />
                    Critical Path to DB Lock
                  </CardTitle>
                  <CardDescription className="text-nexus-text-secondary">
                    The sequence of issues blocking the most patients from DB Lock
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="relative">
                    {/* Critical path visualization */}
                    {Object.entries(LAYERS).map(([layerStr, types]) => {
                      const layer = parseInt(layerStr);
                      const maxPatients = Math.max(...types.map(t => issueCounts[t] || 0), 1);

                      return (
                        <div key={layer} className="mb-4 last:mb-0">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-6 h-6 rounded-full bg-nexus-bg flex items-center justify-center text-xs font-bold text-white">
                              {layer + 1}
                            </div>
                            <span className="text-xs text-nexus-text-secondary uppercase tracking-wider">
                              {layer === 0 ? 'Root Causes' : layer === 5 ? 'Target' : `Level ${layer}`}
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-2 ml-8">
                            {types.map(type => {
                              const config = ISSUE_CONFIG[type];
                              const count = issueCounts[type] || 0;
                              const width = (count / maxPatients) * 100;

                              return (
                                <div
                                  key={type}
                                  className="flex-1 min-w-[120px] bg-nexus-card rounded-lg p-3 border border-nexus-border"
                                  style={{ borderLeftWidth: '3px', borderLeftColor: config?.color }}
                                >
                                  <div className="flex items-center gap-2 mb-2">
                                    <span style={{ color: config?.color }}>{config?.icon}</span>
                                    <span className="text-xs font-medium text-white">{config?.name}</span>
                                  </div>
                                  <div className="h-2 bg-nexus-bg rounded-full overflow-hidden">
                                    <div
                                      className="h-full rounded-full transition-all"
                                      style={{ width: `${width}%`, backgroundColor: config?.color }}
                                    />
                                  </div>
                                  <p className="text-xs text-nexus-text-secondary mt-1">{count.toLocaleString()} patients</p>
                                </div>
                              );
                            })}
                          </div>
                          {layer < 5 && (
                            <div className="flex justify-center my-2">
                              <div className="w-0.5 h-4 bg-nexus-border" />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card border-nexus-border">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Clock className="w-5 h-5 text-blue-400" />
                    Bottleneck Analysis
                  </CardTitle>
                  <CardDescription className="text-nexus-text-secondary">
                    Issues causing the most downstream blockage
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {sortedNodes.slice(0, 8).map((node, idx) => {
                      const config = ISSUE_CONFIG[node.id];
                      const downstreamCount = (ISSUE_DEPENDENCIES[node.id] || []).length;
                      const blockageScore = node.patientCount * downstreamCount;
                      const maxBlockage = sortedNodes[0].patientCount * (ISSUE_DEPENDENCIES[sortedNodes[0].id] || []).length;
                      const blockagePercent = maxBlockage > 0 ? (blockageScore / maxBlockage) * 100 : 0;

                      return (
                        <div key={node.id} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-nexus-text-secondary">#{idx + 1}</span>
                              <span style={{ color: node.color }}>{config?.icon}</span>
                              <span className="text-sm text-white">{node.name}</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs">
                              <span className="text-nexus-text-secondary">
                                {node.patientCount.toLocaleString()} patients
                              </span>
                              <span className="text-nexus-text-secondary">
                                {downstreamCount} downstream
                              </span>
                            </div>
                          </div>
                          <div className="h-3 bg-nexus-bg rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${blockagePercent}%`,
                                background: `linear-gradient(90deg, ${node.color}, ${node.color}88)`
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-6 p-4 bg-nexus-bg/50 rounded-lg border border-nexus-border">
                    <h4 className="text-sm font-medium text-white mb-3">Recommendations</h4>
                    <ul className="space-y-2 text-sm text-nexus-text-secondary">
                      <li className="flex items-start gap-2">
                        <Zap className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                        Focus on root cause issues (Missing Visits, High Query Volume) first
                      </li>
                      <li className="flex items-start gap-2">
                        <Users className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                        Assign dedicated resources to safety-critical (SAE) issues
                      </li>
                      <li className="flex items-start gap-2">
                        <Target className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                        Target Signature Gaps as the final blocker before DB Lock
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
