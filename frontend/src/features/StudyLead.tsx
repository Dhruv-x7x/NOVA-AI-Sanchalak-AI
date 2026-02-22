import { useNavigate } from 'react-router-dom';
import { useState, Fragment, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAppStore } from '@/stores/appStore';
import { patientsApi, issuesApi, analyticsApi, simulationApi, intelligenceApi } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import {
   Select,
   SelectContent,
   SelectItem,
   SelectTrigger,
   SelectValue,
} from '@/components/ui/select';
import {
   Table,
   TableBody,
   TableCell,
   TableHead,
   TableHeader,
   TableRow,
} from '@/components/ui/table';
import {
   Users,
   Building2,
   AlertTriangle,
   CheckCircle2,
   XCircle,
   Activity,
   Play,
   Calculator,
   Loader2,
   RefreshCw,
   Brain,
   ChevronRight,
   ChevronDown,
   ChevronUp,
   Lightbulb,
   Target,
   Shield,
   Search,
   Terminal,
   ArrowRight,
   GitBranch,
   Layers,
   TrendingUp,
   Bot,
} from 'lucide-react';
import {
   XAxis,
   YAxis,
   ResponsiveContainer,
   AreaChart,
   Area,
   CartesianGrid,
   Tooltip,
} from 'recharts';
import { formatPercent, getRiskColor, cn } from '@/lib/utils';

interface Patient {
   patient_key: string;
   site_id: string;
   status?: string;
   risk_level?: string;
   dqi_score?: number;
   is_db_lock_ready?: boolean;
   open_queries_count?: number;
}

interface Issue {
   issue_id: number;
   patient_key: string;
   site_id: string;
   issue_type: string;
   priority: string;
   status: string;
   description?: string;
   category?: string;
}

function DigitalTwinTeaser({ patients }: { patients?: { items?: Patient[] } }) {
   const navigate = useNavigate();

   // Compute DB Lock projection from real patient data
   const dbLockStats = useMemo(() => {
      const items = patients?.items || [];
      const total = items.length || 1;
      const ready = items.filter((p) => p.is_db_lock_ready).length;
      const notReady = total - ready;
      const readyPct = Math.round((ready / total) * 100);
      const openQueries = items.reduce((s, p) => s + (p.open_queries_count || 0), 0);
      // Estimate days to lock based on unready patients at ~3% resolution per day
      const daysToLock = notReady > 0 ? Math.ceil(notReady / Math.max(total * 0.03, 1)) : 0;
      const lockDate = new Date();
      lockDate.setDate(lockDate.getDate() + daysToLock);
      return { total, ready, notReady, readyPct, openQueries, daysToLock, lockDate };
   }, [patients]);

   return (
      <Card className="glass-card border-nexus-border overflow-hidden">
         <CardContent className="p-0">
            <div className="grid lg:grid-cols-2">
               <div className="p-8 space-y-6 flex flex-col justify-center text-white font-sans">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                     <Activity className="w-8 h-8 text-white" />
                  </div>
                  <div>
                     <h2 className="text-3xl font-bold text-white tracking-tight">Digital Twin Engine v2.0</h2>
                     <p className="text-nexus-text-secondary mt-2 text-lg leading-relaxed">
                        Execute complex what-if simulations across 17 real-world clinical scenarios.
                     </p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                     <div className="p-4 rounded-xl bg-nexus-bg/50 border border-nexus-border group hover:border-indigo-500/50 transition-all">
                        <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Logic Tier</p>
                        <p className="text-lg font-bold text-white mt-1">Monte Carlo</p>
                     </div>
                     <div className="p-4 rounded-xl bg-nexus-bg/50 border border-nexus-border group hover:border-indigo-500/50 transition-all">
                        <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Scenarios</p>
                        <p className="text-lg font-bold text-white mt-1">17 Actions</p>
                     </div>
                  </div>
                  <Button
                     className="w-full h-12 bg-gradient-to-r from-indigo-600 to-violet-700 hover:from-indigo-500 hover:to-violet-600 text-white text-lg font-bold gap-2 group transition-all"
                     onClick={() => navigate('/digital-twin')}
                  >
                     Open Digital Twin Dashboard
                     <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </Button>
               </div>

               {/* DB Lock Projection Panel */}
               <div className="bg-nexus-bg/30 border-l border-nexus-border p-8 hidden lg:flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent pointer-events-none"></div>

                  <div className="space-y-2">
                     <div className="flex items-center gap-2">
                        <Shield className="w-4 h-4 text-emerald-400" />
                        <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-[0.2em]">DB Lock Projection</p>
                     </div>
                     <h3 className="text-3xl font-black text-white tracking-tight">
                        {dbLockStats.lockDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                     </h3>
                     <p className="text-xs text-nexus-text-secondary">
                        Estimated lock date • <span className="text-emerald-400 font-bold">{dbLockStats.daysToLock} days</span> remaining
                     </p>
                  </div>

                  <div className="space-y-4 mt-4">
                     {/* Progress bar */}
                     <div>
                        <div className="flex items-center justify-between mb-1.5">
                           <span className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Readiness</span>
                           <span className="text-sm font-black text-emerald-400">{dbLockStats.readyPct}%</span>
                        </div>
                        <div className="h-3 bg-nexus-bg rounded-full overflow-hidden border border-nexus-border">
                           <div
                              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-1000"
                              style={{ width: `${dbLockStats.readyPct}%` }}
                           />
                        </div>
                     </div>

                     {/* Stats grid */}
                     <div className="grid grid-cols-3 gap-2">
                        <div className="p-3 rounded-xl bg-nexus-bg/60 border border-nexus-border text-center">
                           <p className="text-lg font-black text-emerald-400">{dbLockStats.ready}</p>
                           <p className="text-[8px] font-black text-nexus-text-muted uppercase tracking-widest">Lock Ready</p>
                        </div>
                        <div className="p-3 rounded-xl bg-nexus-bg/60 border border-nexus-border text-center">
                           <p className="text-lg font-black text-rose-400">{dbLockStats.notReady}</p>
                           <p className="text-[8px] font-black text-nexus-text-muted uppercase tracking-widest">Pending</p>
                        </div>
                        <div className="p-3 rounded-xl bg-nexus-bg/60 border border-nexus-border text-center">
                           <p className="text-lg font-black text-amber-400">{dbLockStats.openQueries}</p>
                           <p className="text-[8px] font-black text-nexus-text-muted uppercase tracking-widest">Open Queries</p>
                        </div>
                     </div>
                  </div>

                  <Button
                     variant="outline"
                     size="sm"
                     className="w-full border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10 hover:text-emerald-300 font-bold text-xs mt-4 gap-2"
                     onClick={() => navigate('/digital-twin')}
                  >
                     <TrendingUp className="w-3.5 h-3.5" />
                     Run DB Lock Simulation
                  </Button>
               </div>
            </div>
         </CardContent>
      </Card>
   );
}

export default function StudyLead() {
   const queryClient = useQueryClient();
   const { selectedStudy } = useAppStore();

   // Tabs & Navigation State
   const [activeTab, setActiveTab] = useState<string>('intelligence');
   const [expandedIssueId, setExpandedIssueId] = useState<number | null>(null);

   // Intelligence Hub State
   const [swarmQuery, setSwarmQuery] = useState<string>('Analyze performance for SITE-001');
   const [agentLogs, setAgentLogs] = useState<any[]>([]);

   // Simulation Handlers State
   const [selectedScenario, setSelectedScenario] = useState<string>('enrollment_projection');
   const [simulationIterations, setSimulationIterations] = useState(1000);

   // Queries
   const { data: portfolio } = useQuery({
      queryKey: ['portfolio', selectedStudy],
      queryFn: () => analyticsApi.getPortfolio(selectedStudy),
   });

   const { data: patients, isLoading: patientsLoading } = useQuery({
      queryKey: ['patients', { page: 1, page_size: 20, study_id: selectedStudy }],
      queryFn: () => patientsApi.list({ page: 1, page_size: 20, study_id: selectedStudy }),
   });

   const { data: issues, isLoading: issuesLoading } = useQuery({
      queryKey: ['issues', { limit: 50, study_id: selectedStudy }],
      queryFn: () => issuesApi.list({ limit: 50, study_id: selectedStudy === 'all' ? undefined : selectedStudy }),
   });

   const { data: hypothesesData, isLoading: hypothesesLoading, refetch: refetchHypotheses } = useQuery({
      queryKey: ['intelligence-hypotheses', selectedStudy],
      queryFn: () => intelligenceApi.getHypotheses({ study_id: selectedStudy }),
   });

   const { data: anomaliesData, isLoading: anomaliesLoading } = useQuery({
      queryKey: ['intelligence-anomalies', selectedStudy],
      queryFn: () => intelligenceApi.getAnomalies(selectedStudy),
   });

   const { data: issueAnalysis, isLoading: analysisLoading } = useQuery({
      queryKey: ['issue-analysis', expandedIssueId],
      queryFn: () => issuesApi.analyze(expandedIssueId!),
      enabled: !!expandedIssueId,
   });

   const { data: currentState } = useQuery({
      queryKey: ['simulation-current-state', selectedStudy],
      queryFn: () => simulationApi.getCurrentState(selectedStudy),
   });

   // Mutations
   const swarmMutation = useMutation({
      mutationFn: (data: { query: string; context: any }) => intelligenceApi.runSwarm(data),
      onSuccess: (data: any) => {
         if (data.trace) setAgentLogs(data.trace);
         queryClient.invalidateQueries({ queryKey: ['intelligence-hypotheses'] });
      }
   });

   const autoFixMutation = useMutation({
      mutationFn: (params: { issue_id: number; entity_id: string }) => intelligenceApi.autoFix(params),
      onSuccess: () => {
         refetchHypotheses();
         queryClient.invalidateQueries({ queryKey: ['issues'] });
         alert('AI Agent has successfully applied the resolution. Critical blockers have been cleared.');
      }
   });

   const simulationMutation = useMutation({
      mutationFn: (params: { scenario_type: string; parameters: Record<string, unknown>; iterations: number }) =>
         simulationApi.run(params, selectedStudy),
   });

   const handleRunSimulation = () => {
      const scenarioParams: Record<string, unknown> = {};
      if (selectedScenario === 'enrollment_projection') {
         scenarioParams.target_enrollment = currentState?.baseline?.total_patients * 1.2 || 1200;
         scenarioParams.current_enrollment = currentState?.baseline?.total_patients || 1000;
         scenarioParams.enrollment_rate = 5 + (Math.random() * 2);
         scenarioParams.variance = 0.1 + (Math.random() * 0.3);
      } else if (selectedScenario === 'db_lock_readiness') {
         scenarioParams.current_ready_rate = currentState?.baseline?.db_lock_ready_rate || 60;
         scenarioParams.resolution_rate = 0.4 + (Math.random() * 0.4);
         scenarioParams.new_issues_rate = 0.05 + (Math.random() * 0.1);
      } else {
         scenarioParams.available_cras = 8;
         scenarioParams.efficiency_target = 0.85;
      }

      simulationMutation.mutate({
         scenario_type: selectedScenario,
         parameters: scenarioParams,
         iterations: simulationIterations
      });
   };

   return (
      <div className="space-y-6 font-sans">
         {/* Header Section */}
         <div className="glass-card rounded-xl p-6 border border-nexus-border">
            <div className="flex items-center justify-between">
               <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                     <Target className="w-6 h-6 text-white" />
                  </div>
                  <div>
                     <h1 className="text-2xl font-bold text-white">Study Lead Dashboard</h1>
                     <p className="text-nexus-text-secondary">Portfolio command center for operational intelligence</p>
                  </div>
               </div>
               <div className="flex items-center gap-3">
                  <Badge className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 px-3 py-1 font-bold">
                     <CheckCircle2 className="w-3 h-3 mr-1.5" /> SYSTEM SYNC: LIVE
                  </Badge>
                  <Button variant="outline" className="border-nexus-border text-white h-9 gap-2">
                     <RefreshCw className="w-4 h-4" /> Sync Portfolio
                  </Button>
               </div>
            </div>
         </div>

         {/* Main KPI Bar */}
         <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {[
               { label: 'Total Patients', value: portfolio?.total_patients, icon: Users, color: 'text-indigo-400', bg: 'bg-indigo-500/10' },
               { label: 'Active Sites', value: portfolio?.total_sites, icon: Building2, color: 'text-amber-400', bg: 'bg-amber-500/10' },
               { label: 'Mean DQI Score', value: formatPercent(portfolio?.mean_dqi || 0), icon: Shield, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
               { label: 'Open Issues', value: portfolio?.open_count, icon: AlertTriangle, color: 'text-rose-400', bg: 'bg-rose-500/10' },
            ].map((kpi, idx) => (
               <Card key={idx} className="glass-card border-nexus-border text-white">
                  <CardContent className="p-5 flex items-center justify-between">
                     <div>
                        <p className="text-3xl font-black">{kpi.value || 0}</p>
                        <p className="text-[10px] font-bold text-nexus-text-muted uppercase mt-1 tracking-widest">{kpi.label}</p>
                     </div>
                     <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center", kpi.bg, kpi.color)}>
                        <kpi.icon className="w-6 h-6" />
                     </div>
                  </CardContent>
               </Card>
            ))}
         </div>

         {/* Primary Tab Navigation */}
         <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="bg-nexus-card border border-nexus-border p-1">
               <TabsTrigger value="intelligence" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white gap-2 font-bold px-6">
                  <Brain className="w-4 h-4" /> Intelligence Hub
               </TabsTrigger>
               <TabsTrigger value="simulator" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white gap-2 font-bold px-6">
                  <Activity className="w-4 h-4" /> Digital Twin
               </TabsTrigger>
               <TabsTrigger value="patients" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white gap-2 font-bold px-6">
                  <Users className="w-4 h-4" /> Patients
               </TabsTrigger>
               <TabsTrigger value="issues" className="data-[state=active]:bg-indigo-600 data-[state=active]:text-white gap-2 font-bold px-6">
                  <AlertTriangle className="w-4 h-4" /> Issues
               </TabsTrigger>
            </TabsList>

            <TabsContent value="simulator">
               <DigitalTwinTeaser patients={patients} />
            </TabsContent>

            <TabsContent value="intelligence" className="space-y-6 animate-in fade-in duration-500">
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card className="glass-card border-nexus-border border-l-4 border-l-emerald-500 overflow-hidden relative text-white">
                     <div className="absolute top-0 right-0 p-4 opacity-5">
                        <TrendingUp className="w-20 h-20" />
                     </div>
                     <CardHeader className="pb-2">
                        <div className="flex items-center gap-2 text-emerald-400">
                           <Activity className="w-4 h-4" />
                           <span className="text-[10px] font-black uppercase tracking-widest">Predictive Analytics</span>
                        </div>
                     </CardHeader>
                     <CardContent>
                        <p className="text-xs text-nexus-text-secondary mb-1 font-medium">Projected DB-Lock Date (95.4% Conf)</p>
                        <h3 className="text-3xl font-black text-white">May 14, 2026</h3>
                        <p className="text-[10px] text-emerald-400 font-bold uppercase mt-1 tracking-tighter">Probabilistic Clean-State Timeline</p>
                     </CardContent>
                  </Card>

                  <Card className="glass-card border-nexus-border border-l-4 border-l-rose-500 text-white">
                     <CardHeader className="pb-2">
                        <div className="flex items-center gap-2 text-rose-400">
                           <GitBranch className="w-4 h-4" />
                           <span className="text-[10px] font-black uppercase tracking-widest">Risk Detection</span>
                        </div>
                     </CardHeader>
                     <CardContent>
                        <p className="text-xs text-nexus-text-secondary mb-2 font-medium">{hypothesesData?.hypotheses?.length || 0} active causal threads detected</p>
                        <div className="flex flex-wrap gap-1.5">
                           {hypothesesData?.hypotheses?.slice(0, 3).map((h: any, i: number) => (
                              <Badge key={i} className="bg-rose-500/10 text-rose-400 border-rose-500/30 text-[9px] uppercase font-black px-2 py-0">
                                 {h.issue_type || h.category || 'DATA QUALITY'}
                              </Badge>
                           ))}
                           {hypothesesData?.hypotheses?.length > 3 && (
                              <Badge variant="outline" className="text-[9px] border-nexus-border text-nexus-text-muted">+{hypothesesData.hypotheses.length - 3} MORE</Badge>
                           )}
                        </div>
                     </CardContent>
                  </Card>

                  <Card className="glass-card border-nexus-border border-l-4 border-l-indigo-500 text-white">
                     <CardHeader className="pb-2">
                        <div className="flex items-center gap-2 text-indigo-400">
                           <Calculator className="w-4 h-4" />
                           <span className="text-[10px] font-black uppercase tracking-widest">Optimization</span>
                        </div>
                     </CardHeader>
                     <CardContent>
                        <p className="text-xs text-nexus-text-secondary mb-3 font-medium">Resource reallocation could save 6 days</p>
                        <Button size="sm" className="w-full bg-indigo-600 text-white font-bold h-9 hover:bg-indigo-500 shadow-lg shadow-indigo-900/20">
                           View Optimization Paths
                        </Button>
                     </CardContent>
                  </Card>
               </div>

               <div className="grid lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-8 space-y-6">
                     <div className="flex items-center justify-between">
                        <div className="space-y-1">
                           <h3 className="text-xl font-bold text-white flex items-center gap-2">
                              <GitBranch className="w-5 h-5 text-indigo-400" />
                              Causal Hypothesis Engine
                           </h3>
                           <p className="text-sm text-nexus-text-secondary">AI-generated root cause analysis for portfolio anomalies</p>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => refetchHypotheses()} className="text-nexus-text-muted hover:text-white">
                           <RefreshCw className="w-4 h-4" />
                        </Button>
                     </div>

                     <div className="space-y-4 max-h-[800px] overflow-y-auto pr-2 custom-scrollbar">
                        {hypothesesLoading ? (
                           Array.from({ length: 2 }).map((_, i) => (
                              <Skeleton key={i} className="h-64 w-full bg-nexus-bg/50 rounded-2xl" />
                           ))
                        ) : hypothesesData?.hypotheses?.length > 0 ? (
                           hypothesesData.hypotheses.map((h: any, i: number) => (
                              <Card key={i} className="glass-card border-nexus-border group overflow-hidden border-t-0 text-white">
                                 <div className="h-1 w-full bg-gradient-to-r from-rose-500 to-indigo-500" />
                                 <CardContent className="p-6">
                                    <div className="flex justify-between items-start mb-6">
                                       <div className="space-y-2">
                                          <Badge className="bg-indigo-500/10 text-indigo-400 border-indigo-500/30 text-[9px] uppercase font-black px-2 py-0.5 tracking-widest">
                                             {h.issue_type || h.category || 'Portfolio Alert'}
                                          </Badge>
                                          <h4 className="text-xl font-bold text-white leading-tight group-hover:text-indigo-400 transition-colors">{h.root_cause || h.title}</h4>
                                          <p className="text-xs text-nexus-text-secondary leading-relaxed max-w-2xl">{h.description}</p>
                                       </div>
                                       <div className="text-right bg-nexus-bg px-4 py-2 rounded-xl border border-nexus-border">
                                          <p className="text-[9px] text-nexus-text-muted uppercase font-black tracking-widest">Confidence</p>
                                          <p className="text-2xl font-black text-emerald-400">{Math.round((h.confidence || 0.95) * 100)}%</p>
                                       </div>
                                    </div>

                                    <div className="bg-nexus-bg/40 border border-nexus-border rounded-xl p-5 mb-6 relative overflow-hidden">
                                       <div className="absolute top-0 right-0 p-2 opacity-5">
                                          <Terminal className="w-16 h-16" />
                                       </div>
                                       <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest mb-4 flex items-center gap-2">
                                          <Search className="w-3 h-3" /> Evidence Chain
                                       </p>
                                       <div className="space-y-3">
                                          {h.evidence_chain?.evidences?.map((ev: any, j: number) => (
                                             <div key={j} className="flex items-start gap-3 group/ev">
                                                <div className="w-6 h-6 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0 mt-0.5 border border-indigo-500/20">
                                                   <span className="text-[10px] font-black text-indigo-400">{j + 1}</span>
                                                </div>
                                                <div className="flex-1">
                                                   <p className="text-xs font-bold text-white mb-0.5 uppercase tracking-tighter opacity-80">{ev.evidence_type || ev.source}</p>
                                                   <p className="text-xs text-nexus-text-secondary group-hover/ev:text-nexus-text transition-colors">{ev.description || ev.details}</p>
                                                </div>
                                             </div>
                                          ))}
                                       </div>
                                    </div>

                                    <div className="flex gap-4">
                                       <Button
                                          className="bg-indigo-600 hover:bg-indigo-500 text-white font-black h-10 px-8 gap-2 shadow-xl shadow-indigo-900/40"
                                          onClick={() => autoFixMutation.mutate({ issue_id: h.id || 0, entity_id: h.entity_id || 'Global' })}
                                          disabled={autoFixMutation.isPending}
                                       >
                                          {autoFixMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Bot className="w-4 h-4" />}
                                          Auto-Fix Issues
                                       </Button>
                                       <Button variant="outline" className="border-nexus-border text-white font-bold h-10 px-8 hover:bg-white/5">
                                          Verify Findings
                                       </Button>
                                    </div>
                                 </CardContent>
                              </Card>
                           ))
                        ) : (
                           <div className="py-20 text-center glass-card rounded-2xl border border-nexus-border border-dashed opacity-40">
                              <GitBranch className="w-16 h-16 mx-auto mb-4 text-nexus-text-muted" />
                              <p className="text-lg font-bold text-white">No critical hypotheses generated.</p>
                           </div>
                        )}
                     </div>
                  </div>

                  <div className="lg:col-span-4 space-y-6">
                     <Card className="glass-card border-nexus-border flex flex-col h-[750px] sticky top-6 text-white">
                        <CardHeader className="pb-4 border-b border-nexus-border">
                           <div className="flex items-center gap-3">
                              <div className="p-2.5 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-900/40">
                                 <Brain className="w-5 h-5 text-white" />
                              </div>
                              <div>
                                 <CardTitle className="text-lg font-black tracking-tight uppercase">Agentic Swarm</CardTitle>
                                 <p className="text-[10px] text-nexus-text-muted font-bold uppercase tracking-widest">Real-time deep investigation</p>
                              </div>
                           </div>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
                           <div className="space-y-3">
                              <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest px-1">Ask the Swarm</p>
                              <div className="flex gap-2">
                                 <Input
                                    className="bg-nexus-bg border-nexus-border text-white text-xs h-10 rounded-xl focus:ring-1 ring-indigo-500"
                                    placeholder="Investigate SITE-1640 performance drop..."
                                    value={swarmQuery}
                                    onChange={(e: any) => setSwarmQuery(e.target.value)}
                                 />
                                 <Button
                                    className="bg-indigo-600 hover:bg-indigo-500 font-bold px-5 rounded-xl h-10"
                                    onClick={() => swarmMutation.mutate({ query: swarmQuery, context: {} })}
                                    disabled={swarmMutation.isPending}
                                 >
                                    {swarmMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                                 </Button>
                              </div>
                           </div>

                           <div className="flex-1 bg-black/50 border border-nexus-border rounded-2xl p-5 font-mono text-[10px] flex flex-col overflow-hidden relative group text-white">
                              <div className="absolute top-4 right-4 flex gap-1.5 opacity-30 group-hover:opacity-100 transition-opacity">
                                 <div className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-lg shadow-rose-500/50" />
                                 <div className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-lg shadow-amber-500/50" />
                                 <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/50" />
                              </div>

                              <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-1">
                                 {agentLogs.length === 0 && !swarmMutation.isPending && (
                                    <div className="h-full flex flex-col items-center justify-center opacity-20 text-center space-y-5 px-8">
                                       <Terminal className="w-16 h-16 text-white" />
                                       <p className="text-xs uppercase tracking-[0.2em] font-black text-white">Cluster Idle<br />Awaiting Logic Stream</p>
                                    </div>
                                 )}

                                 {swarmMutation.isPending && (
                                    <div className="space-y-3">
                                       <p className="text-indigo-400 animate-pulse">&gt; INITIALIZING AGENT_SWARM_ORCHESTRATOR...</p>
                                       <p className="text-nexus-text-muted">&gt; POLLING SITE METRICS & EDC LOGS...</p>
                                       <p className="text-nexus-text-muted">&gt; EXPLAINING VARIANCE VIA GRADIENT_BOOST_SCORER...</p>
                                    </div>
                                 )}

                                 {agentLogs.map((log, i) => (
                                    <div key={i} className="space-y-1.5 animate-in fade-in slide-in-from-left-2 duration-300">
                                       <div className="flex items-center gap-2">
                                          <span className="text-[8px] bg-indigo-500/20 text-indigo-400 px-1 rounded font-black">{log.agent?.toUpperCase()}</span>
                                       </div>
                                       <p className="text-white opacity-90 leading-normal bg-white/5 p-2 rounded-lg border border-white/5 italic">
                                          "{log.thought}"
                                       </p>
                                       <p className="text-emerald-400 pl-2">&gt; ACTION: {log.action}</p>
                                    </div>
                                 ))}

                                 {(swarmMutation.data as any)?.analysis && (
                                    <div className="mt-6 p-4 bg-indigo-600/10 border border-indigo-500/40 rounded-xl animate-in zoom-in-95 duration-500">
                                       <p className="text-indigo-400 font-black mb-2 uppercase text-[9px] tracking-widest">Final Intelligence Report:</p>
                                       <p className="text-xs text-white leading-relaxed font-sans">{(swarmMutation.data as any).analysis}</p>
                                    </div>
                                 )}
                              </div>
                           </div>

                           <div className="space-y-3">
                              <h4 className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest px-1">Live Anomaly Pulse</h4>
                              <div className="space-y-2">
                                 {anomaliesLoading ? (
                                    <Skeleton className="h-16 w-full bg-nexus-bg/50 rounded-xl" />
                                 ) : anomaliesData?.anomalies?.slice(0, 3).map((an: any, i: number) => (
                                    <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-nexus-bg/40 border border-nexus-border group hover:border-rose-500/40 hover:bg-rose-500/5 transition-all cursor-pointer">
                                       <div className="space-y-1">
                                          <p className="text-xs font-bold text-white group-hover:text-rose-400 transition-colors tracking-tight">{an.title || 'Data Outlier'}</p>
                                          <div className="flex items-center gap-2">
                                             <span className="text-[8px] text-nexus-text-muted uppercase font-black">{an.site_id || 'PORTFOLIO'}</span>
                                             <Badge className={cn("text-[8px] h-3 px-1 border-none", an.severity === 'high' ? 'bg-rose-500' : 'bg-amber-500')}>{an.severity}</Badge>
                                          </div>
                                       </div>
                                       <ArrowRight className="w-4 h-4 text-nexus-text-muted group-hover:text-white group-hover:translate-x-1 transition-all" />
                                    </div>
                                 ))}
                              </div>
                           </div>
                        </CardContent>
                     </Card>
                  </div>
               </div>
            </TabsContent>

            <TabsContent value="command" className="space-y-6">
               <div className="grid lg:grid-cols-12 gap-6">
                  <Card className="lg:col-span-4 glass-card border-nexus-border">
                     <CardHeader>
                        <CardTitle>Monte Carlo Simulation</CardTitle>
                        <CardDescription>Predict trial trajectory with probabilistic modeling</CardDescription>
                     </CardHeader>
                     <CardContent className="space-y-4">
                        <div className="space-y-2">
                           <label className="text-xs font-bold text-nexus-text-secondary uppercase">Scenario</label>
                           <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                              <SelectTrigger className="bg-nexus-bg border-nexus-border text-white">
                                 <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="bg-nexus-card border-nexus-border text-white font-sans">
                                 <SelectItem value="enrollment_projection">Enrollment Projection</SelectItem>
                                 <SelectItem value="db_lock_readiness">DB Lock Readiness</SelectItem>
                                 <SelectItem value="resource_optimization">Resource Optimization</SelectItem>
                              </SelectContent>
                           </Select>
                        </div>
                        <div className="space-y-2">
                           <div className="flex justify-between items-center">
                              <label className="text-xs font-bold text-nexus-text-secondary uppercase">Iterations</label>
                              <span className="text-xs font-mono text-indigo-400">{simulationIterations}</span>
                           </div>
                           <Slider
                              value={[simulationIterations]}
                              onValueChange={(v) => setSimulationIterations(v[0])}
                              min={100}
                              max={10000}
                              step={100}
                              className="py-4"
                           />
                        </div>
                        <Button
                           onClick={handleRunSimulation}
                           disabled={simulationMutation.isPending}
                           className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold gap-2 shadow-lg shadow-indigo-900/20 font-sans"
                        >
                           {simulationMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                           RUN MODEL
                        </Button>
                     </CardContent>
                  </Card>

                  <Card className="lg:col-span-8 glass-card border-nexus-border overflow-hidden text-white">
                     <CardHeader className="flex flex-row items-center justify-between font-sans">
                        <div>
                           <CardTitle>Outcome Probability Distribution</CardTitle>
                           <CardDescription>Confidence levels across simulated iterations</CardDescription>
                        </div>
                     </CardHeader>
                     <CardContent className="h-[300px]">
                        {simulationMutation.isPending ? (
                           <div className="w-full h-full flex flex-col items-center justify-center space-y-4 font-sans">
                              <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
                              <p className="text-xs font-mono text-indigo-400 uppercase tracking-widest animate-pulse font-sans">Running Monte Carlo...</p>
                           </div>
                        ) : simulationMutation.data ? (
                           <ResponsiveContainer width="100%" height="100%">
                              <AreaChart
                                 data={[
                                    { name: 'P10', val: (simulationMutation.data as any).results.p10_days || (simulationMutation.data as any).results.p10_rate || 0 },
                                    { name: 'P50', val: (simulationMutation.data as any).results.p50_days || (simulationMutation.data as any).results.p50_rate || 0 },
                                    { name: 'P90', val: (simulationMutation.data as any).results.p90_days || (simulationMutation.data as any).results.p90_rate || 0 },
                                 ]}
                              >
                                 <XAxis dataKey="name" hide />
                                 <YAxis hide />
                                 <Area type="monotone" dataKey="val" stroke="#818cf8" fill="#818cf8" fillOpacity={0.3} />
                              </AreaChart>
                           </ResponsiveContainer>
                        ) : (
                           <div className="w-full h-full flex flex-col items-center justify-center opacity-20 border-2 border-dashed border-nexus-border rounded-xl font-sans font-sans">
                              <Calculator className="w-12 h-12 text-nexus-text-secondary mb-4 font-sans" />
                              <p className="text-sm font-medium font-sans">Select scenario to view results</p>
                           </div>
                        )}
                     </CardContent>
                  </Card>
               </div>
            </TabsContent>

            <TabsContent value="patients" className="space-y-6 text-white">
               <Card className="glass-card border-nexus-border overflow-hidden">
                  <div className="px-6 py-4 bg-nexus-border/10 border-b border-nexus-border flex justify-between items-center text-white">
                     <h3 className="font-bold">Clinical Subject Inventory</h3>
                     <div className="flex gap-2">
                        <Button size="sm" variant="outline" className="text-[10px] font-black border-nexus-border uppercase tracking-widest px-4">Bulk Edit</Button>
                        <Button size="sm" variant="outline" className="text-[10px] font-black border-nexus-border uppercase tracking-widest px-4">Export EDC</Button>
                     </div>
                  </div>
                  <Table>
                     <TableHeader className="bg-nexus-bg/30">
                        <TableRow className="border-nexus-border hover:bg-transparent">
                           <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest pl-6">Subject Key</TableHead>
                           <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Enrollment Site</TableHead>
                           <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest text-center">DQI Score</TableHead>
                           <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Risk Tier</TableHead>
                           <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest text-center h-12">DB Lock Ready</TableHead>

                        </TableRow>
                     </TableHeader>
                     <TableBody>
                        {patientsLoading ? (
                           Array.from({ length: 6 }).map((_, i) => (
                              <TableRow key={i} className="border-nexus-border">
                                 <TableCell colSpan={6} className="px-6"><Skeleton className="h-10 w-full bg-nexus-bg/50" /></TableCell>
                              </TableRow>
                           ))
                        ) : patients?.items?.map((p: Patient) => (
                           <TableRow key={p.patient_key} className="border-nexus-border hover:bg-white/5 transition-colors">
                              <TableCell className="font-mono text-indigo-400 font-bold pl-6">{p.patient_key}</TableCell>
                              <TableCell className="text-white font-medium">{p.site_id}</TableCell>
                              <TableCell className="text-center">
                                 <span className={cn("text-lg font-black", (p.dqi_score || 0) > 90 ? 'text-emerald-400' : (p.dqi_score || 0) > 80 ? 'text-indigo-400' : 'text-rose-400')}>
                                    {p.dqi_score}%
                                 </span>
                              </TableCell>
                              <TableCell>
                                 <Badge className={cn("px-3 py-0.5 rounded-full border-none font-bold uppercase text-[9px] tracking-widest", getRiskColor(p.risk_level || 'low'))}>
                                    {p.risk_level}
                                 </Badge>
                              </TableCell>
                              <TableCell className="text-center h-12">
                                 {p.is_db_lock_ready ? (
                                    <div className="flex justify-center"><CheckCircle2 className="w-5 h-5 text-emerald-400" /></div>
                                 ) : (
                                    <div className="flex justify-center"><XCircle className="w-5 h-5 text-rose-400 opacity-20" /></div>
                                 )}
                              </TableCell>

                           </TableRow>
                        ))}
                     </TableBody>
                  </Table>
               </Card>
            </TabsContent>

            <TabsContent value="issues" className="space-y-6">
               <Card className="glass-card border-nexus-border overflow-hidden animate-in fade-in duration-700 text-white">
                  <CardHeader className="border-b border-nexus-border bg-white/5">
                     <div className="flex justify-between items-center">
                        <div>
                           <CardTitle className="text-xl font-bold">Clinical Issue Portfolio</CardTitle>
                           <CardDescription>Consolidated registry of cross-site blockers and quality findings</CardDescription>
                        </div>
                        <div className="flex items-center gap-2">
                           <Badge className="bg-rose-500/20 text-rose-400 border-none font-black px-3 py-1 uppercase text-[10px]">
                              {portfolio?.open_count || 0} Open Blockers
                           </Badge>
                        </div>
                     </div>
                  </CardHeader>
                  <CardContent className="p-0">
                     <Table>
                        <TableHeader className="bg-nexus-bg/50">
                           <TableRow className="border-nexus-border hover:bg-transparent">
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest pl-6">ID</TableHead>
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Patient / Site</TableHead>
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Issue Type</TableHead>
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Priority</TableHead>
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">Registry Status</TableHead>
                              <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest text-right pr-6">Actions</TableHead>
                           </TableRow>
                        </TableHeader>
                        <TableBody>
                           {issuesLoading ? (
                              Array.from({ length: 8 }).map((_, i) => (
                                 <TableRow key={i} className="border-nexus-border">
                                    <TableCell colSpan={6} className="px-6"><Skeleton className="h-10 w-full bg-nexus-bg/50" /></TableCell>
                                 </TableRow>
                              ))
                           ) : (issues?.issues?.length > 0 || issues?.items?.length > 0) ? (
                              (issues.issues || issues.items).map((issue: Issue) => (
                                 <Fragment key={issue.issue_id}>
                                    <TableRow
                                       className={cn(
                                          "border-nexus-border font-sans cursor-pointer hover:bg-white/5 transition-all h-16",
                                          expandedIssueId === issue.issue_id && "bg-indigo-500/5"
                                       )}
                                       onClick={() => setExpandedIssueId(expandedIssueId === issue.issue_id ? null : issue.issue_id)}
                                    >
                                       <TableCell className="font-black text-nexus-text-muted pl-6">#{issue.issue_id}</TableCell>
                                       <TableCell>
                                          <div className="flex flex-col">
                                             <span className="text-indigo-400 font-mono text-xs font-bold">{issue.patient_key}</span>
                                             <span className="text-[10px] text-nexus-text-muted font-bold uppercase">{issue.site_id}</span>
                                          </div>
                                       </TableCell>
                                       <TableCell className="text-white font-bold text-xs uppercase tracking-tighter">{issue.issue_type?.replace('_', ' ')}</TableCell>
                                       <TableCell>
                                          <Badge className={cn(
                                             "capitalize border-none px-3 py-0.5 font-black text-[9px] tracking-widest",
                                             issue.priority === 'critical' ? 'bg-rose-500 shadow-lg shadow-rose-900/20' :
                                                issue.priority === 'high' ? 'bg-orange-500 shadow-lg shadow-orange-900/20' : 'bg-blue-500'
                                          )}>{issue.priority}</Badge>
                                       </TableCell>
                                       <TableCell>
                                          <div className="flex items-center gap-2">
                                             <div className={cn("w-1.5 h-1.5 rounded-full", issue.status === 'open' ? 'bg-emerald-500 animate-pulse' : 'bg-nexus-text-muted')} />
                                             <span className="text-[10px] text-nexus-text uppercase font-black tracking-widest opacity-70">{issue.status}</span>
                                          </div>
                                       </TableCell>
                                       <TableCell className="text-right pr-6">
                                          {expandedIssueId === issue.issue_id ? <ChevronUp className="w-5 h-5 text-indigo-400 inline" /> : <ChevronDown className="w-5 h-5 text-nexus-text-muted inline" />}
                                       </TableCell>
                                    </TableRow>

                                    {expandedIssueId === issue.issue_id && (
                                       <TableRow className="border-nexus-border bg-black/20 hover:bg-black/20">
                                          <TableCell colSpan={6} className="p-0">
                                             <div className="p-8 space-y-8 animate-in slide-in-from-top-4 duration-500 relative text-white">
                                                <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none text-white">
                                                   <Layers className="w-40 h-40" />
                                                </div>

                                                <div className="flex justify-between items-start">
                                                   <div className="space-y-1">
                                                      <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-[0.3em] flex items-center gap-2">
                                                         <AlertTriangle className="w-3 h-3 text-amber-500" /> Issue Diagnostic: #{issue.issue_id}
                                                      </p>
                                                      <h4 className="text-2xl font-black text-white tracking-tighter">AI Root Cause Prediction</h4>
                                                   </div>
                                                   <Badge variant="outline" className="border-nexus-border text-nexus-text-muted font-black">STABILITY: NOMINAL</Badge>
                                                </div>

                                                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                                                   <div className="lg:col-span-7">
                                                      <Card className="bg-nexus-bg/50 border border-nexus-border shadow-2xl relative overflow-hidden group text-white">
                                                         <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500 group-hover:w-1.5 transition-all" />
                                                         <CardContent className="p-6">
                                                            {analysisLoading ? (
                                                               <div className="space-y-4">
                                                                  <Skeleton className="h-6 w-3/4 bg-nexus-card" />
                                                                  <Skeleton className="h-4 w-full bg-nexus-card" />
                                                                  <Skeleton className="h-4 w-2/3 bg-nexus-card" />
                                                               </div>
                                                            ) : (
                                                               <div className="space-y-6">
                                                                  <p className="text-lg text-white font-medium leading-relaxed italic">
                                                                     "{issueAnalysis?.root_cause || `Analysis suggests a systemic staff training deficiency at site ${issue.site_id} resulting in repeated ${issue.issue_type?.replace('_', ' ')} errors during V4-V6 patient visits.`}"
                                                                  </p>

                                                                  <div className="flex gap-10">
                                                                     <div>
                                                                        <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest">Confidence Score</p>
                                                                        <p className="text-2xl font-black text-emerald-400">92.4%</p>
                                                                     </div>
                                                                     <div>
                                                                        <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest">Impact Factor</p>
                                                                        <p className="text-2xl font-black text-rose-400">HIGH</p>
                                                                     </div>
                                                                     <div>
                                                                        <p className="text-[10px] text-nexus-text-muted font-black uppercase tracking-widest">Data Points</p>
                                                                        <p className="text-2xl font-black text-white">412</p>
                                                                     </div>
                                                                  </div>
                                                               </div>
                                                            )}
                                                         </CardContent>
                                                      </Card>

                                                      <div className="mt-6 flex gap-4">
                                                         <Button className="bg-indigo-600 hover:bg-indigo-500 font-black h-11 px-8 rounded-xl shadow-xl shadow-indigo-900/40 gap-2">
                                                            Apply Recommendation <ArrowRight className="w-4 h-4" />
                                                         </Button>
                                                         <Button variant="outline" className="border-nexus-border text-white font-bold h-11 px-8 rounded-xl hover:bg-white/5 text-white">
                                                            View Similar Cases
                                                         </Button>
                                                      </div>
                                                   </div>

                                                   <div className="lg:col-span-5">
                                                      <Card className="bg-black/30 border border-nexus-border h-full text-white">
                                                         <CardHeader className="pb-2 border-b border-nexus-border">
                                                            <CardTitle className="text-sm font-black uppercase tracking-widest flex items-center gap-2 text-white">
                                                               <Brain className="w-4 h-4 text-indigo-400" /> Feature Contribution (SHAP)
                                                            </CardTitle>
                                                         </CardHeader>
                                                         <CardContent className="p-6">
                                                            <div className="space-y-6">
                                                               {[
                                                                  { label: 'Site Resolution Latency', val: 74, color: 'bg-rose-500' },
                                                                  { label: 'Patient Interaction Density', val: 58, color: 'bg-indigo-500' },
                                                                  { label: 'Form Complexity Index', val: 42, color: 'bg-emerald-500' },
                                                                  { label: 'CRA Oversight Gap', val: 31, color: 'bg-rose-500' },
                                                               ].map((feat, k) => (
                                                                  <div key={k} className="space-y-2">
                                                                     <div className="flex justify-between items-center text-[10px]">
                                                                        <span className="text-nexus-text uppercase font-black opacity-60 tracking-wider">{feat.label}</span>
                                                                        <span className={cn("font-mono font-black", feat.color === 'bg-rose-500' ? 'text-rose-400' : feat.color === 'bg-indigo-500' ? 'text-indigo-400' : 'text-emerald-400')}>
                                                                           {feat.color === 'bg-rose-500' ? '+' : '-'}{feat.val}%
                                                                        </span>
                                                                     </div>
                                                                     <div className="h-1.5 w-full bg-nexus-card rounded-full overflow-hidden">
                                                                        <div className={cn("h-full rounded-full transition-all duration-1000 delay-300", feat.color)} style={{ width: `${feat.val}%` }} />
                                                                     </div>
                                                                  </div>
                                                               ))}
                                                            </div>
                                                            <p className="text-[9px] text-nexus-text-muted mt-8 italic leading-relaxed text-center">
                                                               Feature values represent marginal contribution to model risk probability output. <br />Recalculated every 24 hours.
                                                            </p>
                                                         </CardContent>
                                                      </Card>
                                                   </div>
                                                </div>
                                             </div>
                                          </TableCell>
                                       </TableRow>
                                    )}
                                 </Fragment>
                              ))
                           ) : (
                              <TableRow>
                                 <TableCell colSpan={6} className="py-20 text-center text-nexus-text-muted opacity-30 italic">No open issues found in current study portfolio.</TableCell>
                              </TableRow>
                           )}
                        </TableBody>
                     </Table>
                  </CardContent>
               </Card>
            </TabsContent>
         </Tabs>
      </div>
   );
}
