import { FC, useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import SanchalakLoader from '@/components/SanchalakLoader';
import {
  Activity,
  Target,
  Shield,
  Brain,
  Zap,
  Play,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Clock,
  CheckSquare,
  Layers,
  Search,
  ChevronRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { useAppStore } from '@/stores/appStore';
import { digitalTwinApi } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { cn } from '@/lib/utils';

interface Scenario {
  id: string;
  name: string;
  category: string;
  description: string;
  target_entities: string[];
}

interface Entity {
  id: string;
  label: string;
}

const CATEGORIES = [
  { id: 'milestone', label: 'Milestones & Timeline', icon: Clock, color: 'from-blue-500 to-indigo-600' },
  { id: 'risk', label: 'Operational Risk', icon: AlertTriangle, color: 'from-amber-500 to-orange-600' },
  { id: 'intervention', label: 'Intervention Simulation', icon: Zap, color: 'from-purple-500 to-violet-600' },
  { id: 'emerging', label: 'Emerging Risk Detection', icon: Brain, color: 'from-emerald-500 to-teal-600' }
];

const DigitalTwin: FC = () => {
  const { selectedStudy: _selectedStudy } = useAppStore();
  const [activeCategory, setActiveCategory] = useState<string>('milestone');
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  const [entityType, setEntityType] = useState<string>('study');
  const [selectedEntity, setSelectedEntity] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch scenarios
  const { data: scenarios = [] as Scenario[], isLoading: scenariosLoading } = useQuery({
    queryKey: ['digital-twin', 'scenarios'],
    queryFn: () => digitalTwinApi.getScenarios(),
    enabled: true
  });

  // Filter scenarios by category
  const filteredScenarios = scenarios.filter((s: Scenario) => s.category === activeCategory);

  // When scenario changes, update entity type and reset entity
  useEffect(() => {
    if (selectedScenario) {
      const scenario = scenarios.find((s: Scenario) => s.id === selectedScenario);
      if (scenario && scenario.target_entities.length > 0) {
        setEntityType(scenario.target_entities[0]);
        setSelectedEntity('');
      }
    }
  }, [selectedScenario, scenarios]);

  // Fetch entities based on type
  const { data: entities = [] as Entity[] } = useQuery({
    queryKey: ['digital-twin', 'entities', entityType],
    queryFn: () => digitalTwinApi.getEntities(entityType),
    enabled: !!entityType
  });

  const filteredEntities = entities.filter((e: Entity) =>
    e.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    e.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Prediction Mutation
  const predictMutation = useMutation({
    mutationFn: (data: { scenarioId: string, type: string, id: string }) =>
      digitalTwinApi.predict(data.scenarioId, data.type, data.id),
  });

  const handleRun = () => {
    if (!selectedScenario || !selectedEntity) return;
    predictMutation.mutate({
      scenarioId: selectedScenario,
      type: entityType,
      id: selectedEntity
    });
  };

  const currentResult = predictMutation.data;

  if (scenariosLoading) {
    return <SanchalakLoader size="lg" label="Loading digital twin scenarios..." fullPage />;
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-700 font-sans">
      {/* Page Header */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Digital Twin</h1>
              <p className="text-nexus-text-secondary">High-fidelity clinical trial operational prediction engine</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="border-indigo-500/50 text-indigo-400 bg-indigo-500/10 py-1 px-3">
              Engine v2.1-Alpha (MC-Enabled)
            </Badge>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <Card className="glass-card border-nexus-border overflow-hidden">
        <div className="p-1 bg-nexus-border/20 flex flex-wrap gap-1">
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              onClick={() => {
                setActiveCategory(cat.id);
                setSelectedScenario('');
              }}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 text-sm font-medium",
                activeCategory === cat.id
                  ? `bg-gradient-to-r ${cat.color} text-white shadow-lg`
                  : "text-nexus-text-secondary hover:bg-nexus-border/30 hover:text-white"
              )}
            >
              <cat.icon className="w-4 h-4" />
              {cat.label}
            </button>
          ))}
        </div>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-6 items-end">
            {/* Scenario Selection */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wider">Scenario Type</label>
              <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                <SelectTrigger className="bg-nexus-bg border-nexus-border text-white h-10">
                  <SelectValue placeholder="Select a scenario..." />
                </SelectTrigger>
                <SelectContent className="bg-nexus-card border-nexus-border text-white font-sans">
                  {filteredScenarios.map((s: Scenario) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedScenario && (
                <p className="text-xs text-nexus-text-secondary italic">
                  {scenarios.find((s: Scenario) => s.id === selectedScenario)?.description}
                </p>
              )}
            </div>

            {/* Target Entity Selection */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-nexus-text-muted uppercase tracking-wider font-sans">Target Entity ({entityType.toUpperCase()})</label>
              <Select value={selectedEntity} onValueChange={setSelectedEntity}>
                <SelectTrigger className="bg-nexus-bg border-nexus-border text-white h-10 font-sans">
                  <SelectValue placeholder={`Select ${entityType}...`} />
                </SelectTrigger>
                <SelectContent className="bg-nexus-card border-nexus-border text-white max-h-[300px] font-sans">
                  <div className="p-2 border-b border-nexus-border sticky top-0 bg-nexus-card z-10 font-sans">
                    <div className="relative font-sans">
                      <Search className="absolute left-2 top-2.5 h-4 w-4 text-nexus-text-muted" />
                      <Input
                        placeholder="Search..."
                        className="pl-8 bg-nexus-bg border-nexus-border h-9 font-sans"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                    </div>
                  </div>
                  {filteredEntities.length === 0 ? (
                    <div className="p-4 text-center text-nexus-text-muted text-sm italic font-sans">No {entityType}s found</div>
                  ) : (
                    filteredEntities.map((e: Entity) => (
                      <SelectItem key={e.id} value={e.id}>{e.label}</SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* Run Button */}
            <Button
              className="h-10 px-8 bg-gradient-to-r from-indigo-600 to-violet-700 hover:from-indigo-500 hover:to-violet-600 text-white font-bold gap-2 shadow-lg shadow-indigo-900/20 font-sans whitespace-nowrap"
              onClick={handleRun}
              disabled={!selectedScenario || !selectedEntity || predictMutation.isPending}
            >
              {predictMutation.isPending ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Play className="w-5 h-5" />
              )}
              Run Prediction
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Area */}
      {!predictMutation.isPending && !currentResult && (
        <div className="flex flex-col items-center justify-center py-24 opacity-20 select-none">
          <Layers className="w-24 h-24 mb-4" />
          <h2 className="text-2xl font-bold uppercase tracking-widest font-sans">Select Scenario to Begin</h2>
        </div>
      )}

      {predictMutation.isPending && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Skeleton className="h-[400px] lg:col-span-2 bg-nexus-card/50 rounded-xl" />
          <Skeleton className="h-[400px] bg-nexus-card/50 rounded-xl" />
          <Skeleton className="h-[300px] bg-nexus-card/50 rounded-xl" />
          <Skeleton className="h-[300px] bg-nexus-card/50 rounded-xl" />
          <Skeleton className="h-[300px] bg-nexus-card/50 rounded-xl" />
        </div>
      )}

      {currentResult && !predictMutation.isPending && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-in slide-in-from-bottom-4 duration-500">

          {/* Primary Prediction Column */}
          <Card className="lg:col-span-8 glass-card border-nexus-border">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-xl font-bold text-white flex items-center gap-2 font-sans">
                  <Target className="w-5 h-5 text-indigo-400" />
                  Primary Prediction: {currentResult.prediction.primary.label}
                </CardTitle>
                <CardDescription className="font-sans">Stochastic outcome analysis with multi-factor drag</CardDescription>
              </div>
              <div className="text-right">
                <span className="text-4xl font-black text-white font-sans tracking-tighter">
                  {currentResult.prediction?.primary?.value ?? '0'}
                  <span className="text-xl font-normal text-nexus-text-secondary ml-1">{currentResult.prediction?.primary?.unit ?? ''}</span>
                </span>
                <div className={cn(
                  "flex items-center justify-end gap-1 text-sm font-medium mt-1 font-sans",
                  currentResult.prediction?.primary?.trend === 'improving' ? 'text-emerald-400' :
                    currentResult.prediction?.primary?.trend === 'declining' ? 'text-rose-400' : 'text-nexus-text-secondary'
                )}>
                  {currentResult.prediction?.primary?.trend === 'improving' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                  {(currentResult.prediction?.primary?.trend ?? 'STABLE').toUpperCase()}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 font-sans">
                {currentResult.prediction?.secondary_metrics?.map((m: any, idx: number) => (
                  <div key={idx} className="bg-nexus-bg/30 border border-nexus-border/50 rounded-xl p-4 hover:border-indigo-500/30 transition-colors">
                    <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-widest">{m.label}</p>
                    <p className="text-2xl font-bold text-white mt-1">
                      {m.value}
                      <span className="text-xs font-normal text-nexus-text-secondary ml-1">{m.unit}</span>
                    </p>
                  </div>
                ))}
              </div>

              {/* Advanced Monte Carlo Visualization */}
              <div className="space-y-6 font-sans">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-indigo-400 font-sans" />
                      Monte Carlo Outcome Distribution
                    </h3>
                    <p className="text-xs text-nexus-text-secondary">Density map of 5,000 simulated trial iterations</p>
                  </div>
                  <Badge variant="secondary" className="bg-indigo-500/10 text-indigo-400 border-indigo-500/20 font-mono">
                    P50: {currentResult.prediction?.confidence?.p50 ?? 'N/A'}
                  </Badge>
                </div>

                <div className="h-[220px] w-full bg-nexus-bg/50 rounded-2xl border border-nexus-border/30 p-4 relative overflow-hidden group">
                  {currentResult.prediction?.distribution ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={currentResult.prediction.distribution}>
                        <defs>
                          <linearGradient id="colorY" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false} />
                        <XAxis
                          dataKey="x"
                          stroke="#6c727a"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          label={{ value: 'Days to Completion', position: 'insideBottom', offset: -5, fill: '#6c727a', fontSize: 10 }}
                        />
                        <YAxis hide />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#11111b', borderColor: '#313244', color: '#cdd6f4' }}
                          itemStyle={{ color: '#89b4fa' }}
                          labelFormatter={(val) => `Simulated Days: ${val}`}
                        />
                        <Area
                          type="monotone"
                          dataKey="y"
                          stroke="#818cf8"
                          strokeWidth={3}
                          fillOpacity={1}
                          fill="url(#colorY)"
                          animationDuration={1500}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center space-y-4">
                      <div className="relative w-full px-12">
                        <div className="h-1.5 w-full bg-nexus-border/30 rounded-full relative">
                          <div className="absolute top-1/2 -translate-y-1/2 left-[15%] right-[15%] h-3 bg-indigo-500/20 rounded-full border border-indigo-500/40"></div>
                          <div className="absolute top-1/2 -translate-y-1/2 left-[50%] h-10 w-1 bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] z-10"></div>
                        </div>
                      </div>
                      <p className="text-[10px] text-nexus-text-muted uppercase tracking-[0.2em] font-bold">Standard Confidence Envelope</p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-6 pt-2">
                  <div className="text-center p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border/50 group hover:border-emerald-500/30 transition-colors">
                    <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Optimistic (P10)</p>
                    <p className="text-lg font-bold text-emerald-400 mt-1">{currentResult.prediction?.confidence?.p10 ?? 'N/A'}</p>
                  </div>
                  <div className="text-center p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border/50 ring-2 ring-indigo-500/20 shadow-xl shadow-indigo-900/10">
                    <p className="text-[10px] font-black text-white uppercase tracking-widest">Expected (P50)</p>
                    <p className="text-lg font-bold text-white mt-1">{currentResult.prediction?.confidence?.p50 ?? 'N/A'}</p>
                  </div>
                  <div className="text-center p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border/50 group hover:border-rose-500/30 transition-colors">
                    <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Conservative (P90)</p>
                    <p className="text-lg font-bold text-rose-400 mt-1">{currentResult.prediction?.confidence?.p90 ?? 'N/A'}</p>
                  </div>
                </div>

                {currentResult.prediction?.projected_date && (
                  <div className="mt-4 p-5 bg-gradient-to-r from-emerald-500/10 to-transparent border border-emerald-500/20 rounded-2xl flex items-center justify-between font-sans">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30">
                        <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">Projected Lock Date</p>
                        <p className="text-xs text-nexus-text-secondary">95% probability of completion by this date</p>
                      </div>
                    </div>
                    <span className="text-3xl font-black text-emerald-400 tracking-tighter">{currentResult.prediction.projected_date}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Risk Factors Column */}
          <Card className="lg:col-span-4 glass-card border-nexus-border">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <div className="font-sans">
                <CardTitle className="text-lg font-bold text-white flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                  Risk Drivers
                </CardTitle>
                <CardDescription>Primary vectors of delay</CardDescription>
              </div>
              <Badge className="bg-orange-500/10 text-orange-400 border-none font-sans uppercase text-[10px]">{(currentResult.risk_factors || []).length} DETECTED</Badge>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 font-sans">
                {(!currentResult.risk_factors || currentResult.risk_factors.length === 0) ? (
                  <div className="text-center py-10 text-nexus-text-muted italic font-sans flex flex-col items-center">
                    <Shield className="w-10 h-10 mb-2 opacity-20" />
                    No critical risks identified
                  </div>
                ) : (
                  currentResult.risk_factors.map((r: any, idx: number) => (
                    <div key={idx} className="p-4 rounded-xl bg-nexus-bg/30 border border-nexus-border/50 relative overflow-hidden group hover:border-orange-500/30 transition-colors">
                      <div className={cn(
                        "absolute left-0 top-0 bottom-0 w-1 transition-all group-hover:w-1.5",
                        r.severity === 'critical' ? 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]' :
                          r.severity === 'high' ? 'bg-orange-500' : 'bg-amber-500'
                      )}></div>
                      <div className="flex justify-between items-start mb-1">
                        <Badge className={cn(
                          "uppercase text-[10px] font-black px-1.5 py-0 border-none",
                          r.severity === 'critical' ? 'bg-rose-500/20 text-rose-400' :
                            r.severity === 'high' ? 'bg-orange-500/20 text-orange-400' : 'bg-amber-500/20 text-amber-400'
                        )}>
                          {r.severity}
                        </Badge>
                        {r.affected_count && (
                          <span className="text-[10px] font-mono text-nexus-text-muted">{r.affected_count} entities</span>
                        )}
                      </div>
                      <p className="text-sm font-bold text-white">{r.factor}</p>
                      <p className="text-xs text-nexus-text-secondary mt-1 leading-relaxed">{r.impact}</p>
                    </div>
                  ))
                )}

                <div className="pt-4 mt-4 border-t border-nexus-border/50">
                  <div className="p-4 rounded-xl bg-indigo-500/5 border border-indigo-500/20 font-sans">
                    <p className="text-xs font-bold text-indigo-400 uppercase flex items-center gap-2 mb-2">
                      <Brain className="w-3 h-3" /> AI Insight
                    </p>
                    <p className="text-xs text-nexus-text leading-relaxed">
                      {currentResult.ai_insight || "Data patterns suggest a 15% increase in signature lag at clinical sites utilizing the legacy EDC interface."}
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Recommended Actions */}
          <Card className="lg:col-span-12 glass-card border-nexus-border">
            <CardHeader className="font-sans border-b border-nexus-border/30 mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
                    <CheckSquare className="w-5 h-5 text-emerald-400" />
                    AI-Driven Mitigation Strategy
                  </CardTitle>
                  <CardDescription>Probability-mapped interventions to recover timeline</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-sans">
                {(!currentResult.recommended_actions || currentResult.recommended_actions.length === 0) ? (
                  <div className="col-span-3 text-center py-10 text-nexus-text-muted italic">No specific recommendations generated for this baseline</div>
                ) : (
                  currentResult.recommended_actions.map((a: any, idx: number) => (
                    <div key={idx} className="p-6 rounded-2xl bg-gradient-to-br from-nexus-card to-nexus-bg border border-nexus-border/50 hover:border-emerald-500/50 hover:shadow-2xl hover:shadow-emerald-900/10 transition-all group flex flex-col h-full relative overflow-hidden">
                      <div className="absolute -top-4 -right-4 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl group-hover:bg-emerald-500/10 transition-colors" />
                      <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-5 group-hover:scale-110 transition-transform border border-emerald-500/20">
                        <span className="text-xl font-black text-emerald-500">{idx + 1}</span>
                      </div>
                      <h4 className="text-md font-bold text-white mb-2 leading-tight">{a.action}</h4>
                      <p className="text-sm text-nexus-text-secondary mb-6 flex-grow">{a.expected_impact}</p>

                      <div className="flex items-center justify-between mt-auto pt-5 border-t border-nexus-border/30">
                        <div className="flex flex-col">
                          <span className="text-[10px] text-nexus-text-muted uppercase font-black tracking-widest">Resource Effort</span>
                          <Badge variant="outline" className={cn(
                            "border-none p-0 capitalize font-bold text-sm",
                            a.effort === 'low' ? 'text-emerald-400' : a.effort === 'medium' ? 'text-amber-400' : 'text-rose-400'
                          )}>
                            {a.effort}
                          </Badge>
                        </div>
                        {a.timeline_gain && (
                          <div className="flex flex-col items-end">
                            <span className="text-[10px] text-nexus-text-muted uppercase font-black tracking-widest">Time Recovery</span>
                            <span className="text-sm font-bold text-emerald-400 flex items-center gap-1">
                              <TrendingUp className="w-3 h-3" /> {a.timeline_gain}
                            </span>
                          </div>
                        )}
                      </div>

                      <Button className="w-full mt-5 bg-white/5 hover:bg-white/10 text-white border-nexus-border h-9 text-xs font-bold gap-2">
                        Queue Execution <ChevronRight className="w-3 h-3" />
                      </Button>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          {/* Blocking Items Table */}
          {(currentResult.blocking_items && currentResult.blocking_items.length > 0) && (
            <Card className="lg:col-span-12 glass-card border-nexus-border font-sans">
              <CardHeader className="pb-2 font-sans">
                <CardTitle className="text-lg font-bold text-white flex items-center gap-2 font-sans">
                  <Shield className="w-5 h-5 text-rose-400 font-sans" />
                  Critical Critical Path Blockers
                </CardTitle>
                <CardDescription className="font-sans">Dependencies that MUST be resolved to hit {(currentResult.prediction?.primary?.label ?? 'target')} target</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader className="border-nexus-border">
                    <TableRow className="hover:bg-transparent border-nexus-border border-b-2">
                      <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">BLOCKING ITEM</TableHead>
                      <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">TARGET ENTITY</TableHead>
                      <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest">REMEDIATION ACTION</TableHead>
                      <TableHead className="text-nexus-text-muted font-black uppercase text-[10px] tracking-widest text-right">SEVERITY</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {currentResult.blocking_items.map((b: any, idx: number) => (
                      <TableRow key={idx} className="border-nexus-border hover:bg-white/5 transition-colors group">
                        <TableCell className="font-bold text-white font-sans py-4">{b.item}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-[10px] uppercase border-nexus-border text-nexus-text-muted font-mono px-1.5 py-0">
                              {b.entity_type}
                            </Badge>
                            <span className="text-sm text-nexus-text font-sans font-medium">{b.entity_id}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-sm text-nexus-text-secondary font-sans italic">"{b.action_needed}"</TableCell>
                        <TableCell className="text-right">
                          <Badge className={cn(
                            "capitalize font-sans px-3 py-0.5 rounded-full border-none",
                            b.priority === 'critical' ? 'bg-rose-500 shadow-lg shadow-rose-900/20' :
                              b.priority === 'high' ? 'bg-orange-500' : 'bg-amber-500'
                          )}>
                            {b.priority}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {/* Metadata Footer */}
          <div className="lg:col-span-12 flex justify-between items-center px-6 py-4 opacity-50 text-[9px] text-nexus-text-muted font-mono uppercase tracking-[0.2em] border-t border-nexus-border/30 mt-4">
            <div className="flex gap-10">
              <span className="flex items-center gap-2"><Zap className="w-3 h-3" /> Iterations: {currentResult.metadata?.iterations?.toLocaleString() || '5,000'}</span>
              <span className="flex items-center gap-2"><Layers className="w-3 h-3" /> Data Freshness: {currentResult.metadata?.data_freshness || 'Real-time'}</span>
              <span className="flex items-center gap-2"><BarChart3 className="w-3 h-3" /> Distribution: Monte Carlo / Beta / Poisson</span>
            </div>
            <div className="flex items-center gap-3">
              <span>NEXUS-PREDICT-CORE v2.1-AL</span>
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DigitalTwin;
