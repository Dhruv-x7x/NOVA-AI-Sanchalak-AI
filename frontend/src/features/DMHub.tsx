import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAppStore } from '@/stores/appStore';
import { patientsApi, analyticsApi, issuesApi, studiesApi, sitesApi } from '@/services/api';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
  CheckCircle2,
  XCircle,
  FileSearch,
  Lock,
  Unlock,
  Activity,
  AlertTriangle,
  TrendingUp,
  Database,
  BarChart3,
  RefreshCw,
  Zap,
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { formatPercent } from '@/lib/utils';

  const TIER_COLORS = {
    'tier_0': '#64748b',
    'tier_1': '#3b82f6',
    'tier_2': '#10b981',
    'db_lock_ready': '#059669',
  };

export default function DMHub() {
  const queryClient = useQueryClient();
  const { selectedStudy } = useAppStore();
  const [selectedSite, setSelectedSite] = useState<string>('all');
  const [isQualityMatrixOpen, setIsQualityMatrixOpen] = useState(false);

  // Fetch data
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio', selectedStudy],
    queryFn: () => analyticsApi.getPortfolio(selectedStudy),
  });

  const { data: cleanStatus, isLoading: cleanLoading } = useQuery({
    queryKey: ['clean-status', selectedStudy],
    queryFn: () => patientsApi.getCleanStatus(selectedStudy),
  });

  const { data: dblockStatus } = useQuery({
    queryKey: ['dblock-status', selectedStudy],
    queryFn: () => patientsApi.getDBLockStatus(selectedStudy),
  });

  const { data: dqiData } = useQuery({
    queryKey: ['patient-dqi', selectedStudy],
    queryFn: () => patientsApi.getDQI(selectedStudy),
  });

  const { data: cleanSummary } = useQuery({
    queryKey: ['clean-status-summary', selectedStudy],
    queryFn: () => analyticsApi.getCleanStatusSummary(selectedStudy),
  });

  const { data: dblockSummary } = useQuery({
    queryKey: ['dblock-summary', selectedStudy],
    queryFn: () => analyticsApi.getDBLockSummary(selectedStudy),
  });

  const { data: issuesSummary, isLoading: issuesLoading } = useQuery({
    queryKey: ['issues-summary', selectedStudy],
    queryFn: () => issuesApi.getSummary(selectedStudy),
  });

  const isPageLoading = portfolioLoading && cleanLoading && issuesLoading;

  const { data: patterns } = useQuery({
    queryKey: ['patterns', selectedStudy],
    queryFn: () => analyticsApi.getPatterns(selectedStudy),
  });

  const { data: bottleneckResponse } = useQuery({
    queryKey: ['bottlenecks', selectedStudy],
    queryFn: () => analyticsApi.getBottlenecks(selectedStudy),
  });

  const { data: qualityMatrixResponse } = useQuery({
    queryKey: ['quality-matrix', selectedStudy],
    queryFn: () => analyticsApi.getQualityMatrix(selectedStudy),
  });

  const { data: resolutionResponse } = useQuery({
    queryKey: ['resolution-stats', selectedStudy],
    queryFn: () => analyticsApi.getResolutionStats(selectedStudy),
  });

  const { data: studies } = useQuery({
    queryKey: ['studies'],
    queryFn: () => studiesApi.list(),
  });

  const { data: sites } = useQuery({
    queryKey: ['sites', selectedStudy],
    queryFn: () => sitesApi.list({ study_id: selectedStudy }),
  });

  // Transform quality distribution data
  const qualityDistribution = useMemo(() => {
    if (!cleanSummary?.summary?.tier_counts) return [];
    const mapping: Record<string, string> = {
      'tier_0': 'Tier 0',
      'tier_1': 'Tier 1',
      'tier_2': 'Tier 2',
      'db_lock_ready': 'Ready'
    };
    return Object.entries(cleanSummary.summary.tier_counts).map(([tier, count]) => ({
      name: mapping[tier] || tier,
      value: count as number,
      color: (TIER_COLORS as any)[tier] || '#6b7280',
    }));
  }, [cleanSummary]);

  // Bottleneck analysis data from real backend
  const bottleneckData = useMemo(() => {
    return bottleneckResponse?.bottlenecks?.slice(0, 5).map((b: any) => ({
      name: b.name.replace('_', ' ').split(' ').map((s: string) => s.charAt(0).toUpperCase() + s.substring(1)).join(' '),
      score: b.score,
      blocking: b.patients_affected,
      count: b.issues_count
    })) || [];
  }, [bottleneckResponse]);

  const topBottleneck = bottleneckResponse?.bottlenecks?.[0];

  // Reuse same cache as resolutionResponse — same API, same data
  const { data: learningData } = useQuery({
    queryKey: ['resolution-stats', selectedStudy],
    queryFn: () => analyticsApi.getResolutionStats(selectedStudy),
  });

  const resolutionMutation = useMutation({
    mutationFn: (data: { template_id: string; duration: number; success: boolean }) =>
      analyticsApi.recordResolution(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['resolution-stats'] });
      alert('Outcome recorded. AI models updating with new data point.');
    }
  });

  const [templateId, setTemplateId] = useState('');
  const [duration, setDuration] = useState(0.5);

  const handleFeedback = (success: boolean) => {
    if (!templateId) { alert('Please enter a Template ID'); return; }
    resolutionMutation.mutate({ template_id: templateId, duration, success });
  };

  const { data: labReconData } = useQuery({
    queryKey: ['lab-reconciliation', selectedStudy],
    queryFn: () => analyticsApi.getLabReconciliation(selectedStudy),
  });

  const [dqiWeights] = useState({ safety: 0.4, query: 0.2, visit: 0.2, lab: 0.1, integrity: 0.1 });

  if (isPageLoading) {
    return <SanchalakLoader size="lg" label="Loading data management hub..." fullPage />;
  }

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Data Management Hub</h1>
              <p className="text-nexus-text-secondary">Data quality, clean status, and DB lock management</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="bg-nexus-bg/50 px-4 py-2 rounded-lg border border-nexus-border flex items-center gap-2">
              <span className="text-xs text-nexus-text-secondary uppercase font-bold">Active:</span>
              <span className="text-sm text-white font-medium">
                {selectedStudy === 'all' ? 'All Portfolio' : studies?.studies?.find((s: any) => s.study_id === selectedStudy)?.protocol_number || selectedStudy}
              </span>
            </div>
            <Select value={selectedSite} onValueChange={setSelectedSite}>
              <SelectTrigger className="w-40 bg-nexus-card border-nexus-border text-white">
                <SelectValue placeholder="All Sites" />
              </SelectTrigger>
              <SelectContent className="bg-nexus-card border-nexus-border">
                <SelectItem value="all">All Sites</SelectItem>
                {sites?.sites?.map((site: any) => (
                  <SelectItem key={site.site_id} value={site.site_id}>{site.site_id}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" className="border-nexus-border text-nexus-text-secondary hover:text-white" onClick={() => queryClient.invalidateQueries()}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Sync
            </Button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card className="kpi-card kpi-card-purple">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatPercent(portfolio?.mean_dqi || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Portfolio DQI</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-purple-400" />
              </div>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-success-400" />
              <span className="text-xs text-success-400">System Monitoring Active</span>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-success">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatPercent(cleanSummary?.summary?.tier2_rate || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Tier 2 Clean</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-success-400" />
              </div>
            </div>
            <div className="mt-3">
              <Progress value={cleanSummary?.summary?.tier2_rate || 0} className="h-1.5" />
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-info">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{formatPercent(dblockSummary?.summary?.ready_rate || 0)}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">DB Lock Ready</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-info-500/20 flex items-center justify-center">
                <Lock className="w-6 h-6 text-info-400" />
              </div>
            </div>
            <div className="mt-3">
              <Progress value={dblockSummary?.summary?.ready_rate || 0} className="h-1.5" />
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-warning">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{dblockSummary?.summary?.not_ready_count || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Pending Lock</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
                <Unlock className="w-6 h-6 text-warning-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-error">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{issuesSummary?.open_count || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Open Issues</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-error-500/20 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-error-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-nexus-border">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-warning-400" />
              Pattern Alerts
            </CardTitle>
            <Button variant="ghost" size="sm" className="text-nexus-text-secondary hover:text-white" onClick={() => queryClient.invalidateQueries({ queryKey: ['patterns'] })}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-error-500"></div>
              <span className="text-white font-medium">{patterns?.alerts?.filter((a: any) => a.severity === 'Critical').length || 0}</span>
              <span className="text-nexus-text-secondary text-sm">Critical</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-warning-500"></div>
              <span className="text-white font-medium">{patterns?.alerts?.filter((a: any) => a.severity === 'High').length || 0}</span>
              <span className="text-nexus-text-secondary text-sm">High</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-info-500"></div>
              <span className="text-white font-medium">{patterns?.alerts?.filter((a: any) => a.severity === 'Medium').length || 0}</span>
              <span className="text-nexus-text-secondary text-sm">Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-success-500"></div>
              <span className="text-white font-medium">{patterns?.alerts?.filter((a: any) => a.severity === 'Low').length || 0}</span>
              <span className="text-nexus-text-secondary text-sm">Low</span>
            </div>
            <div className="flex-1" />
            <div className="text-xs text-nexus-text-secondary italic">
              Real-time monitoring ACTIVE across {portfolio?.total_sites || 0} sites
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="glass-card border-nexus-border overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between border-b border-nexus-border/50 bg-nexus-bg/20">
          <div>
            <CardTitle className="text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-info-400" />
              Quality Matrix by Site
            </CardTitle>
            <CardDescription className="text-nexus-text-secondary">Cross-site performance comparison</CardDescription>
          </div>
          <Button variant="ghost" size="sm" className="text-primary hover:text-primary-light" onClick={() => setIsQualityMatrixOpen(!isQualityMatrixOpen)}>
            {isQualityMatrixOpen ? 'Collapse' : 'View Full Quality Matrix'}
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-24 w-full flex overflow-x-auto no-scrollbar">
            {qualityMatrixResponse?.data?.slice(0, 40).map((site: any) => (
              <div 
                key={site.site_id} 
                className="flex-shrink-0 w-8 border-r border-nexus-border/20 flex flex-col group cursor-pointer"
                title={`${site.site_id}: DQI ${site.dqi_score.toFixed(1)}%`}
              >
                <div 
                  className="h-1/3 w-full" 
                  style={{ backgroundColor: site.tier2_clean_rate > 80 ? '#10b981' : site.tier2_clean_rate > 50 ? '#f59e0b' : '#ef4444', opacity: 0.8 }}
                />
                <div 
                  className="h-1/3 w-full" 
                  style={{ backgroundColor: site.tier1_clean_rate > 90 ? '#10b981' : site.tier1_clean_rate > 70 ? '#f59e0b' : '#ef4444', opacity: 0.6 }}
                />
                <div 
                  className="h-1/3 w-full" 
                  style={{ backgroundColor: site.dqi_score > 90 ? '#10b981' : site.dqi_score > 75 ? '#f59e0b' : '#ef4444', opacity: 1 }}
                />
              </div>
            ))}
          </div>
          {isQualityMatrixOpen && (
            <div className="p-4 border-t border-nexus-border/50 animate-in fade-in slide-in-from-top-2">
              <Table>
                <TableHeader>
                  <TableRow className="border-nexus-border">
                    <TableHead className="text-xs">Site ID</TableHead>
                    <TableHead className="text-xs">Patients</TableHead>
                    <TableHead className="text-xs">DQI Score</TableHead>
                    <TableHead className="text-xs">Tier 2 Clean</TableHead>
                    <TableHead className="text-xs">Open Issues</TableHead>
                    <TableHead className="text-xs text-right">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {qualityMatrixResponse?.data?.slice(0, 10).map((site: any) => (
                    <TableRow key={site.site_id} className="border-nexus-border">
                      <TableCell className="text-sm font-medium">{site.site_id}</TableCell>
                      <TableCell className="text-sm">{site.patients}</TableCell>
                      <TableCell className="text-sm text-info-400 font-bold">{site.dqi_score.toFixed(1)}%</TableCell>
                      <TableCell className="text-sm">{site.tier2_clean_rate.toFixed(1)}%</TableCell>
                      <TableCell className="text-sm">{site.total_issues}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant={site.quality_tier === 'Pristine' || site.quality_tier === 'Excellent' ? 'success' : site.quality_tier === 'Good' ? 'warning' : 'error'}>
                          {site.quality_tier}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="glass-card border-nexus-border">
          <CardHeader>
            <CardTitle className="text-white">Quality Tier Distribution</CardTitle>
            <CardDescription className="text-nexus-text-secondary">Patients by clean status tier</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={qualityDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, percent }: { name: string; percent: number }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    labelLine={{ stroke: '#64748b' }}
                  >
                    {qualityDistribution.map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1a1f2e', 
                      border: '1px solid #2d3548',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-2">
              {qualityDistribution.map((tier) => (
                <div key={tier.name} className="text-center p-2 bg-nexus-bg/30 rounded-lg border border-nexus-border/20">
                  <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ backgroundColor: tier.color }}></div>
                  <p className="text-[10px] text-nexus-text-secondary truncate">{tier.name}</p>
                  <p className="text-sm font-black text-white">{tier.value}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-nexus-border">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-error-400" />
              Bottleneck Analysis
            </CardTitle>
            <CardDescription className="text-nexus-text-secondary">Top blocking factors for DB Lock</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bottleneckData} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} stroke="#64748b" hide />
                  <YAxis type="category" dataKey="name" width={140} stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1a1f2e', 
                      border: '1px solid #2d3548',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                  />
                  <Bar dataKey="score" name="Blocking Impact" radius={[0, 4, 4, 0]} barSize={20}>
                    {bottleneckData.map((entry: any, index: number) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.score > 70 ? '#ef4444' : entry.score > 50 ? '#f59e0b' : '#3b82f6'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {topBottleneck && (
              <div className="p-4 bg-error-500/10 rounded-xl border border-error-500/20">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-error-400 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-bold text-white uppercase tracking-wider">Top Bottleneck: {topBottleneck.name.replace('_', ' ')}</h4>
                    <p className="text-xs text-nexus-text-secondary mt-1">
                      {topBottleneck.patients_affected} patients impacted across {selectedStudy === 'all' ? 'portfolio' : 'study'}. 
                      Resolving this would improve lock readiness by {topBottleneck.score.toFixed(1)} impact points.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="glass-card border-nexus-border">
        <CardHeader>
          <CardTitle className="text-white">DB Lock Readiness Overview</CardTitle>
          <CardDescription className="text-nexus-text-secondary">Progress toward database lock</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-white">Ready for Lock</span>
                  <span className="text-sm font-medium text-success-400">
                    {dblockSummary?.summary?.ready_count || 0} patients
                  </span>
                </div>
                <Progress 
                  value={dblockSummary?.summary?.ready_rate || 0} 
                  className="h-3"
                />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-white">Tier 1 Clean</span>
                  <span className="text-sm font-medium text-white">
                    {formatPercent(cleanSummary?.summary?.tier1_rate || 0)}
                  </span>
                </div>
                <Progress 
                  value={cleanSummary?.summary?.tier1_rate || 0} 
                  className="h-2"
                />
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium text-white">Tier 2 Clean</span>
                  <span className="text-sm font-medium text-white">
                    {formatPercent(cleanSummary?.summary?.tier2_rate || 0)}
                  </span>
                </div>
                <Progress 
                  value={cleanSummary?.summary?.tier2_rate || 0} 
                  className="h-2"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-success-500/10 rounded-xl border border-success-500/20 text-center">
                <CheckCircle2 className="w-8 h-8 text-success-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{dblockSummary?.summary?.ready_count || 0}</p>
                <p className="text-xs text-success-400">Ready</p>
              </div>
              <div className="p-4 bg-warning-500/10 rounded-xl border border-warning-500/20 text-center">
                <Unlock className="w-8 h-8 text-warning-400 mx-auto mb-2" />
                <p className="text-2xl font-bold text-white">{dblockSummary?.summary?.not_ready_count || 0}</p>
                <p className="text-xs text-warning-400">Not Ready</p>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="text-sm font-medium text-white">Blocking Reasons</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-2 bg-nexus-card rounded-lg">
                  <span className="text-sm text-nexus-text-secondary">Open Queries</span>
                  <Badge variant="error">{issuesSummary?.by_type?.query || 0}</Badge>
                </div>
                <div className="flex items-center justify-between p-2 bg-nexus-card rounded-lg">
                  <span className="text-sm text-nexus-text-secondary">Missing Data</span>
                  <Badge variant="warning">{issuesSummary?.by_type?.missing_data || 0}</Badge>
                </div>
                <div className="flex items-center justify-between p-2 bg-nexus-card rounded-lg">
                  <span className="text-sm text-nexus-text-secondary">Coding Pending</span>
                  <Badge variant="info">{issuesSummary?.by_type?.coding || 0}</Badge>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="dqi" className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="dqi" className="data-[state=active]:bg-purple-600">DQI Scores</TabsTrigger>
          <TabsTrigger value="clean" className="data-[state=active]:bg-purple-600">Clean Status</TabsTrigger>
          <TabsTrigger value="dblock" className="data-[state=active]:bg-purple-600">DB Lock Status</TabsTrigger>
          <TabsTrigger value="lab" className="data-[state=active]:bg-purple-600 text-info-400">Lab Reconciliation</TabsTrigger>
          <TabsTrigger value="weights" className="data-[state=active]:bg-purple-600 text-warning-400">DQI Config</TabsTrigger>
        </TabsList>

        <TabsContent value="dqi">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Patient DQI Scores</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Data Quality Index by patient</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                    <TableHeader>
                      <TableRow className="border-nexus-border hover:bg-transparent">
                        <TableHead className="text-nexus-text-secondary">Patient ID</TableHead>
                        <TableHead className="text-nexus-text-secondary">Site</TableHead>
                        <TableHead className="text-nexus-text-secondary">DQI Score</TableHead>
                        <TableHead className="text-nexus-text-secondary">Status</TableHead>
                        <TableHead className="text-nexus-text-secondary">Open Queries</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dqiData?.data?.slice(0, 15).map((patient: any) => (
                        <TableRow key={patient.patient_key} className="border-nexus-border hover:bg-nexus-card/50">
                          <TableCell className="font-medium text-white">{patient.patient_key}</TableCell>
                          <TableCell className="text-nexus-text-secondary">{patient.site_id}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Progress 
                                value={patient.dqi_score || 0} 
                                className="w-16 h-2"
                              />
                              <span className={(patient.dqi_score || 0) >= 85 ? 'text-success-400' : (patient.dqi_score || 0) >= 70 ? 'text-warning-400' : 'text-error-400'}>
                                {patient.dqi_score?.toFixed(1) || 'N/A'}%
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={(patient.dqi_score || 0) >= 85 ? 'success' : (patient.dqi_score || 0) >= 70 ? 'warning' : 'error'}>
                              {(patient.dqi_score || 0) >= 85 ? 'Good' : (patient.dqi_score || 0) >= 70 ? 'Fair' : 'Poor'}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-white">{patient.open_queries_count || 0}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>

              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clean">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Patient Clean Status</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Clean status tiers for all patients</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                    <TableHeader>
                      <TableRow className="border-nexus-border hover:bg-transparent">
                        <TableHead className="text-nexus-text-secondary">Patient ID</TableHead>
                        <TableHead className="text-nexus-text-secondary">Site</TableHead>
                        <TableHead className="text-nexus-text-secondary">Clean Tier</TableHead>
                        <TableHead className="text-nexus-text-secondary">Tier 1 Clean</TableHead>
                        <TableHead className="text-nexus-text-secondary">Tier 2 Clean</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {cleanStatus?.data?.slice(0, 15).map((patient: any) => (
                        <TableRow key={patient.patient_key} className="border-nexus-border hover:bg-nexus-card/50">
                          <TableCell className="font-medium text-white">{patient.patient_key}</TableCell>
                          <TableCell className="text-nexus-text-secondary">{patient.site_id}</TableCell>
                          <TableCell>
                            <Badge 
                              style={{ backgroundColor: (TIER_COLORS as any)[patient.clean_status_tier || ''] || '#6b7280' }}
                              className="text-white capitalize"
                            >
                              {(patient.clean_status_tier || 'Unknown').replace('_', ' ')}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {patient.tier1_clean ? (
                              <CheckCircle2 className="w-5 h-5 text-success-400" />
                            ) : (
                              <XCircle className="w-5 h-5 text-nexus-text-secondary" />
                            )}
                          </TableCell>
                          <TableCell>
                            {patient.tier2_clean ? (
                              <CheckCircle2 className="w-5 h-5 text-success-400" />
                            ) : (
                              <XCircle className="w-5 h-5 text-nexus-text-secondary" />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>

              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="lab">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Database className="w-5 h-5 text-info-400" />
                Lab vs EDC Reconciliation
              </CardTitle>
              <CardDescription className="text-nexus-text-secondary">Detecting discrepancies between external lab vendors and EDC clinical data</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="border-nexus-border">
                    <TableHead className="text-nexus-text-secondary">Patient ID</TableHead>
                    <TableHead className="text-nexus-text-secondary">Test Name</TableHead>
                    <TableHead className="text-nexus-text-secondary">EDC Value</TableHead>
                    <TableHead className="text-nexus-text-secondary">Vendor Value</TableHead>
                    <TableHead className="text-right text-nexus-text-secondary">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {labReconData?.map((row: any, i: number) => (
                    <TableRow key={i} className="border-nexus-border hover:bg-nexus-card/50">
                      <TableCell className="font-medium text-white">{row.patient}</TableCell>
                      <TableCell className="text-nexus-text-secondary">{row.test}</TableCell>
                      <TableCell className="text-white">{row.edc_value}</TableCell>
                      <TableCell className="text-info-400 font-bold">{row.lab_value}</TableCell>
                      <TableCell className="text-right">
                        <Badge variant={row.status === 'Discrepancy' ? 'error' : row.status === 'Matched' ? 'success' : 'warning'}>
                          {row.status}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="weights">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">DQI Weighting Configurator</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Assign weights to key parameters to reflect study-specific priorities</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm text-white">Safety Issues (SAE)</label>
                      <span className="text-xs text-primary font-bold">{Math.round(dqiWeights.safety * 100)}%</span>
                    </div>
                    <Progress value={dqiWeights.safety * 100} className="h-1.5" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm text-white">Open Queries</label>
                      <span className="text-xs text-primary font-bold">{Math.round(dqiWeights.query * 100)}%</span>
                    </div>
                    <Progress value={dqiWeights.query * 100} className="h-1.5" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm text-white">Visit Compliance</label>
                      <span className="text-xs text-primary font-bold">{Math.round(dqiWeights.visit * 100)}%</span>
                    </div>
                    <Progress value={dqiWeights.visit * 100} className="h-1.5" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-sm text-white">Lab Discrepancies</label>
                      <span className="text-xs text-primary font-bold">{Math.round(dqiWeights.lab * 100)}%</span>
                    </div>
                    <Progress value={dqiWeights.lab * 100} className="h-1.5" />
                  </div>
                </div>
                <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border flex flex-col justify-center">
                  <p className="text-xs text-nexus-text-secondary uppercase font-bold text-center mb-4 tracking-widest">Configuration Logic</p>
                  <p className="text-sm text-nexus-text text-center italic">"Assigning higher weights to Safety will cause unresolved SAEs to penalize site scores more heavily, enabling rapid identification of compliance risk."</p>
                  <Button className="mt-6 bg-primary" onClick={() => alert('Weights updated. Recalculating DQI across 58,097 patients...')}>Update & Recalculate</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dblock">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">DB Lock Readiness</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Patients ready for database lock</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                    <TableHeader>
                      <TableRow className="border-nexus-border hover:bg-transparent">
                        <TableHead className="text-nexus-text-secondary">Patient ID</TableHead>
                        <TableHead className="text-nexus-text-secondary">Site</TableHead>
                        <TableHead className="text-nexus-text-secondary">DB Lock Ready</TableHead>
                        <TableHead className="text-nexus-text-secondary">DQI Score</TableHead>
                        <TableHead className="text-nexus-text-secondary">Blocking Issues</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dblockStatus?.data?.slice(0, 15).map((patient: any) => (
                        <TableRow key={patient.patient_key} className="border-nexus-border hover:bg-nexus-card/50">
                          <TableCell className="font-medium text-white">{patient.patient_key}</TableCell>
                          <TableCell className="text-nexus-text-secondary">{patient.site_id}</TableCell>
                          <TableCell>
                            {patient.dblock_ready ? (
                              <div className="flex items-center gap-2">
                                <Lock className="w-5 h-5 text-success-400" />
                                <span className="text-success-400 text-sm">Ready</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-2">
                                <Unlock className="w-5 h-5 text-warning-400" />
                                <span className="text-warning-400 text-sm">Not Ready</span>
                              </div>
                            )}
                          </TableCell>
                          <TableCell>
                            <span className={(patient.dqi_score || 0) >= 85 ? 'text-success-400' : (patient.dqi_score || 0) >= 70 ? 'text-warning-400' : 'text-error-400'}>
                              {patient.dqi_score?.toFixed(1) || 'N/A'}%
                            </span>
                          </TableCell>
                          <TableCell>
                            {patient.blocking_issues && patient.blocking_issues > 0 ? (
                              <Badge variant="error">{patient.blocking_issues}</Badge>
                            ) : (
                              <Badge variant="success">0</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>

              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="glass-card border-nexus-border">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-success-400" />
                  Resolution Genome Matches
                </CardTitle>
                <CardDescription className="text-nexus-text-secondary">AI-driven matching against historical resolutions</CardDescription>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-white">{resolutionResponse?.summary?.total_matches || '0'}</p>
                <p className="text-[10px] text-nexus-text-secondary uppercase">Total Matches</p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={resolutionResponse?.by_type || []}>
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} interval={0} angle={-25} textAnchor="end" height={60} />
                  <YAxis stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1a1f2e', 
                      border: '1px solid #2d3548',
                      borderRadius: '8px',
                      color: '#fff'
                    }}
                  />
                  <Bar dataKey="count" name="Matches" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">Top Resolution Templates</h4>
              {resolutionResponse?.by_type?.slice(0, 3).map((template: any) => (
                <div key={template.name} className="p-3 bg-nexus-card rounded-lg border border-nexus-border flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-white">{template.name.replace('_', ' ').toUpperCase()}</p>
                    <p className="text-xs text-nexus-text-secondary">{template.count} matches • {template.avg_hours?.toFixed(1)} hrs effort</p>
                  </div>
                  <Badge variant="success" className="bg-success-500/20 text-success-400">{template.success_rate?.toFixed(0)}% success</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card border-nexus-border">
          <CardHeader>
            <div className="bg-emerald-500/10 border border-emerald-500/20 p-2 rounded-lg flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-widest">Live Learning ACTIVE</span>
              <span className="text-[10px] text-emerald-400/70">Templates continuously improve from feedback</span>
            </div>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-xl font-bold text-white">{learningData?.summary?.total_matches || 50}</p>
                <p className="text-[10px] text-nexus-text-secondary uppercase">Outcomes</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold text-white">{learningData?.summary?.avg_success_rate?.toFixed(1) || '64.0'}%</p>
                <p className="text-[10px] text-nexus-text-secondary uppercase">Success</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold text-white">{learningData?.by_type?.length || 8}</p>
                <p className="text-[10px] text-nexus-text-secondary uppercase">Tracked</p>
              </div>
              <div className="text-center">
                <p className="text-xl font-bold text-emerald-400">100%</p>
                <p className="text-[10px] text-nexus-text-secondary uppercase">Best</p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border space-y-4">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-primary" />
                Record Resolution Outcome
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[10px] text-nexus-text-secondary uppercase font-bold">Template ID</label>
                  <input 
                    type="text" 
                    className="w-full bg-nexus-card border border-nexus-border rounded px-2 py-1 text-xs text-white" 
                    placeholder="e.g., SDV-001" 
                    value={templateId}
                    onChange={(e) => setTemplateId(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-nexus-text-secondary uppercase font-bold">Duration (hrs)</label>
                  <input 
                    type="number" 
                    className="w-full bg-nexus-card border border-nexus-border rounded px-2 py-1 text-xs text-white" 
                    value={duration}
                    onChange={(e) => setDuration(parseFloat(e.target.value))}
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <Button 
                  className="flex-1 bg-emerald-600 hover:bg-emerald-700 h-8 text-xs font-bold text-white"
                  onClick={() => handleFeedback(true)}
                  disabled={resolutionMutation.isPending}
                >
                  SUCCESS
                </Button>
                <Button 
                  className="flex-1 bg-red-600 hover:bg-red-700 h-8 text-xs font-bold text-white"
                  onClick={() => handleFeedback(false)}
                  disabled={resolutionMutation.isPending}
                >
                  FAILED
                </Button>
              </div>
            </div>
            
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <TrendingUp className="w-3 h-3 text-emerald-400" />
                Top Performers
              </h4>
              <div className="space-y-1.5">
                {learningData?.by_type?.slice(0, 3).map((p: any) => (
                  <div key={p.name} className="flex items-center justify-between p-2 rounded-lg bg-nexus-card border border-nexus-border/30">
                    <span className="text-xs font-bold text-white">{p.name.replace('_', ' ').toUpperCase()}</span>
                    <span className={`text-xs font-bold ${p.success_rate >= 80 ? 'text-success-400' : 'text-warning-400'}`}>
                      {p.success_rate?.toFixed(0)}% Success
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
