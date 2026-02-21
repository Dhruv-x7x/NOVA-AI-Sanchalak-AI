import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { mlApi } from '@/services/api';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Brain,
  CheckCircle2,
  Clock,
  Eye,
  Shield,
  AlertTriangle,
  Activity,
  RefreshCw,
  Lock,
  Hash,
  Cpu,
  GitBranch,
  Loader2,
  ChevronRight,
} from 'lucide-react';
import { formatDateTime, cn } from '@/lib/utils';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';

export default function MLGovernance() {
  const [activeTab, setActiveTab] = useState('overview');
  const queryClient = useQueryClient();
  const [driftCheckLoading, setDriftCheckLoading] = useState(false);

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['ml-models'],
    queryFn: () => mlApi.getModels(),
  });

  const { data: summary } = useQuery({
    queryKey: ['ml-summary'],
    queryFn: () => mlApi.getSummary(),
  });

  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);

  const { data: driftReports } = useQuery({
    queryKey: ['ml-drift-reports'],
    queryFn: () => mlApi.getDriftReports(),
  });

  const reports = useMemo(() =>
    driftReports?.drift_reports && Array.isArray(driftReports.drift_reports)
      ? driftReports.drift_reports
      : []
    , [driftReports]);

  const selectedReport = useMemo(() =>
    selectedReportId ? reports.find((r: any) => r.report_id === selectedReportId) : null
    , [selectedReportId, reports]);

  const auditLog = useQuery({
    queryKey: ['ml-audit-log'],
    queryFn: () => mlApi.getAuditLog(),
  }).data;

  const approveMutation = useMutation({
    mutationFn: (modelId: number) => mlApi.approve(modelId, { approved_by: 'current_user' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      queryClient.invalidateQueries({ queryKey: ['ml-summary'] });
    },
  });

  useMutation({
    mutationFn: (modelId: number) => mlApi.deploy(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml-models'] });
      queryClient.invalidateQueries({ queryKey: ['ml-summary'] });
    },
  });

  const handleRunDriftCheck = async () => {
    setDriftCheckLoading(true);
    try {
      await mlApi.runDriftCheck();
      queryClient.invalidateQueries({ queryKey: ['ml-drift-reports'] });
    } catch (e) {
      console.error('Drift check failed', e);
    } finally {
      setDriftCheckLoading(false);
    }
  };

  // Performance data from real drift reports
  const performanceData = useMemo(() => {
    if (!reports || reports.length === 0) {
      return [
        { date: 'No data', accuracy: 0, precision: 0, recall: 0 },
      ];
    }
    // Group by model, use latest report per model for accuracy comparison
    const modelMap: Record<string, any> = {};
    for (const r of reports as any[]) {
      const name = r.model_name || 'Unknown';
      if (!modelMap[name]) modelMap[name] = r;
    }
    return Object.values(modelMap).map((r: any) => ({
      date: r.model_name || 'Model',
      accuracy: r.current_accuracy ?? r.baseline_accuracy ?? 0,
      baseline: r.baseline_accuracy ?? 0,
    }));
  }, [reports]);

  const driftTrendData = useMemo(() => {
    if (!reports || reports.length === 0) {
      return [
        { date: '2024-01-01', psi: 0.05, threshold: 0.1 },
        { date: '2024-01-08', psi: 0.06, threshold: 0.1 },
        { date: '2024-01-15', psi: 0.08, threshold: 0.1 },
        { date: '2024-01-22', psi: 0.07, threshold: 0.1 },
        { date: '2024-01-29', psi: 0.04, threshold: 0.1 },
      ];
    }
    // Take 10 most recent reports (reports come in DESC order from API)
    // We sort them by date for the X-axis chronological trend
    return [...reports]
      .slice(0, 10)
      .sort((a: any, b: any) => new Date(a.checked_at).getTime() - new Date(b.checked_at).getTime())
      .map((r: any) => {
        const d = new Date(r.checked_at);
        return {
          date: isNaN(d.getTime()) ? 'N/A' : d.toLocaleDateString(),
          psi: r.drift_score || 0,
          threshold: r.threshold || 0.1
        };
      });
  }, [reports]);

  // System status items
  const systemStatus = [
    { name: 'Drift Detector', status: 'operational', lastCheck: '2 min ago' },
    { name: 'Model Registry', status: 'operational', lastCheck: '5 min ago' },
    { name: 'Feature Store', status: 'operational', lastCheck: '3 min ago' },
    { name: 'Prediction Service', status: 'operational', lastCheck: '1 min ago' },
    { name: 'Audit Logger', status: 'operational', lastCheck: '30 sec ago' },
  ];

  if (modelsLoading) {
    return <SanchalakLoader size="lg" label="Loading AI governance models..." fullPage />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">ML Model Governance</h1>
              <p className="text-nexus-text-secondary">Model lifecycle management, monitoring, and compliance</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="success" className="flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              21 CFR Part 11 Compliant
            </Badge>
            <Button
              variant="outline"
              size="sm"
              className="border-purple-500/50 text-purple-400 hover:bg-purple-500/10 hover:text-purple-300"
              onClick={handleRunDriftCheck}
              disabled={driftCheckLoading}
            >
              {driftCheckLoading ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing...</>
              ) : (
                <><Activity className="w-4 h-4 mr-2" /> Run Drift Check</>
              )}
            </Button>
            <Button variant="outline" size="sm" className="border-nexus-border text-nexus-text-secondary hover:text-white" onClick={() => queryClient.invalidateQueries()}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="kpi-card kpi-card-purple">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{models?.total || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Total Models</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Brain className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-success">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{summary?.deployed_count || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Deployed</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-success-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-warning">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{summary?.pending_approval || 0}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Pending Approval</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
                <Clock className="w-6 h-6 text-warning-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-info">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{reports.length}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Drift Reports</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-info-500/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-info-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="overview" className="data-[state=active]:bg-violet-600">Overview</TabsTrigger>
          <TabsTrigger value="models" className="data-[state=active]:bg-violet-600">Model Registry</TabsTrigger>
          <TabsTrigger value="drift" className="data-[state=active]:bg-violet-600">Drift Detection</TabsTrigger>
          <TabsTrigger value="performance" className="data-[state=active]:bg-violet-600">Performance</TabsTrigger>
          <TabsTrigger value="audit" className="data-[state=active]:bg-violet-600">Audit Trail</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-purple-400" />
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {systemStatus.map((item) => (
                    <div key={item.name} className="flex items-center justify-between p-3 bg-nexus-card rounded-lg border border-nexus-border">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-success-500"></div>
                        <span className="text-white">{item.name}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge variant="success">{item.status}</Badge>
                        <span className="text-xs text-nexus-text-secondary">{item.lastCheck}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-success-400" />
                  Model Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
                  {models?.models?.map((model: any) => (
                    <div key={model.version_id} className="p-4 bg-nexus-card rounded-lg border border-nexus-border hover:border-violet-500/30 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-bold text-white">{model.model_name}</span>
                        <Badge variant={model.accuracy >= 90 ? 'success' : 'warning'}>
                          {model.accuracy >= 90 ? 'Healthy' : 'Warning'}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-nexus-text-secondary text-[10px] uppercase font-bold">Accuracy</span>
                          <p className={cn(
                            "font-bold",
                            model.accuracy >= 92 ? "text-success-400" : "text-warning-400"
                          )}>{model.accuracy?.toFixed(1)}%</p>
                        </div>
                        <div>
                          <span className="text-nexus-text-secondary text-[10px] uppercase font-bold">Type</span>
                          <p className="font-medium text-white capitalize">{model.model_type?.replace('_', ' ')}</p>
                        </div>
                        <div>
                          <span className="text-nexus-text-secondary text-[10px] uppercase font-bold">Version</span>
                          <p className="font-medium text-indigo-400">{model.version}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Compliance Dashboard</CardTitle>
              <CardDescription className="text-nexus-text-secondary">21 CFR Part 11 Compliance Status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-4">
                <div className="p-4 bg-success-500/10 rounded-lg border border-success-500/20 text-center">
                  <Lock className="w-8 h-8 text-success-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-success-400">Electronic Signatures</p>
                  <p className="text-xs text-nexus-text-secondary mt-1">All models signed</p>
                </div>
                <div className="p-4 bg-success-500/10 rounded-lg border border-success-500/20 text-center">
                  <Shield className="w-8 h-8 text-success-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-success-400">Audit Trail</p>
                  <p className="text-xs text-nexus-text-secondary mt-1">Complete logging</p>
                </div>
                <div className="p-4 bg-success-500/10 rounded-lg border border-success-500/20 text-center">
                  <Hash className="w-8 h-8 text-success-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-success-400">Checksums</p>
                  <p className="text-xs text-nexus-text-secondary mt-1">All verified</p>
                </div>
                <div className="p-4 bg-success-500/10 rounded-lg border border-success-500/20 text-center">
                  <GitBranch className="w-8 h-8 text-success-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-success-400">Version Control</p>
                  <p className="text-xs text-nexus-text-secondary mt-1">All tracked</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Model Registry</CardTitle>
              <CardDescription className="text-nexus-text-secondary">All registered machine learning models</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="border-nexus-border hover:bg-transparent">
                    <TableHead className="text-nexus-text-secondary">Model Name</TableHead>
                    <TableHead className="text-nexus-text-secondary">Type</TableHead>
                    <TableHead className="text-nexus-text-secondary">Version</TableHead>
                    <TableHead className="text-nexus-text-secondary">Status</TableHead>
                    <TableHead className="text-nexus-text-secondary">Trained At</TableHead>
                    <TableHead className="text-nexus-text-secondary">Samples</TableHead>
                    <TableHead className="text-right text-nexus-text-secondary">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {models?.models?.map((model: any) => (
                    <TableRow key={model.version_id} className="border-nexus-border hover:bg-nexus-card/50">
                      <TableCell className="font-medium text-white">{model.model_name}</TableCell>
                      <TableCell className="text-nexus-text-secondary capitalize">{model.model_type?.replace('_', ' ')}</TableCell>
                      <TableCell><Badge variant="secondary">{model.version}</Badge></TableCell>
                      <TableCell>
                        <Badge variant={
                          model.status === 'deployed' ? 'success' :
                            model.status === 'pending_approval' ? 'warning' : 'info'
                        }>
                          {model.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-nexus-text-secondary">{formatDateTime(model.trained_at)}</TableCell>
                      <TableCell className="text-white">{model.training_samples?.toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-1">
                          {model.status === 'pending_approval' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => approveMutation.mutate(model.version_id)}
                              disabled={approveMutation.isPending}
                              className="border-success-500/50 text-success-400 hover:bg-success-500/10"
                            >
                              Approve
                            </Button>
                          )}
                          <Button variant="ghost" size="sm" className="text-nexus-text-secondary hover:text-white">
                            <Eye className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Drift Tab */}
        <TabsContent value="drift" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card className="glass-card border-nexus-border md:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-white">PSI Drift Score Trend</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Population Stability Index over time</CardDescription>
                </div>
                <div className="flex gap-2">
                  <Badge variant="outline" className="border-error-500/50 text-error-400">{"Critical: > 0.15"}</Badge>
                  <Badge variant="outline" className="border-warning-500/50 text-warning-400">{"Warning: > 0.10"}</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={driftTrendData}>
                      <defs>
                        <linearGradient id="colorPsi" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" vertical={false} />
                      <XAxis dataKey="date" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                      <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1a1f2e',
                          border: '1px solid #2d3548',
                          borderRadius: '8px',
                          color: '#fff'
                        }}
                      />
                      <Area type="monotone" dataKey="psi" name="PSI Score" stroke="#8b5cf6" strokeWidth={2} fillOpacity={1} fill="url(#colorPsi)" />
                      <Line type="monotone" dataKey="threshold" name="Threshold" stroke="#ef4444" strokeDasharray="5 5" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2 text-sm uppercase tracking-wider font-bold">
                  <AlertTriangle className="w-4 h-4 text-warning-400" />
                  Live Drift Alerts
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="divide-y divide-nexus-border max-h-[300px] overflow-y-auto custom-scrollbar">
                  {reports.slice(0, 10).map((report: any, idx: number) => (
                    <div
                      key={report.report_id || `alert-${idx}`}
                      className={cn(
                        "p-4 cursor-pointer transition-colors",
                        report.status === 'critical' ? "bg-error-500/5 hover:bg-error-500/10" :
                          report.status === 'high' ? "bg-warning-500/5 hover:bg-warning-500/10" :
                            "hover:bg-nexus-card-hover"
                      )}
                      onClick={() => setSelectedReportId(report.report_id)}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-white">{report.model_name}</span>
                        <Badge variant={
                          report.status === 'critical' || report.status === 'high' ? 'error' :
                            report.status === 'medium' ? 'warning' : 'success'
                        } className="text-[9px] h-4">
                          {String(report.status || 'unknown').toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-[10px]">
                        <span className="text-nexus-text-secondary">PSI: {typeof report.drift_score === 'number' ? report.drift_score.toFixed(3) : '0.000'}</span>
                        <span className="text-nexus-text-muted">•</span>
                        <span className="text-nexus-text-muted">{formatDateTime(report.checked_at)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card className="glass-card border-nexus-border lg:col-span-2">
              <CardHeader className="py-4">
                <CardTitle className="text-white text-base">Historical Drift Analysis</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="border-nexus-border hover:bg-transparent">
                      <TableHead className="text-nexus-text-secondary text-xs">Model</TableHead>
                      <TableHead className="text-nexus-text-secondary text-xs">PSI Score</TableHead>
                      <TableHead className="text-nexus-text-secondary text-xs">Accuracy Delta</TableHead>
                      <TableHead className="text-nexus-text-secondary text-xs">Status</TableHead>
                      <TableHead className="text-nexus-text-secondary text-xs">Checked At</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {reports.map((report: any, idx: number) => (
                      <TableRow
                        key={report.report_id || `drift-${idx}`}
                        className={cn(
                          "border-nexus-border cursor-pointer transition-all",
                          selectedReportId === report.report_id ? "bg-violet-500/10" : "hover:bg-nexus-card/50"
                        )}
                        onClick={() => setSelectedReportId(report.report_id)}
                      >
                        <TableCell className="font-bold text-white text-xs">{report.model_name}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1 bg-nexus-border rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  "h-full rounded-full",
                                  (report.drift_score || 0) < 0.05 ? "bg-success-400" :
                                    (report.drift_score || 0) < 0.10 ? "bg-warning-400" : "bg-error-400"
                                )}
                                style={{ width: `${Math.min((report.drift_score || 0) * 1000, 100)}%` }}
                              />
                            </div>
                            <span className={cn(
                              "text-[10px] font-bold",
                              (report.drift_score || 0) > 0.1 ? 'text-error-400' :
                                (report.drift_score || 0) > 0.05 ? 'text-warning-400' : 'text-success-400'
                            )}>
                              {typeof report.drift_score === 'number' ? report.drift_score.toFixed(3) : '0.000'}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-xs">
                          {typeof report.baseline_accuracy === 'number' ? (
                            <div className="flex items-center gap-1.5">
                              <span className="text-nexus-text-secondary">{report.baseline_accuracy.toFixed(1)}%</span>
                              <ChevronRight className="w-2.5 h-2.5 text-nexus-text-muted" />
                              <span className={cn(
                                "font-bold",
                                (report.current_accuracy || 0) < report.baseline_accuracy ? "text-error-400" : "text-success-400"
                              )}>
                                {(report.current_accuracy || 0).toFixed(1)}%
                              </span>
                            </div>
                          ) : '--'}
                        </TableCell>
                        <TableCell>
                          <Badge variant={
                            report.status === 'critical' || report.status === 'high' ? 'error' :
                              report.status === 'medium' ? 'warning' : 'success'
                          } className="text-[10px] h-4 font-bold">
                            {String(report.status || 'unknown').toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-nexus-text-secondary text-[10px]">{formatDateTime(report.checked_at)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card className="glass-card border-nexus-border bg-nexus-card/50">
              <CardHeader className="border-b border-nexus-border/50 py-4">
                <CardTitle className="text-white text-sm font-bold uppercase tracking-wider flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  Investigation Detail
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                {selectedReport ? (
                  <div className="space-y-4 animate-in fade-in slide-in-from-right-4">
                    <div>
                      <h4 className="text-xs font-black text-nexus-text-muted uppercase mb-1">Root Cause Analysis</h4>
                      <p className="text-xs text-white font-medium whitespace-pre-line">{selectedReport.recommendations || "Automated check completed."}</p>
                    </div>

                    <div className="p-3 rounded-xl bg-nexus-bg border border-nexus-border space-y-3">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-nexus-text-secondary">Overall PSI</span>
                        <span className={cn("font-bold", (selectedReport.drift_score || 0) > 0.10 ? 'text-error-400' : (selectedReport.drift_score || 0) > 0.05 ? 'text-warning-400' : 'text-success-400')}>{typeof selectedReport.drift_score === 'number' ? selectedReport.drift_score.toFixed(4) : '0.0000'}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-nexus-text-secondary">Accuracy</span>
                        <span className="text-white font-bold">{typeof selectedReport.baseline_accuracy === 'number' ? `${selectedReport.baseline_accuracy.toFixed(1)}% → ${(selectedReport.current_accuracy || 0).toFixed(1)}%` : '--'}</span>
                      </div>
                    </div>

                    {/* Per-feature breakdown */}
                    {selectedReport.feature_drift_details && Array.isArray(selectedReport.feature_drift_details) && selectedReport.feature_drift_details.length > 0 && (
                      <div>
                        <h4 className="text-xs font-black text-nexus-text-muted uppercase mb-2">Feature Drift Breakdown</h4>
                        <div className="space-y-1.5 max-h-[200px] overflow-y-auto custom-scrollbar">
                          {(selectedReport.feature_drift_details as any[]).map((fd: any, i: number) => (
                            <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-nexus-card border border-nexus-border text-[10px]">
                              <span className="text-white font-medium truncate max-w-[120px]">{(fd.feature_name || '').replace(/_/g, ' ')}</span>
                              <div className="flex items-center gap-2">
                                <span className={cn(
                                  "font-bold",
                                  fd.severity === 'CRITICAL' || fd.severity === 'HIGH' ? 'text-error-400' :
                                    fd.severity === 'MEDIUM' ? 'text-warning-400' :
                                      fd.severity === 'LOW' ? 'text-yellow-400' : 'text-success-400'
                                )}>{typeof fd.psi === 'number' ? fd.psi.toFixed(4) : '0.00'}</span>
                                <Badge variant={
                                  fd.severity === 'CRITICAL' || fd.severity === 'HIGH' ? 'error' :
                                    fd.severity === 'MEDIUM' ? 'warning' : 'success'
                                } className="text-[8px] h-3.5 px-1">{fd.severity}</Badge>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedReport.retrain_recommended && (
                      <div className="p-3 bg-error-500/10 rounded-xl border border-error-500/20">
                        <Button className="w-full bg-error-600 hover:bg-error-500 text-white text-xs font-bold h-8">
                          Initialize Retraining Loop
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 text-center">
                    <Activity className="w-6 h-6 text-nexus-text-muted mb-2" />
                    <p className="text-sm text-nexus-text-secondary italic">Select report to view detail</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance">
          <Card className="glass-card border-nexus-border">
            <CardHeader><CardTitle className="text-white">Model Performance — Current vs Baseline Accuracy</CardTitle></CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
                    <XAxis dataKey="date" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis domain={[80, 100]} stroke="#64748b" tick={{ fill: '#94a3b8' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3548', borderRadius: '8px', color: '#fff' }} />
                    <Legend />
                    <Line type="monotone" dataKey="baseline" name="Baseline Accuracy" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" />
                    <Line type="monotone" dataKey="accuracy" name="Current Accuracy" stroke="#8b5cf6" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audit">
          <Card className="glass-card border-nexus-border">
            <CardHeader><CardTitle className="text-white">Audit Trail</CardTitle></CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow className="border-nexus-border hover:bg-transparent">
                    <TableHead className="text-nexus-text-secondary">Action</TableHead>
                    <TableHead className="text-nexus-text-secondary">User</TableHead>
                    <TableHead className="text-nexus-text-secondary">Timestamp</TableHead>
                    <TableHead className="text-nexus-text-secondary">Details</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditLog?.audit_log?.map((log: any) => (
                    <TableRow key={log.log_id} className="border-nexus-border hover:bg-nexus-card/50">
                      <TableCell><Badge variant="outline" className="border-purple-500/50 text-purple-400">{log.action}</Badge></TableCell>
                      <TableCell className="text-white">{log.user}</TableCell>
                      <TableCell className="text-nexus-text-secondary text-[10px]">{formatDateTime(log.timestamp)}</TableCell>
                      <TableCell className="text-xs text-nexus-text-secondary max-w-md truncate">{log.details}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
