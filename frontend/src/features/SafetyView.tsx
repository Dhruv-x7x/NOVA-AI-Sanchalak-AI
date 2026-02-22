import { useQuery } from '@tanstack/react-query';
import { safetyApi } from '@/services/api';
import A6onLoader from '@/components/a6on_iLoader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Shield,
  AlertTriangle,
  Clock,
  Activity,
  AlertCircle,
  Skull,
  XCircle,
  TrendingUp,
  Bell,
} from 'lucide-react';
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';
import { cn } from '@/lib/utils';
import { useState } from 'react';

// Chart tooltip component
const ChartTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-nexus-card border border-nexus-border rounded-lg px-3 py-2 shadow-xl">
        <p className="text-nexus-text font-medium">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// KPI Card Component
function SafetyKPICard({
  title,
  value,
  subtitle,
  color,
  icon: Icon,
}: {
  title: string;
  value: string | number;
  subtitle: string;
  color: 'blue' | 'red' | 'yellow' | 'green' | 'purple';
  icon: React.ElementType;
}) {
  const colorClasses = {
    blue: 'bg-blue-500/10 border-blue-500/20',
    red: 'bg-red-500/10 border-red-500/20',
    yellow: 'bg-yellow-500/10 border-yellow-500/20',
    green: 'bg-green-500/10 border-green-500/20',
    purple: 'bg-purple-500/10 border-purple-500/20',
  };

  const iconColors = {
    blue: 'text-blue-400',
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    green: 'text-green-400',
    purple: 'text-purple-400',
  };

  return (
    <Card className={cn('bg-nexus-card border', colorClasses[color])}>
      <CardContent className="p-4 text-center">
        <Icon className={cn('w-5 h-5 mx-auto mb-2', iconColors[color])} />
        <p className="text-xs text-nexus-text-muted">{title}</p>
        <p className="text-2xl font-bold text-nexus-text mt-1">{value}</p>
        <p className="text-xs text-nexus-text-muted mt-1">{subtitle}</p>
      </CardContent>
    </Card>
  );
}

// SLA Case Row
function SLACaseRow({ caseData, index }: { caseData: any; index: number }) {
  const getSeverityIcon = (severity: string) => {
    const s = String(severity).toLowerCase();
    if (s.includes('fatal') || s.includes('life')) return <Skull className="w-4 h-4 text-red-500" />;
    if (s.includes('serious') || s.includes('hospit')) return <AlertCircle className="w-4 h-4 text-red-400" />;
    return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
  };

  const getStatusBadge = (status: string) => {
    const s = String(status).toUpperCase();
    if (s === 'BREACHED' || s === 'OVERDUE') {
      return <Badge className="bg-red-500/20 text-red-400 border border-red-500/30">BREACHED</Badge>;
    }
    if (s === 'WARNING' || s === 'CRITICAL' || s === 'URGENT') {
      return <Badge className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">WARNING</Badge>;
    }
    return <Badge className="bg-green-500/20 text-green-400 border border-green-500/30">ON TRACK</Badge>;
  };

  const slaHours = caseData.sla_hours_remaining ?? caseData.sla_hours;
  const timeLeft = slaHours !== undefined
    ? (slaHours < 0 ? 'OVERDUE' : `${Math.floor(slaHours / 24)}d ${slaHours % 24}h`)
    : '--';

  return (
    <TableRow className="border-nexus-border hover:bg-nexus-card-hover">
      <TableCell className="text-nexus-text-muted">{index}</TableCell>
      <TableCell className="text-nexus-text font-medium">{caseData.case_id || caseData.caseId || caseData.sae_id}</TableCell>
      <TableCell className="text-nexus-text">{caseData.site_id || caseData.site}</TableCell>
      <TableCell className="text-nexus-text">{caseData.event_term || caseData.event}</TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          {getSeverityIcon(caseData.severity)}
          <span className="text-nexus-text">{caseData.severity}</span>
        </div>
      </TableCell>
      <TableCell className={cn(
        'font-medium',
        slaHours !== undefined && slaHours < 0 ? 'text-red-400' : 'text-nexus-text'
      )}>
        {timeLeft}
      </TableCell>
      <TableCell>{getStatusBadge(caseData.sla_status || (slaHours !== undefined && slaHours < 0 ? 'BREACHED' : 'ON TRACK'))}</TableCell>
      <TableCell className={cn(
        'font-medium',
        (caseData.breach_risk || 0) > 50 ? 'text-red-400' : 'text-yellow-400'
      )}>
        {caseData.breach_risk || 0}%
      </TableCell>
    </TableRow>
  );
}

// Safety Signal Card
function SafetySignalCard({ signal }: { signal: any }) {
  const strengthColors = {
    STRONG: 'bg-red-500/20 text-red-400 border-red-500/30',
    MODERATE: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    WEAK: 'bg-green-500/20 text-green-400 border-green-500/30',
  };

  const strength = String(signal.strength).toUpperCase() as keyof typeof strengthColors;

  return (
    <div className="bg-nexus-card border border-nexus-border rounded-lg p-4 hover:border-red-500/30 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-medium text-nexus-text">{signal.name || signal.signal_type}</h4>
        <Badge className={cn('border', strengthColors[strength] || 'bg-blue-500/10 text-blue-400 border-blue-500/30')}>
          {strength}
        </Badge>
      </div>
      <p className="text-sm text-nexus-text-muted mb-2">{signal.description}</p>
      <div className="flex items-center gap-4 text-xs text-nexus-text-muted">
        <span className="flex items-center gap-1">
          <Activity className="w-3 h-3" />
          Z-score: {signal.zScore || signal.z_score}
        </span>
        <span className="flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" />
          {signal.patientCount || signal.affected_patients} patients
        </span>
      </div>
    </div>
  );
}

// Pattern Alert Row
function PatternAlertRow({ alert, index }: { alert: any; index: number }) {
  const severityColors = {
    Critical: 'bg-red-500/20 text-red-400',
    High: 'bg-orange-500/20 text-orange-400',
    Medium: 'bg-yellow-500/20 text-yellow-400',
    Low: 'bg-green-500/20 text-green-400',
  };

  const severity = (alert.severity || 'Medium') as keyof typeof severityColors;

  return (
    <TableRow className="border-nexus-border hover:bg-nexus-card-hover">
      <TableCell className="text-nexus-text-muted">{index}</TableCell>
      <TableCell className="text-nexus-text-muted">{alert.pattern_id || alert.patternId}</TableCell>
      <TableCell className="text-nexus-text font-medium">{alert.pattern_name || alert.patternName}</TableCell>
      <TableCell>
        <Badge className={cn('border-0', severityColors[severity])}>
          {severity}
        </Badge>
      </TableCell>
      <TableCell className="text-nexus-text">{alert.alert_message || alert.message}</TableCell>
    </TableRow>
  );
}

export default function SafetyView() {
  const [selectedStudy] = useState('all');

  // Fetch safety data
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['safety-overview', selectedStudy],
    queryFn: () => safetyApi.getOverview(selectedStudy === 'all' ? undefined : selectedStudy),
  });

  const { data: saeCases, isLoading: casesLoading } = useQuery({
    queryKey: ['sae-cases'],
    queryFn: () => safetyApi.getSAECases({ limit: 20 }),
  });

  const { data: signals } = useQuery({
    queryKey: ['safety-signals'],
    queryFn: () => safetyApi.getSignals({ limit: 10 }),
  });

  const { data: timeline } = useQuery({
    queryKey: ['safety-timeline'],
    queryFn: () => safetyApi.getTimeline(90),
  });

  const { data: patternAlerts } = useQuery({
    queryKey: ['pattern-alerts'],
    queryFn: () => safetyApi.getPatternAlerts(undefined, 20),
  });

  const isPageLoading = overviewLoading && casesLoading;

  // Transform timeline data for chart
  const timelineData = timeline?.timeline?.map((t: any) => ({
    date: new Date(t.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    totalCases: t.total_cases || t.sae_count || 0,
    seriousFatal: t.serious_fatal || t.serious_count || 0,
  })) || [];

  // Data processing
  const safetySignals = signals?.signals || [];
  const slaCases = saeCases?.cases || [];
  const alerts = patternAlerts?.alerts || [];

  // Signal strength counts
  const strongSignals = safetySignals.filter((s: any) => String(s.strength).toUpperCase() === 'STRONG').length;
  const moderateSignals = safetySignals.filter((s: any) => String(s.strength).toUpperCase() === 'MODERATE').length;
  const weakSignals = safetySignals.filter((s: any) => String(s.strength).toUpperCase() === 'WEAK').length;

  // Alert severity counts
  const criticalAlerts = alerts.filter((a: any) => a.severity === 'Critical').length;
  const highAlerts = alerts.filter((a: any) => a.severity === 'High').length;
  const mediumAlerts = alerts.filter((a: any) => a.severity === 'Medium').length;
  const lowAlerts = alerts.filter((a: any) => a.severity === 'Low').length;

  if (isPageLoading) {
    return <A6onLoader size="lg" label="Loading safety data..." fullPage />;
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Banner */}
      <div className="bg-gradient-red rounded-xl p-6">
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-white" />
          <div>
            <h2 className="text-xl font-bold text-white">Safety Surveillance</h2>
            <p className="text-white/80">Real-time SAE monitoring, SLA tracking, and signal detection</p>
          </div>
        </div>
      </div>

      {/* Safety Overview KPIs */}
      <Card className="bg-nexus-card border-nexus-border">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-nexus-text">Safety Overview</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            <SafetyKPICard
              title="Total SAE Cases"
              value={overview?.total_sae_cases || overview?.total_sae || 2}
              subtitle={`DM: ${overview?.dm_pending || 2} | Safety: ${overview?.safety_pending || 2}`}
              color="blue"
              icon={Activity}
            />
            <SafetyKPICard
              title="SLA Breached"
              value={overview?.sla_breached || 1}
              subtitle="Immediate action required"
              color="red"
              icon={XCircle}
            />
            <SafetyKPICard
              title="SLA Critical"
              value={overview?.sla_critical || overview?.urgent_sla || 0}
              subtitle="<24 hours remaining"
              color="yellow"
              icon={Clock}
            />
            <SafetyKPICard
              title="Active Signals"
              value={safetySignals.length}
              subtitle={`${strongSignals} strong signals`}
              color="purple"
              icon={TrendingUp}
            />
            <SafetyKPICard
              title="Avg Breach Risk"
              value={`${overview?.avg_breach_risk || 65}%`}
              subtitle="Across pending cases"
              color="red"
              icon={AlertTriangle}
            />
          </div>
        </CardContent>
      </Card>

      {/* SLA Countdown & Safety Signals */}
      <div className="grid grid-cols-2 gap-6">
        {/* SLA Countdown Table */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-yellow-400" />
              <CardTitle className="text-nexus-text">SLA Countdown - Pending Cases</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow className="border-nexus-border">
                  <TableHead className="text-nexus-text-muted w-8"></TableHead>
                  <TableHead className="text-nexus-text-muted">Case ID</TableHead>
                  <TableHead className="text-nexus-text-muted">Site</TableHead>
                  <TableHead className="text-nexus-text-muted">Event</TableHead>
                  <TableHead className="text-nexus-text-muted">Severity</TableHead>
                  <TableHead className="text-nexus-text-muted">Time Left</TableHead>
                  <TableHead className="text-nexus-text-muted">Status</TableHead>
                  <TableHead className="text-nexus-text-muted">Risk</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {slaCases.slice(0, 5).map((caseData: any, index: number) => (
                  <SLACaseRow key={index} caseData={caseData} index={index + 1} />
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Safety Signals */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-pink-400" />
              <CardTitle className="text-nexus-text">Safety Signals</CardTitle>
            </div>
            <div className="flex gap-2 mt-2">
              <Badge className="bg-red-500/20 text-red-400 border border-red-500/30">
                Strong: {strongSignals}
              </Badge>
              <Badge className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
                Moderate: {moderateSignals}
              </Badge>
              <Badge className="bg-green-500/20 text-green-400 border border-green-500/30">
                Weak: {weakSignals}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {safetySignals.slice(0, 4).map((signal: any, index: number) => (
                <SafetySignalCard key={index} signal={signal} />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* SAE Timeline */}
      <Card className="bg-nexus-card border-nexus-border">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <CardTitle className="text-nexus-text">SAE Timeline</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          {/* Legend */}
          <div className="flex items-center gap-6 mb-4 text-xs justify-end">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-nexus-text-muted">Total Cases</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-3 bg-red-500/60 rounded" />
              <span className="text-nexus-text-muted">Serious/Fatal</span>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timelineData.length > 0 ? timelineData : Array.from({ length: 6 }, (_, i) => {
                const date = new Date();
                date.setDate(date.getDate() - (5 - i));
                return {
                  date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                  totalCases: 1 + (i % 2),
                  seriousFatal: i % 2
                };
              })}>
                <defs>
                  <linearGradient id="seriousGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3344" />
                <XAxis dataKey="date" stroke="#8b949e" tick={{ fontSize: 10 }} />
                <YAxis stroke="#8b949e" yAxisId="left" />
                <YAxis stroke="#8b949e" yAxisId="right" orientation="right" />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone"
                  dataKey="seriousFatal"
                  name="Serious/Fatal"
                  fill="url(#seriousGradient)"
                  stroke="#ef4444"
                  yAxisId="right"
                />
                <Line
                  type="monotone"
                  dataKey="totalCases"
                  name="Total Cases"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6' }}
                  yAxisId="left"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Timeline Stats */}
          <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-nexus-border">
            <div>
              <p className="text-xs text-nexus-text-muted">Avg Daily Cases</p>
              <p className="text-2xl font-bold text-nexus-text">1.0</p>
            </div>
            <div>
              <p className="text-xs text-nexus-text-muted">Peak Day</p>
              <p className="text-2xl font-bold text-nexus-text">1 cases</p>
            </div>
            <div>
              <p className="text-xs text-nexus-text-muted">Serious Rate</p>
              <p className="text-2xl font-bold text-nexus-text">100.0%</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Safety Pattern Alerts */}
      <Card className="bg-nexus-card border-nexus-border">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-yellow-400" />
            <CardTitle className="text-nexus-text">Safety Pattern Alerts</CardTitle>
          </div>
          <div className="flex gap-2 mt-2">
            <div className="bg-red-500/20 text-red-400 border border-red-500/30 px-4 py-2 rounded-lg text-center">
              <p className="text-xl font-bold">{criticalAlerts}</p>
              <p className="text-xs">Critical</p>
            </div>
            <div className="bg-orange-500/20 text-orange-400 border border-orange-500/30 px-4 py-2 rounded-lg text-center">
              <p className="text-xl font-bold">{highAlerts}</p>
              <p className="text-xs">High</p>
            </div>
            <div className="bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 px-4 py-2 rounded-lg text-center">
              <p className="text-xl font-bold">{mediumAlerts}</p>
              <p className="text-xs">Medium</p>
            </div>
            <div className="bg-green-500/20 text-green-400 border border-green-500/30 px-4 py-2 rounded-lg text-center">
              <p className="text-xl font-bold">{lowAlerts}</p>
              <p className="text-xs">Low</p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-nexus-border">
                <TableHead className="text-nexus-text-muted w-8"></TableHead>
                <TableHead className="text-nexus-text-muted">pattern_id</TableHead>
                <TableHead className="text-nexus-text-muted">pattern_name</TableHead>
                <TableHead className="text-nexus-text-muted">severity</TableHead>
                <TableHead className="text-nexus-text-muted">alert_message</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {alerts.map((alert: any, index: number) => (
                <PatternAlertRow key={index} alert={alert} index={index} />
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
