import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { analyticsApi, issuesApi, studiesApi } from '@/services/api';
import { useAppStore } from '@/stores/appStore';
import A6onLoader from '@/components/a6on_iLoader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Users,
  TrendingUp,
  TrendingDown,
  Lock,
  AlertTriangle,
  Sparkles,
  BarChart3,
  Globe,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
} from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  Line,
  Area,
  AreaChart,
  ReferenceLine,
} from 'recharts';
import { formatNumber, cn } from '@/lib/utils';

interface ChartEntry {
  name: string;
  value: number;
  color?: string;
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  label?: string;
}

interface Study {
  study_id: string;
  name: string;
  protocol_number: string;
  patient_count: number;
  avg_dqi: number;
  status: string;
  phase?: string;
  therapeutic_area?: string;
}

interface RegionMetric {
  region: string;
  avg_dqi: number;
  patient_count: number;
  site_count: number;
}

// Chart tooltip component
const ChartTooltip = ({ active, payload, label }: TooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-nexus-card border border-nexus-border rounded-lg px-3 py-2 shadow-xl">
        <p className="text-nexus-text font-medium">{label}</p>
        {payload.map((entry) => (
          <p key={entry.name} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// KPI Card Component
function KPICard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendLabel,
  color = 'blue',
  badge,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  trend?: { value: number; direction: 'up' | 'down' };
  trendLabel?: string;
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple';
  badge?: { text: string; variant: 'success' | 'warning' | 'error' };
}) {
  const colorClasses = {
    blue: 'bg-blue-500/10 border-blue-500/20 text-blue-400',
    green: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    yellow: 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400',
    red: 'bg-red-500/10 border-red-500/20 text-red-400',
    purple: 'bg-purple-500/10 border-purple-500/20 text-purple-400',
  };

  const iconColors = {
    blue: 'text-blue-400',
    green: 'text-emerald-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    purple: 'text-purple-400',
  };

  return (
    <Card className={cn('bg-nexus-card border-nexus-border overflow-hidden', colorClasses[color])}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium text-nexus-text-muted truncate">{title}</p>
            <p className="mt-1 text-2xl font-bold text-nexus-text truncate">{value}</p>
            {subtitle && (
              <p className="mt-0.5 text-xs text-nexus-text-muted truncate">{subtitle}</p>
            )}
            {trend && (
              <div className="mt-1.5 flex items-center gap-1 text-xs min-w-0">
                {trend.direction === 'up' ? (
                  <TrendingUp className="w-3.5 h-3.5 shrink-0 text-emerald-400" />
                ) : (
                  <TrendingDown className="w-3.5 h-3.5 shrink-0 text-red-400" />
                )}
                {trendLabel && <span className="text-nexus-text-muted truncate">{trendLabel}</span>}
              </div>
            )}
            {badge && (
              <Badge
                variant={badge.variant}
                className={cn(
                  'mt-1.5 text-xs',
                  badge.variant === 'success' && 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
                  badge.variant === 'warning' && 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
                  badge.variant === 'error' && 'bg-red-500/20 text-red-400 border-red-500/30'
                )}
              >
                {badge.text}
              </Badge>
            )}
          </div>
          <div className={cn('p-2 rounded-lg shrink-0', colorClasses[color])}>
            <Icon className={cn('w-5 h-5', iconColors[color])} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// DQI Donut Chart
function DQIDonutChart({ data, mean }: { data: ChartEntry[]; mean: number }) {
  const COLORS: Record<string, string> = {
    'Pristine': '#10b981',
    'Excellent': '#22d3ee',
    'Good': '#f59e0b',
    'Fair': '#f97316',
    'Critical': '#ef4444',
    'Emergency': '#dc2626',
  };

  const getDQIColor = (val: number) => {
    if (val >= 95) return COLORS['Pristine'];
    if (val >= 85) return COLORS['Excellent'];
    if (val >= 70) return COLORS['Good'];
    if (val >= 50) return COLORS['Fair'];
    if (val >= 30) return COLORS['Critical'];
    return COLORS['Emergency'];
  };

  // Ensure data is valid for Recharts
  const chartData = (data || []).slice(0, 6);

  return (
    <div className="relative h-64 overflow-hidden">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={80}
            paddingAngle={2}
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[entry.name] || '#6b7280'}
              />
            ))}
          </Pie>
          <Tooltip content={<ChartTooltip />} />
          <Legend
            verticalAlign="bottom"
            height={36}
            formatter={(value) => <span className="text-nexus-text-muted text-xs">{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
      {/* Center label */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center" style={{ marginTop: '-18px' }}>
        <p className="text-3xl font-bold" style={{ color: getDQIColor(mean) }}>{mean.toFixed(1)}</p>
        <p className="text-sm text-nexus-text-muted">Mean DQI</p>
      </div>
    </div>
  );
}

// Clean Patient Progress Funnel
function CleanPatientFunnel({ data }: { data: { total: number; tier1: number; tier2: number; dbLock: number; sdtm: number } }) {
  const safeTotal = data.total || 1;
  const funnelData = [
    { name: 'Total Patients', value: data.total, percent: 100, color: '#3b82f6' },
    { name: 'Tier 1 Clean', value: data.tier1, percent: Math.round((data.tier1 / safeTotal) * 100), color: '#10b981' },
    { name: 'Tier 2 Clean', value: data.tier2, percent: Math.round((data.tier2 / safeTotal) * 100), color: '#10b981' },
    { name: 'DB Lock Ready', value: data.dbLock, percent: Math.round((data.dbLock / safeTotal) * 100), color: '#10b981' },
    { name: 'Submission Ready', value: data.sdtm, percent: Math.round((data.sdtm / safeTotal) * 100), color: '#8b5cf6' },
  ];

  return (
    <div className="space-y-4">
      {funnelData.map((item) => (
        <div key={item.name} className="space-y-2">
          <div className="flex justify-between items-center text-sm">
            <span className="text-nexus-text-muted">{item.name}</span>
            <div className="flex items-center gap-2">
              <span className="text-nexus-text font-medium">
                {item.value >= 1000 ? `${(item.value / 1000).toFixed(1)}k` : item.value}
              </span>
              <span className="text-nexus-text-muted">{item.percent}%</span>
            </div>
          </div>
          <div
            className="h-8 rounded-lg transition-all duration-500"
            style={{
              width: `${Math.max(item.percent, 8)}%`,
              backgroundColor: item.color,
            }}
          />
        </div>
      ))}

      {/* Rates */}
      <div className="grid grid-cols-4 gap-4 pt-4 border-t border-nexus-border">
        <div className="text-center">
          <p className="text-2xl font-bold text-nexus-text">{Math.round((data.tier1 / safeTotal) * 100)}%</p>
          <p className="text-[10px] text-nexus-text-secondary uppercase font-bold">Tier 1</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-nexus-text">{Math.round((data.tier2 / safeTotal) * 100)}%</p>
          <p className="text-[10px] text-nexus-text-secondary uppercase font-bold">Tier 2</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-nexus-text">{Math.round((data.dbLock / safeTotal) * 100)}%</p>
          <p className="text-[10px] text-nexus-text-secondary uppercase font-bold">Lock</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-purple-400">{Math.round((data.sdtm / safeTotal) * 100)}%</p>
          <p className="text-[10px] text-nexus-text-secondary uppercase font-bold">Ready</p>
        </div>
      </div>
    </div>
  );
}

// Study Level Metrics Chart
function StudyMetricsChart({ data, sortBy }: { data: Array<{ name: string; patients: number; dqi: number }>; sortBy: string }) {
  const sortedData = [...data].sort((a, b) => {
    if (sortBy === 'patients') return b.patients - a.patients;
    if (sortBy === 'dqi') return b.dqi - a.dqi;
    return 0;
  }).slice(0, 10);

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sortedData} layout="vertical" margin={{ left: 60 }}>
          <XAxis type="number" stroke="#8b949e" hide />
          <YAxis type="category" dataKey="name" stroke="#8b949e" width={80} tick={{ fontSize: 10, fill: '#e8eaed' }} />
          <Tooltip content={<ChartTooltip />} />
          <Bar
            dataKey={sortBy === 'dqi' ? 'dqi' : 'patients'}
            fill="#3b82f6"
            radius={[0, 4, 4, 0]}
            label={{ position: 'right', fill: '#e8eaed', fontSize: 11 }}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Regional Performance Chart
function RegionalChart({ data }: { data: Array<{ name: string; dqi: number }> }) {
  const getBarColor = (dqi: number) => {
    if (dqi >= 85) return '#10b981';
    if (dqi >= 70) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" domain={[0, 100]} stroke="#8b949e" hide />
          <YAxis type="category" dataKey="name" stroke="#8b949e" width={80} tick={{ fontSize: 11 }} />
          <Tooltip content={<ChartTooltip />} />
          <ReferenceLine x={85} stroke="#ef4444" strokeDasharray="5 5" />
          <Bar dataKey="dqi" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.dqi)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// DB Lock Projection Chart
function DBLockProjectionChart({ data }: { data: { historical: Array<{ date: string; value: number }>; projected: Array<{ date: string; value: number | null; projected: number }> } }) {
  const allData = [
    ...data.historical.map(d => ({ ...d, type: 'historical' })),
    ...data.projected.map(d => ({ ...d, type: 'projected' })),
  ];

  return (
    <div className="h-64 overflow-hidden">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={allData}>
          <defs>
            <linearGradient id="projectedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="date" stroke="#8b949e" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 60]} stroke="#8b949e" hide />
          <Tooltip content={<ChartTooltip />} />
          <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
          <Area type="monotone" dataKey="projected" stroke="#10b981" fill="url(#projectedGradient)" strokeDasharray="5 5" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// Loading skeleton
function LoadingSkeleton() {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        {[1, 2, 3, 4, 5].map((i) => (
          <Card key={i} className="bg-nexus-card border-nexus-border">
            <CardContent className="p-5">
              <Skeleton className="h-4 w-24 mb-2 bg-nexus-border" />
              <Skeleton className="h-8 w-32 mb-2 bg-nexus-border" />
              <Skeleton className="h-3 w-20 bg-nexus-border" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader>
            <Skeleton className="h-5 w-32 bg-nexus-border" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-64 w-full bg-nexus-border" />
          </CardContent>
        </Card>
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader>
            <Skeleton className="h-5 w-32 bg-nexus-border" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-64 w-full bg-nexus-border" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function ExecutiveOverview() {
  const [studySortBy, setStudySortBy] = useState('dqi');
  const [dataTableOpen, setDataTableOpen] = useState(false);
  const { selectedStudy } = useAppStore();

  // Fetch portfolio summary with study filter
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio', selectedStudy],
    queryFn: () => analyticsApi.getPortfolio(selectedStudy),
  });

  // Fetch DQI distribution with study filter
  const { data: dqiDistribution, isLoading: dqiLoading } = useQuery({
    queryKey: ['dqi-distribution', selectedStudy],
    queryFn: () => analyticsApi.getDQIDistribution(selectedStudy),
  });

  // Fetch issues summary with study filter
  const { data: issuesSummary, isLoading: issuesLoading } = useQuery({
    queryKey: ['issues-summary', selectedStudy],
    queryFn: () => issuesApi.getSummary(selectedStudy),
  });

  // Fetch studies
  const { data: studies } = useQuery({
    queryKey: ['studies'],
    queryFn: () => studiesApi.list(),
  });

  // Fetch regional metrics
  const { data: regionalMetrics } = useQuery({
    queryKey: ['regional-metrics'],
    queryFn: () => analyticsApi.getRegional(),
  });

  const isLoading = portfolioLoading || dqiLoading || issuesLoading;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-8 h-8 text-primary" />
            <h1 className="text-2xl font-bold text-nexus-text">Executive Command Center</h1>
          </div>
          <p className="text-nexus-text-muted font-mono text-sm">// PORTFOLIO_INTELLIGENCE_v10.0</p>
        </div>
        <A6onLoader size="lg" label="Loading portfolio intelligence..." fullPage />
      </div>
    );
  }

  // Transform DQI distribution for chart
  const dqiChartData = Array.isArray(dqiDistribution?.distribution)
    ? dqiDistribution.distribution.slice(0, 10).map((item: { dqi_band: string; count: number; percentage: number }) => ({
      name: item.dqi_band,
      value: item.count,
      percent: item.percentage,
    }))
    : [];

  // DQI summary stats with percentages
  const distribution = Array.isArray(dqiDistribution?.distribution) ? dqiDistribution.distribution : [];
  const totalDqiCount = dqiDistribution?.total || distribution.reduce((sum: number, d: { count?: number; value?: number }) => sum + (d.count || d.value || 0), 0) || 1;

  // Robust case-insensitive and key-checking lookup
  const getBandCount = (bandName: string) => {
    const item = distribution.find((d: { dqi_band?: string; name?: string; count?: number; value?: number }) =>
      (d.dqi_band || '').toLowerCase() === bandName.toLowerCase() ||
      (d.name || '').toLowerCase() === bandName.toLowerCase()
    );
    return item?.count || item?.value || 0;
  };

  const pristine = getBandCount('Pristine');
  const excellent = getBandCount('Excellent');
  const good = getBandCount('Good');
  const fair = getBandCount('Fair');
  const critical = getBandCount('Critical');
  const emergency = getBandCount('Emergency');

  const pristineExcellentCount = pristine + excellent;
  const goodFair = good + fair;
  const needsAttention = critical + emergency;

  const pristineExcellentPct = ((pristineExcellentCount / totalDqiCount) * 100).toFixed(1);
  const goodFairPct = ((goodFair / totalDqiCount) * 100).toFixed(1);
  const needsAttentionPct = ((needsAttention / totalDqiCount) * 100).toFixed(1);

  // Study level data - hard slice to 10
  const studyData = (Array.isArray(studies?.studies) ? (studies.studies as Study[]) : [])
    .map((s: Study) => ({
      name: s.study_id,
      patients: s.patient_count || 0,
      dqi: s.avg_dqi || 0,
    }));

  // Regional data - hard slice to 10
  const regionData = (Array.isArray(regionalMetrics?.regions) ? (regionalMetrics.regions as RegionMetric[]) : [])
    .slice(0, 10)
    .map((r: RegionMetric) => ({
      name: r.region,
      dqi: Math.round((r.avg_dqi || 0) * 10) / 10,
      patients: r.patient_count || 0,
      sites: r.site_count || 0,
    }));

  // Clean patient funnel data - use real data from portfolio API
  const totalPatients = portfolio?.total_patients || 0;
  const tier1Count = portfolio?.tier1_clean_count || 0;
  const tier2Count = portfolio?.tier2_clean_count || 0;
  const dbLockCount = portfolio?.dblock_ready_count || 0;
  const sdtmCount = portfolio?.sdtm_ready_count || 0;

  const funnelData = {
    total: totalPatients,
    tier1: tier1Count,
    tier2: tier2Count,
    dbLock: dbLockCount,
    sdtm: sdtmCount,
  };

  // DB Lock projection data (simulated)
  const dbLockProjection = {
    historical: [
      { date: 'Nov 16', value: 5 },
      { date: 'Nov 30', value: 6 },
      { date: 'Dec 14', value: 7 },
      { date: 'Dec 28', value: 8 },
      { date: 'Jan 11', value: 9 },
    ],
    projected: [
      { date: 'Jan 25', value: 10, projected: 10 },
      { date: 'Feb 8', value: null, projected: 13 },
      { date: 'Feb 22', value: null, projected: 17 },
    ],
  };

  return (
    <div className="space-y-6 fade-in">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <BarChart3 className="w-8 h-8 text-primary" />
          <h1 className="text-2xl font-bold text-nexus-text">Executive Command Center</h1>
        </div>
        <p className="text-nexus-text-muted font-mono text-sm">// PORTFOLIO_INTELLIGENCE_v10.0</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6">
        <KPICard
          title="Total Patients"
          value={formatNumber(portfolio?.total_patients || 0)}
          subtitle={`Active enrollment`}
          icon={Users}
          color="blue"
          trendLabel={`${formatNumber(portfolio?.total_sites || 0)} sites`}
        />
        <KPICard
          title="Portfolio DQI"
          value={(portfolio?.mean_dqi || 0).toFixed(1)}
          subtitle="Data Quality Index"
          icon={BarChart3}
          color="green"
          badge={{ text: 'Elite', variant: 'success' }}
        />
        <KPICard
          title="Clean Rate"
          value={`${((portfolio?.tier2_clean_rate || 0)).toFixed(1)}%`}
          subtitle="Tier 2 completion"
          icon={Sparkles}
          color="yellow"
          trend={{ value: funnelData.tier2, direction: 'down' }}
          trendLabel="clean"
        />
        <KPICard
          title="DB Lock Ready"
          value={`${((portfolio?.dblock_ready_rate || 0)).toFixed(1)}%`}
          subtitle="Eligible patients"
          icon={Lock}
          color="yellow"
          trend={{ value: funnelData.dbLock, direction: 'down' }}
          trendLabel={`${formatNumber(funnelData.dbLock)}/${formatNumber(funnelData.total)}`}
        />
        <KPICard
          title="Submission Ready"
          value={`${((portfolio?.sdtm_ready_rate || 0)).toFixed(1)}%`}
          subtitle="SDTM mapped & clean"
          icon={CheckCircle2}
          color="green"
          badge={{ text: 'Phase III', variant: 'success' }}
        />
        <KPICard
          title="Open Issues"
          value={formatNumber(issuesSummary?.open_count || 0)}
          subtitle="Requires attention"
          icon={AlertTriangle}
          color="red"
          badge={{
            text: `${issuesSummary?.critical_count || 0} critical`,
            variant: issuesSummary?.critical_count > 0 ? 'error' : 'success'
          }}
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* DQI Distribution */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-primary" />
              <CardTitle className="text-nexus-text">DQI Distribution</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <DQIDonutChart
              data={dqiChartData}
              mean={portfolio?.mean_dqi || 0}
            />

            {/* Summary stats below chart */}
            <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-nexus-border">
              <div>
                <p className="text-2xl font-bold text-emerald-400">{formatNumber(pristineExcellentCount)}</p>
                <p className="text-xs text-nexus-text-muted">Pristine/Excellent</p>
                <Badge className="mt-1 bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {pristineExcellentPct}%
                </Badge>
              </div>
              <div>
                <p className="text-2xl font-bold text-yellow-400">{formatNumber(goodFair)}</p>
                <p className="text-xs text-nexus-text-muted">Good/Fair</p>
                <Badge className="mt-1 bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {goodFairPct}%
                </Badge>
              </div>
              <div>
                <p className="text-2xl font-bold text-red-400">{formatNumber(needsAttention)}</p>
                <p className="text-xs text-nexus-text-muted">Needs Attention</p>
                <Badge className="mt-1 bg-red-500/20 text-red-400 border-red-500/30">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {needsAttentionPct}%
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Clean Patient Progress */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-yellow-400" />
              <CardTitle className="text-nexus-text">Clean Patient Progress</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <CleanPatientFunnel data={funnelData} />
          </CardContent>
        </Card>
      </div>

      {/* Study Level Metrics */}
      <Card className="bg-nexus-card border-nexus-border">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <CardTitle className="text-nexus-text">Study-Level Metrics</CardTitle>
            </div>
            <Select value={studySortBy} onValueChange={setStudySortBy}>
              <SelectTrigger className="w-40 bg-nexus-card border-nexus-border text-nexus-text">
                <span className="text-nexus-text-muted mr-2">Sort by</span>
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-nexus-card border-nexus-border">
                <SelectItem value="patients" className="text-nexus-text">Patients</SelectItem>
                <SelectItem value="dqi" className="text-nexus-text">DQI Score</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <StudyMetricsChart data={studyData} sortBy={studySortBy} />

          {/* Expandable data table */}
          <button
            onClick={() => setDataTableOpen(!dataTableOpen)}
            className="flex items-center gap-2 mt-4 text-sm text-nexus-text-muted hover:text-nexus-text transition-colors"
          >
            {dataTableOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            View Full Data Table
          </button>

          {/* Data Table */}
          {dataTableOpen && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-nexus-border">
                    <th className="text-left py-3 px-4 text-nexus-text-muted font-medium">Study ID</th>
                    <th className="text-right py-3 px-4 text-nexus-text-muted font-medium">Patients</th>
                    <th className="text-right py-3 px-4 text-nexus-text-muted font-medium">Avg DQI</th>
                    <th className="text-left py-3 px-4 text-nexus-text-muted font-medium">Status</th>
                    <th className="text-left py-3 px-4 text-nexus-text-muted font-medium">Phase</th>
                    <th className="text-left py-3 px-4 text-nexus-text-muted font-medium">Therapeutic Area</th>
                  </tr>
                </thead>
                <tbody>
                  {studies?.studies
                    ?.sort((a: Study, b: Study) => {
                      if (studySortBy === 'patients') return (b.patient_count || 0) - (a.patient_count || 0);
                      if (studySortBy === 'dqi') return (b.avg_dqi || 0) - (a.avg_dqi || 0);
                      return 0;
                    })
                    ?.slice(0, 30)
                    .map((study: Study) => (
                      <tr key={study.study_id} className="border-b border-nexus-border/50 hover:bg-nexus-bg/50 transition-colors">
                        <td className="py-3 px-4 text-nexus-text font-medium">{study.study_id}</td>
                        <td className="py-3 px-4 text-nexus-text text-right">{formatNumber(study.patient_count || 0)}</td>
                        <td className="py-3 px-4 text-right">
                          <span className={cn(
                            'font-medium',
                            (study.avg_dqi || 0) >= 90 ? 'text-emerald-400' :
                              (study.avg_dqi || 0) >= 70 ? 'text-yellow-400' : 'text-red-400'
                          )}>
                            {(study.avg_dqi || 0).toFixed(1)}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <Badge
                            variant={study.status === 'Active' ? 'success' : 'warning'}
                            className={cn(
                              study.status === 'Active' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
                                'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                            )}
                          >
                            {study.status || 'Unknown'}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-nexus-text-muted">{study.phase || '-'}</td>
                        <td className="py-3 px-4 text-nexus-text-muted">{study.therapeutic_area || '-'}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              {(!studies?.studies || studies.studies.length === 0) && (
                <p className="text-center py-8 text-nexus-text-muted">No studies found</p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Regional Performance & DB Lock Projection */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Regional Performance */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Globe className="w-5 h-5 text-cyan-400" />
              <CardTitle className="text-nexus-text">Regional Performance</CardTitle>
            </div>
            <p className="text-xs text-nexus-text-muted mt-1">Mean DQI by Region</p>
          </CardHeader>
          <CardContent>
            <RegionalChart data={regionData} />

            {/* Regional summary cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
              {regionData.slice(0, 8).map((region: { name: string; patients: number; sites: number; dqi: number }) => (
                <div key={region.name} className="p-3 rounded-lg bg-nexus-bg border border-nexus-border">
                  <p className="font-medium text-nexus-text text-sm">{region.name}</p>
                  <p className="text-xs text-nexus-text-muted mt-1">
                    {formatNumber(region.patients)} patients - {region.sites} sites - DQI: {region.dqi}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* DB Lock Projection */}
        <Card className="bg-nexus-card border-nexus-border">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Lock className="w-5 h-5 text-yellow-400" />
              <CardTitle className="text-nexus-text">Database Lock Projection</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            {/* Legend */}
            <div className="flex items-center gap-6 mb-4 text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-blue-500" />
                <span className="text-nexus-text-muted">Historical</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-emerald-500" />
                <span className="text-nexus-text-muted">Projected</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-6 h-3 bg-emerald-500/30 rounded" />
                <span className="text-nexus-text-muted">Confidence Band</span>
              </div>
            </div>

            <DBLockProjectionChart data={dbLockProjection} />

            {/* Projection summary */}
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <p className="text-sm text-nexus-text-muted">Current Status</p>
                <p className="text-2xl font-bold text-nexus-text mt-1">
                  {((portfolio?.dblock_ready_rate || 0)).toFixed(1)}%
                </p>
                <p className="text-xs text-nexus-text-muted mt-1">
                  {formatNumber(funnelData.dbLock)} patients ready
                </p>
              </div>
              <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <p className="text-sm text-nexus-text-muted">Target 100% Ready</p>
                <p className="text-2xl font-bold text-nexus-text mt-1">June 15, 2026</p>
                <p className="text-xs text-nexus-text-muted mt-1">~20 weeks remaining</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Submission Milestones (Requirement: Statistical Deliverables) */}
      <Card className="bg-nexus-card border-nexus-border">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            Statistical Deliverables Readiness
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
              <p className="text-[10px] text-nexus-text-secondary uppercase font-bold mb-2">SDTM Mapping</p>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-black text-white">{((portfolio?.tier2_clean_rate || 0)).toFixed(1)}%</span>
                <Badge variant="success" className="bg-emerald-500/20 text-emerald-400">ON TRACK</Badge>
              </div>
              <Progress value={portfolio?.tier2_clean_rate || 0} className="h-1 mt-3" />
            </div>
            <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
              <p className="text-[10px] text-nexus-text-secondary uppercase font-bold mb-2">ADaM Readiness</p>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-black text-white">{((portfolio?.dblock_ready_rate || 0)).toFixed(1)}%</span>
                <Badge variant="warning" className="bg-yellow-500/20 text-yellow-400">IN PROGRESS</Badge>
              </div>
              <Progress value={portfolio?.dblock_ready_rate || 0} className="h-1 mt-3" />
            </div>
            <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
              <p className="text-[10px] text-nexus-text-secondary uppercase font-bold mb-2">TFL Generation</p>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-black text-white">{((portfolio?.sdtm_ready_rate || 0)).toFixed(1)}%</span>
                <Badge variant="secondary" className="bg-blue-500/20 text-blue-400">INITIAL</Badge>
              </div>
              <Progress value={portfolio?.sdtm_ready_rate || 0} className="h-1 mt-3" />
            </div>
            <div className="p-4 bg-nexus-bg/50 rounded-xl border border-nexus-border">
              <p className="text-[10px] text-nexus-text-secondary uppercase font-bold mb-2">Interim Readiness</p>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-black text-emerald-400">{(portfolio?.mean_dqi || 0) > 90 ? 'HIGH' : 'MEDIUM'}</span>
                <Sparkles className="w-5 h-5 text-emerald-400" />
              </div>
              <p className="text-[10px] text-nexus-text-secondary mt-2">Predicted Date: June 15, 2026</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
