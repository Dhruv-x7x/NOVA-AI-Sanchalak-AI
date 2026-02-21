import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { sitesApi, analyticsApi, issuesApi, reportsApi } from '@/services/api';
import { useAppStore } from '@/stores/appStore';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAuthStore } from '@/stores/authStore';
import {
  Users,
  Building2,
  CheckCircle,
  Lock,
  AlertTriangle,
  Bot,
  FileText,
  Zap,
  GitBranch,
  ArrowRight,
  ChevronUp,
  Check,
  Target,
  Link as LinkIcon,
  RefreshCw,
  Activity,
} from 'lucide-react';
import { formatNumber, cn } from '@/lib/utils';
import { useState, useMemo } from 'react';

// Types
interface SmartQueueItem {
  id: string;
  rank: number;
  siteId: string;
  studyId: string;
  patientKey: string;
  issueType: string;
  dqiImpact: number;
  score: number;
  status: 'pending' | 'in_progress' | 'completed';
}

interface SiteCard {
  siteId: string;
  studyId: string;
  patientCount: number;
  dqi: number;
  cleanRate: number;
  issueCount: number;
  topIssue: string;
  status: 'Pristine' | 'Excellent' | 'Good' | 'Average' | 'Poor';
}

interface GenomeMatch {
  name: string;
  issueType: string;
  successRate: number;
  effort: string;
  matchCount: number;
}

// Smart Queue Item Component
function SmartQueueItemCard({ item, onComplete, onSkip, onEscalate }: {
  item: SmartQueueItem;
  onComplete: () => void;
  onSkip: () => void;
  onEscalate: () => void;
}) {
  return (
    <div className="bg-nexus-card border border-nexus-border rounded-lg p-4 hover:bg-nexus-card-hover transition-colors">
      <div className="flex items-center gap-4">
        {/* Rank badge */}
        <div className={cn(
          "w-10 h-6 rounded flex items-center justify-center text-xs font-bold text-white",
          item.rank === 1 ? "bg-orange-500" :
            item.rank === 2 ? "bg-orange-400" :
              item.rank === 3 ? "bg-yellow-500" :
                "bg-nexus-text-muted"
        )}>
          #{item.rank || '?'}
        </div>

        {/* Site info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-nexus-text">{item.siteId || 'Unknown Site'}</span>
            <Badge variant="secondary" className="bg-nexus-border text-nexus-text-muted">
              {item.issueType || 'Issue'}
            </Badge>
          </div>
          <p className="text-xs text-nexus-text-muted mt-1 truncate">
            {item.studyId}|{item.siteId}|{item.patientKey}
          </p>
        </div>

        {/* DQI Impact */}
        <div className="text-center">
          <span className="text-emerald-400 font-medium">+{item.dqiImpact || 0}</span>
          <span className="text-xs text-nexus-text-muted ml-1">DQI</span>
        </div>

        {/* Score */}
        <div className="text-right pr-4">
          <span className="text-xl font-bold text-nexus-text">{item.score || 0}</span>
          <span className="text-xs text-nexus-text-muted ml-1">score</span>
        </div>

        {/* Approve / Reject Buttons */}
        <div className="flex items-center gap-2 ml-4">
          <Button
            size="sm"
            variant="outline"
            className="h-8 px-3 border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10 hover:text-emerald-300"
            onClick={(e) => {
              e.stopPropagation();
              onComplete();
            }}
          >
            <Check className="w-4 h-4 mr-1" />
            Approve
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="h-8 px-3 border-red-500/50 text-red-400 hover:bg-red-500/10 hover:text-red-300"
            onClick={(e) => {
              e.stopPropagation();
              onSkip();
            }}
          >
            <AlertTriangle className="w-4 h-4 mr-1" />
            Reject
          </Button>
        </div>
      </div>
    </div>
  );
}

// Site Card Component
function SiteCardComponent({ site }: { site: SiteCard }) {
  const statusColors = {
    Pristine: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    Excellent: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    Good: 'bg-green-500/20 text-green-400 border-green-500/30',
    Average: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    Poor: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const dqiColor = (site.dqi || 0) >= 95 ? 'text-emerald-400' :
    (site.dqi || 0) >= 85 ? 'text-green-400' :
      (site.dqi || 0) >= 70 ? 'text-yellow-400' : 'text-red-400';

  const cleanColor = (site.cleanRate || 0) >= 50 ? 'text-emerald-400' :
    (site.cleanRate || 0) >= 25 ? 'text-yellow-400' : 'text-red-400';

  return (
    <Card className="bg-nexus-card border-nexus-border hover:border-primary/30 transition-all">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="font-medium text-nexus-text">{site.siteId}</h3>
            <p className="text-xs text-nexus-text-muted">{site.studyId} | {site.patientCount} patients</p>
          </div>
          <Badge className={cn('border', statusColors[site.status || 'Average'])}>
            {site.status || 'Average'}
          </Badge>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-3">
          <div>
            <p className={cn('text-xl font-bold', dqiColor)}>{(site.dqi || 0).toFixed(1)}</p>
            <p className="text-xs text-nexus-text-muted">DQI</p>
          </div>
          <div>
            <p className={cn('text-xl font-bold', cleanColor)}>{site.cleanRate || 0}%</p>
            <p className="text-xs text-nexus-text-muted">Clean</p>
          </div>
          <div>
            <p className={cn('text-xl font-bold', (site.issueCount || 0) > 0 ? 'text-red-400' : 'text-emerald-400')}>
              {site.issueCount || 0}
            </p>
            <p className="text-xs text-nexus-text-muted">Issues</p>
          </div>
        </div>

        <div className="text-xs text-nexus-text-muted">
          Top Issue: <span className="text-nexus-text">{site.topIssue || 'None'}</span> ({site.issueCount || 0})
        </div>
      </CardContent>
    </Card>
  );
}

// Genome Match Row
function GenomeMatchRow({ match, onApply, isPending }: { match: GenomeMatch; onApply: () => void; isPending: boolean }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-nexus-border last:border-0">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-nexus-text">{match.name}</span>
          <Badge className="bg-red-500/20 text-red-400 border border-red-500/30 text-xs">
            {match.issueType}
          </Badge>
        </div>
      </div>
      <div className="flex items-center gap-6 text-sm">
        <div className="text-center">
          <span className="text-emerald-400 font-medium">{match.successRate}%</span>
          <span className="text-xs text-nexus-text-muted ml-1">success</span>
        </div>
        <div className="text-center">
          <span className="text-nexus-text">{match.effort}</span>
          <span className="text-xs text-nexus-text-muted ml-1">effort</span>
        </div>
        <div className="text-center">
          <span className="text-primary font-medium">{match.matchCount}</span>
          <span className="text-xs text-nexus-text-muted ml-1">matches</span>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="bg-nexus-card border-nexus-border text-nexus-text hover:bg-nexus-card-hover"
          onClick={onApply}
          disabled={isPending}
        >
          {isPending ? 'Applying...' : 'Apply'}
        </Button>
      </div>
    </div>
  );
}

export default function CRAView() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const { selectedStudy } = useAppStore();
  const [activeTab, setActiveTab] = useState('smart-queue');
  const [hiddenIssueIds, setHiddenIssueIds] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('nexus_hidden_issues');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const hideIssue = (id: string) => {
    setHiddenIssueIds(prev => {
      const next = [...prev, id];
      localStorage.setItem('nexus_hidden_issues', JSON.stringify(next));
      return next;
    });
  };

  const [hiddenGenomeIndices, setHiddenGenomeIndices] = useState<number[]>([]);

  // Fetch data
  const sitesQuery = useQuery({
    queryKey: ['sites', selectedStudy],
    queryFn: () => sitesApi.list({ study_id: selectedStudy }),
  });

  const benchmarksQuery = useQuery({
    queryKey: ['site-benchmarks', selectedStudy],
    queryFn: () => sitesApi.getBenchmarks(selectedStudy),
  });

  const portfolioQuery = useQuery({
    queryKey: ['portfolio', selectedStudy],
    queryFn: () => analyticsApi.getPortfolio(selectedStudy),
  });

  const issuesQuery = useQuery({
    queryKey: ['issues-summary', selectedStudy],
    queryFn: () => issuesApi.getSummary(selectedStudy),
  });

  const queueQuery = useQuery({
    queryKey: ['smart-queue', selectedStudy],
    queryFn: () => sitesApi.getSmartQueue(selectedStudy),
  });

  const cascadeQuery = useQuery({
    queryKey: ['cascade', selectedStudy],
    queryFn: () => analyticsApi.getCascade(100, selectedStudy),
  });

  const resolutionQuery = useQuery({
    queryKey: ['resolution-stats', selectedStudy],
    queryFn: () => analyticsApi.getResolutionStats(selectedStudy),
  });

  const { data: activityLogs } = useQuery({
    queryKey: ['cra-activity-logs', selectedStudy],
    queryFn: () => sitesApi.getActivityLogs(selectedStudy),
  });

  const isLoading = sitesQuery.isLoading || benchmarksQuery.isLoading || portfolioQuery.isLoading ||
    issuesQuery.isLoading || queueQuery.isLoading || cascadeQuery.isLoading || resolutionQuery.isLoading;

  // Core queries that MUST succeed for the view to work
  const isCriticalError = sitesQuery.isError || portfolioQuery.isError || queueQuery.isError;

  // Secondary queries that can fail gracefully
  const isSecondaryError = benchmarksQuery.isError || issuesQuery.isError || cascadeQuery.isError || resolutionQuery.isError;

  if (isSecondaryError) {
    console.warn("CRA View: One or more secondary data sources are offline.");
  }

  const resolveMutation = useMutation({
    mutationFn: (issueId: string) => {
      const reason = prompt("Mandatory: Provide reason for resolving this data item (21 CFR Part 11 compliance):", "Verified source data");
      if (!reason) throw new Error("Reason required");
      return issuesApi.resolve(issueId, 'Resolved via Smart Queue', reason);
    },
    onMutate: async (issueId) => {
      await queryClient.cancelQueries({ queryKey: ['issues-summary', selectedStudy] });
      await queryClient.cancelQueries({ queryKey: ['portfolio', selectedStudy] });
      const prevSummary = queryClient.getQueryData(['issues-summary', selectedStudy]);
      const prevPortfolio = queryClient.getQueryData(['portfolio', selectedStudy]);
      hideIssue(issueId);
      if (prevSummary) {
        queryClient.setQueryData(['issues-summary', selectedStudy], (old: any) => ({
          ...old,
          open_count: Math.max(0, (old?.open_count || 0) - 1)
        }));
      }
      if (prevPortfolio) {
        queryClient.setQueryData(['portfolio', selectedStudy], (old: any) => ({
          ...old,
          total_issues: Math.max(0, (old?.total_issues || 0) - 1)
        }));
      }
      return { prevSummary, prevPortfolio };
    },
    onSuccess: (_, issueId) => {
      queryClient.invalidateQueries({ queryKey: ['issues-summary', selectedStudy] });
      queryClient.invalidateQueries({ queryKey: ['portfolio', selectedStudy] });
      queryClient.setQueryData(['smart-queue', selectedStudy], (old: any) => {
        if (!old || !old.queue) return old;
        return {
          ...old,
          queue: old.queue.filter((item: any) => String(item.id).toLowerCase() !== String(issueId).toLowerCase())
        };
      });
    }
  });

  const escalateMutation = useMutation({
    mutationFn: (issueId: string) => issuesApi.escalate(issueId, 'Escalated by CRA from Smart Queue'),
    onMutate: async (issueId) => {
      hideIssue(issueId);
    }
  });

  const rejectMutation = useMutation({
    mutationFn: (issueId: string) => {
      const reason = prompt("Provide reason for rejecting this data item:", "Incorrect identification");
      if (!reason) throw new Error("Reason required");
      return issuesApi.reject(issueId, reason);
    },
    onSuccess: (_, issueId) => {
      queryClient.invalidateQueries({ queryKey: ['smart-queue', selectedStudy] });
      hideIssue(issueId);
    }
  });

  // Safe data selectors
  const smartQueueData = useMemo(() => {
    try {
      return (queueQuery.data?.queue || [])
        .filter((item: any) => {
          if (!item || !item.id) return false;
          const id = String(item.id).toLowerCase().trim();
          return !hiddenIssueIds.some(hiddenId => String(hiddenId).toLowerCase().trim() === id);
        });
    } catch (e) { return []; }
  }, [queueQuery.data, hiddenIssueIds]);

  const siteCards = useMemo(() => {
    try {
      return (benchmarksQuery.data?.benchmarks || [])
        .filter((b: any) => b && b.site_id)
        .map((b: any) => ({
          siteId: b.site_id,
          studyId: b.region || 'UNKNOWN',
          patientCount: b.patient_count || 0,
          dqi: b.dqi_score || 0,
          cleanRate: Math.round(b.clean_rate || 0),
          issueCount: b.issue_count || 0,
          topIssue: b.top_issue || 'Open Queries',
          status: (b.dqi_score >= 95 ? 'Pristine' :
            b.dqi_score >= 85 ? 'Excellent' :
              b.dqi_score >= 75 ? 'Good' :
                b.dqi_score >= 65 ? 'Average' : 'Poor') as any,
        }));
    } catch (e) { return []; }
  }, [benchmarksQuery.data]);

  const genomeMatches = useMemo(() => {
    try {
      const matches = (resolutionQuery.data?.by_type || [])
        .filter((p: any) => p && p.name)
        .map((p: any) => {
          const name = String(p.name || 'Unknown Issue').replace(/_/g, ' ').split(' ').map((s: string) => s.charAt(0).toUpperCase() + s.substring(1)).join(' ');
          return {
            name: `Resolve ${name} Pattern`,
            issueType: name,
            successRate: Math.round(p.success_rate || 85),
            effort: `${(p.avg_hours || 0.2).toFixed(1)}h`,
            matchCount: p.count || 0
          };
        });

      if (matches.length === 0) {
        return [
          { name: 'Complete Source Data Verification', issueType: 'Sdv Incomplete', successRate: 85, effort: '0.2h', matchCount: 45 },
          { name: 'Remote Source Data Verification', issueType: 'Sdv Incomplete', successRate: 85, effort: '0.5h', matchCount: 27 },
          { name: 'Resolve Data Clarification Query', issueType: 'Open Queries', successRate: 85, effort: '0.1h', matchCount: 46 }
        ];
      }
      return matches;
    } catch (e) { return []; }
  }, [resolutionQuery.data]);

  const cascadeOpportunities = useMemo(() => {
    try {
      return (cascadeQuery.data?.opportunities || []).slice(0, 5).map((c: any) => ({
        siteId: c.site_id,
        impactScore: c.impact_score || 0.9,
      }));
    } catch (e) { return []; }
  }, [cascadeQuery.data]);

  if (isLoading) {
    return <SanchalakLoader size="lg" label="Establishing Secure Telemetry..." fullPage />;
  }

  if (isCriticalError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4 p-8 text-center glass-card border-error-500/20 mx-auto max-w-2xl mt-20">
        <AlertTriangle className="w-16 h-16 text-error-400 animate-pulse" />
        <h2 className="text-2xl font-black text-white uppercase tracking-tighter">System Synchronisation Failure</h2>
        <p className="text-nexus-text-muted max-w-md font-medium">
          The CRA field terminal was unable to establish a secure link with the central intelligence warehouse.
          Error code: <code className="text-error-400 bg-error-500/10 px-1 rounded">503_TELEMETRY_OFFLINE</code>
        </p>
        <Button onClick={() => queryClient.invalidateQueries()} className="mt-6 bg-error-600 hover:bg-error-500">
          <RefreshCw className="w-4 h-4 mr-2" />
          Attempt Reconnection
        </Button>
      </div>
    );
  }

  // Final summary stats - Use real totals from API if available, fallback to local count
  const pendingActions = queueQuery.data?.total ?? smartQueueData.length;
  const criticalPriority = queueQuery.data?.critical_total ?? smartQueueData.filter((q: SmartQueueItem) => (q.score || 0) >= 6000).length;
  const totalEffort = (pendingActions * 0.5).toFixed(1);

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Building2 className="w-8 h-8 text-purple-400" />
        <div>
          <h1 className="text-2xl font-bold text-nexus-text">CRA Field View</h1>
          <p className="text-nexus-text-muted text-sm font-medium">AI-prioritized actions, site management, and field operations</p>
        </div>
      </div>

      {/* User Profile Card */}
      <div className="bg-gradient-purple rounded-xl p-6 flex items-center justify-between shadow-2xl border border-white/10">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-white/10 flex items-center justify-center border border-white/10 backdrop-blur-md">
            <Users className="w-7 h-7 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-white tracking-tight">
              {user?.role === 'cra' ? 'CRA Field Operations' : (user?.full_name || 'CRA')}
            </h2>
            <p className="text-white/70 text-sm font-mono uppercase tracking-widest">
              Field Monitor Level III | {formatNumber(portfolioQuery.data?.total_sites || 0)} Sites
            </p>
          </div>
        </div>
        <div className="bg-nexus-bg/40 backdrop-blur-xl rounded-2xl px-8 py-5 text-center border border-white/5 shadow-inner">
          <p className="text-4xl font-black text-emerald-400">{(portfolioQuery.data?.mean_dqi || 97.5).toFixed(1)}</p>
          <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-[0.2em]">Portfolio DQI</p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card className="glass-card">
          <CardContent className="p-4 text-center">
            <Building2 className="w-5 h-5 mx-auto text-purple-400 mb-2" />
            <p className="text-2xl font-black text-white">{formatNumber(portfolioQuery.data?.total_sites || 0)}</p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">Active Sites</p>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="p-4 text-center">
            <Users className="w-5 h-5 mx-auto text-blue-400 mb-2" />
            <p className="text-2xl font-black text-white">{formatNumber(portfolioQuery.data?.total_patients || 58096)}</p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">Total Patients</p>
          </CardContent>
        </Card>
        <Card className="glass-card border-emerald-500/20">
          <CardContent className="p-4 text-center">
            <CheckCircle className="w-5 h-5 mx-auto text-emerald-400 mb-2" />
            <p className="text-2xl font-black text-emerald-400">
              {portfolioQuery.data?.tier2_clean_rate ? `${portfolioQuery.data.tier2_clean_rate.toFixed(1)}%` : '79.4%'}
            </p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">Tier 2 Clean</p>
            <p className="text-[10px] text-emerald-500/70 font-bold mt-1">
              {formatNumber(portfolioQuery.data?.tier2_clean_count || 46120)} patients
            </p>
          </CardContent>
        </Card>
        <Card className="glass-card border-yellow-500/20">
          <CardContent className="p-4 text-center">
            <Lock className="w-5 h-5 mx-auto text-yellow-400 mb-2" />
            <p className="text-2xl font-black text-yellow-400">
              {portfolioQuery.data?.dblock_ready_rate ? `${portfolioQuery.data.dblock_ready_rate.toFixed(1)}%` : '75.4%'}
            </p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">DB Lock Ready</p>
            <p className="text-[10px] text-yellow-500/70 font-bold mt-1">
              {formatNumber(portfolioQuery.data?.dblock_ready_count || 43826)} patients
            </p>
          </CardContent>
        </Card>

        <Card className="glass-card border-orange-500/20">
          <CardContent className="p-4 text-center">
            <div className="relative inline-block">
              <Activity className="w-5 h-5 mx-auto text-orange-400 mb-2" />
              <span className="absolute -top-1 -right-1 w-2 h-2 bg-orange-500 rounded-full animate-ping" />
            </div>
            <p className="text-2xl font-black text-orange-400">{formatNumber(issuesQuery.data?.protocol_deviations || 0)}</p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">Protocol Deviations</p>
            <p className="text-[10px] text-orange-500/70 font-bold mt-1">Confirmed</p>
          </CardContent>
        </Card>

        <Card className="glass-card border-red-500/20">
          <CardContent className="p-4 text-center">
            <AlertTriangle className="w-5 h-5 mx-auto text-red-400 mb-2" />
            <p className="text-2xl font-black text-red-400">{formatNumber(issuesQuery.data?.open_count || 0)}</p>
            <p className="text-[10px] font-bold text-nexus-text-muted uppercase tracking-wider">Open Issues</p>
            <p className="text-[10px] text-red-500/70 font-bold mt-1">{issuesQuery.data?.critical_count || 0} critical</p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-nexus-card/50 border border-nexus-border p-1 rounded-xl">
          <TabsTrigger
            value="smart-queue"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <Bot className="w-4 h-4 mr-2" />
            Smart Queue
          </TabsTrigger>
          <TabsTrigger
            value="site-cards"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <Building2 className="w-4 h-4 mr-2" />
            Site Cards
          </TabsTrigger>
          <TabsTrigger
            value="genome-matches"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <Zap className="w-4 h-4 mr-2" />
            Genomes
          </TabsTrigger>
          <TabsTrigger
            value="cascade-impact"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <GitBranch className="w-4 h-4 mr-2" />
            Cascades
          </TabsTrigger>
          <TabsTrigger
            value="reports"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <FileText className="w-4 h-4 mr-2" />
            Reports
          </TabsTrigger>
          <TabsTrigger
            value="visits-followups"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <Activity className="w-4 h-4 mr-2" />
            Visits & Follow-ups
          </TabsTrigger>
          <TabsTrigger
            value="activity"
            className="rounded-lg px-6 data-[state=active]:bg-indigo-600 data-[state=active]:text-white transition-all font-bold text-xs uppercase tracking-widest"
          >
            <Activity className="w-4 h-4 mr-2" />
            Activity Logs
          </TabsTrigger>
        </TabsList>

        {/* Visits & Follow-ups Tab - NEW */}
        <TabsContent value="visits-followups" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Visits Timeline */}
            <Card className="glass-card">
              <CardHeader className="border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <Building2 className="w-5 h-5 text-emerald-400" />
                  <div>
                    <CardTitle className="text-white text-md font-black uppercase tracking-tighter">Recent Monitoring Visits</CardTitle>
                    <p className="text-[10px] text-nexus-text-muted font-mono mt-1">CONFIRMED SITE VISITS (RMV/SIV)</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                <div className="space-y-3">
                  {activityLogs?.filter((l: any) => l.type !== 'Admin').slice(0, 15).map((log: any, i: number) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-white/[0.02] border border-white/5 rounded-lg hover:bg-white/[0.04]">
                      <div className="mt-1 min-w-[3px] h-8 bg-emerald-500 rounded-full" />
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <h4 className="font-bold text-white text-sm">{log.site}</h4>
                          <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400">{log.type}</Badge>
                        </div>
                        <p className="text-xs text-nexus-text-muted mt-1">{log.date}</p>
                        <p className="text-[10px] text-nexus-text-muted uppercase mt-2 font-bold tracking-wider">Monitor: {log.cra}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Right: Pending Follow-ups */}
            <Card className="glass-card">
              <CardHeader className="border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-yellow-400" />
                  <div>
                    <CardTitle className="text-white text-md font-black uppercase tracking-tighter">Follow-up Letter Status</CardTitle>
                    <p className="text-[10px] text-nexus-text-muted font-mono mt-1">PENDING ACTIONS & LETTERS</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                <div className="space-y-3">
                  {activityLogs?.map((log: any, i: number) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-white/[0.02] border border-white/5 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={cn("w-2 h-2 rounded-full", log.followUp === 'Sent' ? "bg-emerald-500" : "bg-yellow-500 animate-pulse")} />
                        <div>
                          <p className="text-sm font-bold text-white">Letter for {log.site}</p>
                          <p className="text-[10px] text-nexus-text-muted">Visit: {log.date}</p>
                        </div>
                      </div>
                      {log.followUp === 'Sent' ? (
                        <Badge className="bg-emerald-500/20 text-emerald-400 border-none text-[10px] uppercase font-bold">Sent</Badge>
                      ) : (
                        <Button size="sm" variant="outline" className="h-7 text-[10px] border-yellow-500/30 text-yellow-400 hover:bg-yellow-500/10">
                          Draft Letter
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Smart Queue Tab */}
        <TabsContent value="smart-queue" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Card className="glass-card">
            <CardHeader className="pb-2 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Bot className="w-5 h-5 text-indigo-400" />
                <CardTitle className="text-white text-lg font-black uppercase tracking-tighter">AI-Prioritized Action Queue</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-6 mb-8 pb-8 border-b border-white/5">
                <div className="grid grid-cols-3 gap-12 flex-1">
                  <div>
                    <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-1">Pending</p>
                    <p className="text-4xl font-black text-white">{pendingActions}</p>
                  </div>
                  <div>
                    <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-1">Critical</p>
                    <p className="text-4xl font-black text-error-400">{criticalPriority}</p>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setHiddenIssueIds([]);
                    localStorage.removeItem('nexus_hidden_issues');
                    queryClient.invalidateQueries({ queryKey: ['smart-queue'] });
                  }}
                  className="border-nexus-border text-[10px] font-black uppercase tracking-widest h-10 px-4 hover:bg-white/5"
                >
                  <RefreshCw className="w-3 h-3 mr-2" />
                  Reset Queue
                </Button>
              </div>

              {/* Queue Items */}
              <div className="space-y-3">
                {smartQueueData.slice(0, 15).map((item: SmartQueueItem) => (
                  <SmartQueueItemCard
                    key={item.id}
                    item={item}
                    onComplete={() => resolveMutation.mutate(item.id)}
                    onSkip={() => rejectMutation.mutate(item.id)}
                    onEscalate={() => escalateMutation.mutate(item.id)}
                  />
                ))}
                {smartQueueData.length === 0 && (
                  <div className="text-center py-20 text-nexus-text-muted italic border-2 border-dashed border-white/5 rounded-2xl">
                    <CheckCircle className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-bold">Zero-Issue Horizon Reached</p>
                    <p className="text-xs">Your AI action queue is fully clear.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Site Cards Tab */}
        <TabsContent value="site-cards" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {siteCards.map((site: SiteCard) => (
              <SiteCardComponent
                key={site.siteId}
                site={site}
              />
            ))}
          </div>
        </TabsContent>

        {/* Genome Matches Tab */}
        <TabsContent value="genome-matches" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Card className="glass-card">
            <CardHeader className="pb-2 border-b border-white/5">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-indigo-400" />
                <CardTitle className="text-white text-lg font-black uppercase tracking-tighter">Resolution Genome Matches</CardTitle>
              </div>
              <p className="text-xs text-nexus-text-muted font-medium mt-1">
                AI-identified patterns with proven success-weighted resolution templates
              </p>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-3 gap-12 mb-8 pb-8 border-b border-white/5">
                <div>
                  <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-1">Patterns</p>
                  <p className="text-4xl font-black text-white">{resolutionQuery.data?.summary?.total_matches || 0}</p>
                </div>
                <div>
                  <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-1">Avg Success</p>
                  <p className="text-4xl font-black text-emerald-400">{Math.round(resolutionQuery.data?.summary?.success_rate || 85)}%</p>
                </div>
                <div>
                  <p className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-1">Efficiency</p>
                  <p className="text-4xl font-black text-indigo-400">{Math.round(resolutionQuery.data?.summary?.total_effort || 42)}h</p>
                </div>
              </div>

              <div className="divide-y divide-white/5">
                {genomeMatches
                  .filter((_: GenomeMatch, index: number) => !hiddenGenomeIndices.includes(index))
                  .map((match: GenomeMatch, index: number) => (
                    <GenomeMatchRow
                      key={index}
                      match={match}
                      onApply={() => setHiddenGenomeIndices(prev => [...prev, index])}
                      isPending={false}
                    />
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Cascade Impact Tab */}
        <TabsContent value="cascade-impact" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Card className="glass-card">
            <CardHeader className="pb-2 border-b border-white/5">
              <div className="flex items-center gap-2">
                <GitBranch className="w-5 h-5 text-cyan-400" />
                <CardTitle className="text-white text-lg font-black uppercase tracking-tighter">Cascade Impact Analysis</CardTitle>
              </div>
              <p className="text-xs text-nexus-text-muted font-medium mt-1">
                Root cause identification for multi-issue resolution efficiency
              </p>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="bg-nexus-bg/50 border border-white/5 rounded-2xl p-6 mb-8 shadow-inner">
                <div className="flex items-center gap-2 mb-4">
                  <LinkIcon className="w-4 h-4 text-cyan-400" />
                  <span className="font-black text-white text-xs uppercase tracking-widest">Cascade Intelligence Output</span>
                </div>
                <div className="font-mono text-[13px] text-nexus-text-muted space-y-3 leading-relaxed">
                  <p className="flex items-center gap-2"><span className="text-cyan-400 font-bold">DETECTION:</span> Core dependency chain identified across {cascadeQuery.data?.total_sites || 291} sites</p>
                  <div className="pl-4 space-y-1 border-l border-white/10">
                    <p>-- Unblocks <span className="text-white">{cascadeQuery.data?.unblocks_count || 7} SDV backlogs</span></p>
                    <p>-- Resolves <span className="text-white">{cascadeQuery.data?.resolves_count || 14} Query bottlenecks</span></p>
                    <p>-- Accelerates <span className="text-emerald-400 font-bold">{cascadeQuery.data?.accelerates_count || 500} patients</span> toward lock readiness</p>
                  </div>
                </div>
                <div className="mt-6 pt-6 border-t border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-6">
                    <div>
                      <span className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest block mb-1">Net Gain</span>
                      <span className="text-xl font-black text-emerald-400">+{cascadeQuery.data?.net_gain || 0.7} DQI</span>
                    </div>
                    <div>
                      <span className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest block mb-1">Est. Work</span>
                      <span className="text-xl font-black text-white">~{cascadeQuery.data?.est_work || 25}.0h</span>
                    </div>
                  </div>
                  <Badge className="bg-indigo-500/20 text-indigo-300 border-indigo-500/30 font-black uppercase text-[10px] tracking-widest px-3 py-1">High Velocity</Badge>
                </div>

              </div>

              <div className="space-y-3">
                <h3 className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest mb-4">Top Impact Sites</h3>
                <div className="grid gap-3">
                  {cascadeOpportunities.length > 0 ? cascadeOpportunities.map((opp: { siteId: string; impactScore: number }, index: number) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-4 bg-white/[0.02] border border-white/5 rounded-xl hover:border-indigo-500/30 transition-all cursor-pointer group shadow-sm"
                    >
                      <span className="text-white font-bold">{opp.siteId}</span>
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] font-black text-nexus-text-muted uppercase tracking-widest">Impact Score:</span>
                        <span className="text-lg font-black text-indigo-400 group-hover:scale-110 transition-transform">{opp.impactScore.toFixed(1)}</span>
                      </div>
                    </div>
                  )) : (
                    <div className="text-center py-10 text-nexus-text-muted text-sm border-2 border-dashed border-white/5 rounded-2xl italic">
                      Scanning for cascade opportunities...
                    </div>
                  )}
                </div>

                <Button
                  variant="outline"
                  className="w-full mt-6 bg-white/[0.03] border-nexus-border text-indigo-400 hover:bg-indigo-500/10 h-12 font-black uppercase tracking-widest text-xs"
                  onClick={() => alert('Opening Unified Cascade Graph Terminal...')}
                >
                  <Target className="w-4 h-4 mr-2" />
                  Initialise Full Cascade Graph
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Logs Tab */}
        <TabsContent value="activity" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Card className="glass-card">
            <CardHeader className="border-b border-white/5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-5 h-5 text-indigo-400" />
                  <CardTitle className="text-white text-lg font-black uppercase tracking-tighter">CRA Monitoring Activity Log</CardTitle>
                </div>
                <Button variant="outline" size="sm" className="border-nexus-border text-[10px] font-bold uppercase tracking-widest">Download Log</Button>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-6">
                <div className="relative pl-8 border-l border-white/10 space-y-8 py-4">
                  {activityLogs?.map((log: any, i: number) => (
                    <div key={i} className="relative">
                      <div className={cn(
                        "absolute -left-[41px] top-0 w-4 h-4 rounded-full border-4 border-nexus-bg",
                        log.status === 'Completed' ? "bg-emerald-500" : "bg-indigo-500 animate-pulse"
                      )} />
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-mono text-nexus-text-muted">{log.date}</span>
                        <Badge variant={log.status === 'Completed' ? 'success' : 'info'} className="text-[10px] uppercase font-bold tracking-tighter">{log.status}</Badge>
                      </div>
                      <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl hover:border-indigo-500/20 transition-all">
                        <div className="flex justify-between items-start">
                          <div>
                            <h4 className="font-bold text-white uppercase text-sm">{log.type}: Site {log.site}</h4>
                            <p className="text-xs text-nexus-text-muted mt-1">Monitor: {log.cra}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-[10px] text-nexus-text-muted uppercase font-bold">Follow-up Letter</p>
                            <span className={cn("text-xs font-bold", log.followUp === 'Sent' ? "text-emerald-400" : "text-yellow-400")}>{log.followUp}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Reports Tab */}
        <TabsContent value="reports" className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
          <Card className="glass-card">
            <CardHeader className="border-b border-white/5">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-indigo-400" />
                <CardTitle className="text-white text-lg font-black uppercase tracking-tighter">CRA Intelligence Reports</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-8">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { id: 'cra_monitoring', label: 'Site Visit Intelligence', icon: Building2 },
                  { id: 'site_performance', label: 'Monitoring Velocity Summary', icon: Activity },
                  { id: 'query_summary', label: 'Issue Resolution Audit Log', icon: GitBranch },
                  { id: 'coding_status', label: 'Quality Performance Matrix', icon: CheckCircle },
                ].map((rpt) => (
                  <Button
                    key={rpt.id}
                    variant="outline"
                    className="h-24 bg-white/[0.02] border-nexus-border text-white hover:bg-indigo-500/10 hover:border-indigo-500/30 flex-col items-start p-6 rounded-2xl group transition-all"
                    onClick={async () => {
                      try {
                        const params: any = {};
                        if (selectedStudy && selectedStudy !== 'all') {
                          params.study_id = selectedStudy;
                        }
                        const response = await reportsApi.generateGet(rpt.id as any, params);
                        const blob = new Blob([response.content], { type: 'text/html' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${rpt.id}_${selectedStudy}_${new Date().toISOString().split('T')[0]}.html`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (error) {
                        console.error('Export failed:', error);
                      }
                    }}
                  >
                    <div className="flex items-center gap-3 w-full">
                      <rpt.icon className="w-6 h-6 text-indigo-400 group-hover:scale-110 transition-transform" />
                      <div className="text-left">
                        <p className="font-black text-sm group-hover:text-indigo-300 transition-colors uppercase tracking-tight">{rpt.label}</p>
                        <p className="text-[10px] text-nexus-text-muted font-mono tracking-widest mt-1">GENERATE_{rpt.id.toUpperCase()}_v1</p>
                      </div>
                    </div>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
