import { useState, useMemo, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { sitesApi, studiesApi } from '@/services/api';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { useAppStore } from '@/stores/appStore';
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
  Building2,
  Users,
  Activity,
  FileText,
  Download,
  Calendar,
  MessageSquare,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Play,
  ChevronRight,
  TrendingUp,
  Home,
  Phone,
  Mail,
  Search,
  RefreshCw,
  Plus,
  Send,
  Trash2,
  Star,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { cn } from '@/lib/utils';

interface Site {
  site_id: string;
  name: string;
  principal_investigator?: string;
  coordinator_name?: string;
  status?: string;
  dqi_score?: number;
  patient_count?: number;
}

interface ActionItem {
  id: string;
  title: string;
  description: string;
  priority: string;
  due: string;
  category: string;
  dqi_impact: string | number;
  status: string;
}

interface Message {
  id: string;
  from: string;
  role: string;
  subject: string;
  body: string;
  date: string;
  unread: boolean;
  starred: boolean;
}

interface DQIIDIssue {
  id: string;
  name: string;
  count: number;
  dqi_impact: number;
  effort: string;
  component: string;
  priority: string;
}

export default function SitePortal() {

  const [selectedIssues, setSelectedIssues] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const { selectedStudy } = useAppStore();
  const [selectedSiteId, setSelectedSiteId] = useState<string>('all');
  const [completedTaskIds, setCompletedTaskIds] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const { data: sitesResponse, isLoading: sitesLoading } = useQuery({
    queryKey: ['sites', selectedStudy],
    queryFn: () => sitesApi.list({ study_id: selectedStudy }),
  });

  // Derived data - filter out mock sites
  const sites = useMemo(() =>
    (sitesResponse?.sites || []).filter((s: Site) => !s.site_id.startsWith('Site_'))
    , [sitesResponse]);


  // Set initial selected site if empty
  useEffect(() => {
    if (!selectedSiteId) setSelectedSiteId('all');
  }, [sites, selectedSiteId]);

  const { data: siteDetails } = useQuery({
    queryKey: ['site-details', selectedSiteId],
    queryFn: () => sitesApi.get(selectedSiteId),
    enabled: !!selectedSiteId && selectedSiteId !== 'all',
  });

  const { data: portalData } = useQuery({
    queryKey: ['site-portal-data', selectedSiteId],
    queryFn: () => sitesApi.getPortalData(selectedSiteId),
    enabled: !!selectedSiteId,
  });

  const { data: benchmarksResponse } = useQuery({
    queryKey: ['site-benchmarks', selectedStudy],
    queryFn: () => sitesApi.getBenchmarks(selectedStudy),
  });

  // Fetch DQI issues for the simulator
  const { data: dqiIssuesResponse } = useQuery({
    queryKey: ['dqi-issues', selectedSiteId],
    queryFn: () => sitesApi.getDQIIssues(selectedSiteId),
    enabled: !!selectedSiteId,
  });

  // Action plan mutation
  const createActionPlanMutation = useMutation({
    mutationFn: (issueIds: string[]) => sitesApi.createActionPlan(selectedSiteId, issueIds),
    onSuccess: (data) => {
      alert(`Action Plan Created!\n\nTotal Effort: ${data.total_effort_hours} hours\nProjected DQI Gain: +${data.projected_dqi_gain}%\n\n${data.actions.length} actions generated.`);
    },
    onError: (error: any) => {
      alert(`Failed to create action plan: ${error.message || 'Unknown error'}`);
    },
  });

  const sitePortal = portalData || {};
  const metrics = sitePortal.metrics || {};

  const currentSite = {
    site_id: selectedSiteId,
    name: sitePortal.name || siteDetails?.name || selectedSiteId,
    pi_name: sitePortal.pi_name || siteDetails?.principal_investigator || 'N/A',
    coordinator_name: sitePortal.coordinator_name || siteDetails?.coordinator_name || 'N/A',
    cra_name: sitePortal.coordinator_name || siteDetails?.coordinator_name || 'N/A',
    cra_email: siteDetails?.coordinator_email || siteDetails?.pi_email || '',
    cra_phone: '',
    dqi_score: metrics.dqi_score ?? 0.0,
    clean_rate: metrics.clean_rate ?? 0.0,
    db_lock_ready: metrics.db_lock_ready ?? 0.0,
    open_issues: metrics.open_issues ?? 0,
    patients_enrolled: metrics.enrolled ?? 0,
    target_enrollment: metrics.target ?? 0,
  };

  // Action items derived from API
  const actionItems = useMemo(() => {
    const rawItems = sitePortal.action_items || [];
    return rawItems.map((item: ActionItem) => ({
      ...item,
      status: completedTaskIds.includes(item.id) ? 'completed' : item.status
    }));
  }, [sitePortal, completedTaskIds]);

  // Overall Progress
  const completionProgress = useMemo(() => sitePortal.completion_progress || {
    data_entry: 0,
    query_resolution: 0,
    medical_coding: 0,
    sae_processing: 0
  }, [sitePortal]);

  // Messages - from portal data or default
  const [messages, setMessages] = useState<Array<{ id: string; from: string; role: string; subject: string; body: string; date: string; unread: boolean; starred: boolean }>>([]);

  // Initialize messages from portal data or use defaults
  useEffect(() => {
    const portalMessages = portalData?.messages || [];
    if (portalMessages.length > 0) {
      setMessages(portalMessages.map((m: Message, idx: number) => ({
        id: m.id || String(idx + 1),
        from: m.from || 'Study Team',
        role: m.role || 'Global',
        subject: m.subject || 'No Subject',
        body: m.body || '',
        date: m.date || 'Recently',
        unread: m.unread ?? false,
        starred: m.starred ?? false,
      })));
    }
  }, [portalData]);

  // DQI Simulator data - from real API or empty
  const dqiIssues = useMemo(() => {
    return dqiIssuesResponse?.issues || [];
  }, [dqiIssuesResponse]);

  // Current DQI from API or fallback
  const currentDQI = dqiIssuesResponse?.current_dqi || currentSite.dqi_score;

  const totalSelectedImpact = useMemo(() =>
    dqiIssues.filter((i: any) => selectedIssues.includes(i.id)).reduce((sum: number, i: any) => sum + i.dqi_impact, 0)
    , [selectedIssues, dqiIssues]);

  const totalSelectedEffort = useMemo(() => {
    const totalMinutes = dqiIssues
      .filter((i: any) => selectedIssues.includes(i.id))
      .reduce((sum: number, i: any) => sum + parseFloat(i.effort) * 60, 0);
    const hours = Math.floor(totalMinutes / 60);
    const mins = Math.round(totalMinutes % 60);
    return `${hours} hours ${mins} minutes`;
  }, [selectedIssues, dqiIssues]);

  const projectedDQI = Math.min(100, currentDQI + totalSelectedImpact);

  // Chart data
  const sitePerformance = (benchmarksResponse?.benchmarks || [])
    .filter((s: any) => !s.site_id.startsWith('Site_'))
    .slice(0, 10).map((site: any) => ({
      name: site.site_id,
      dqi: Math.round(site.dqi_score * 10) / 10,
      patients: site.patient_count,
    }));

  const toggleIssue = (issueId: string) => {
    setSelectedIssues(prev =>
      prev.includes(issueId)
        ? prev.filter(id => id !== issueId)
        : [...prev, issueId]
    );
  };

  const handleAction = (id: number | string, action: string) => {
    setCompletedTaskIds(prev => [...prev, id.toString()]);
    alert(`${action} item ${id} marked as completed.`);
  };

  const markMessageRead = (id: string) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, unread: false } : m));
  };

  const { data: studyDetails } = useQuery({
    queryKey: ['study-details', sitePortal.study_id],
    queryFn: () => studiesApi.get(sitePortal.study_id),
    enabled: !!sitePortal.study_id,
  });

  const targetDate = useMemo(() => {
    if (studyDetails?.end_date) {
      return new Date(studyDetails.end_date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
    }
    return null;
  }, [studyDetails]);

  if (sitesLoading) {
    return <SanchalakLoader size="lg" label="Loading site portal..." fullPage />;
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Header with Site Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Building2 className="w-8 h-8 text-teal-400" />
          <div>
            <h1 className="text-2xl font-bold text-white">Site Portal</h1>
            <p className="text-nexus-text-secondary text-sm">Operational management for clinical sites</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Select value={selectedSiteId} onValueChange={setSelectedSiteId}>
            <SelectTrigger className="w-64 bg-nexus-card border-nexus-border text-white">
              <SelectValue placeholder="Select Site" />
            </SelectTrigger>
            <SelectContent className="bg-nexus-card border-nexus-border text-white">
              <SelectItem key="all" value="all">
                Global Portfolio
              </SelectItem>
              {sites.map((site: any) => (
                <SelectItem key={site.site_id} value={site.site_id}>
                  {site.site_id !== site.name ? `${site.site_id} - ${site.name}` : site.site_id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon"
            className="bg-nexus-card border-nexus-border text-nexus-text-secondary hover:text-white"
            onClick={() => queryClient.invalidateQueries()}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Welcome Banner */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border bg-gradient-to-r from-teal-600/20 to-cyan-600/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
              <Home className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">{currentSite.name}</h1>
              <div className="flex items-center gap-2 text-nexus-text-secondary mt-1">
                <Badge variant="outline" className="bg-teal-500/10 text-teal-400 border-teal-500/20">
                  Site ID: {currentSite.site_id}
                </Badge>
                <span className="text-xs">•</span>
                <span className="text-xs">{selectedStudy || 'All Studies'}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-5xl font-black text-white leading-none">{currentSite.dqi_score.toFixed(1)}%</p>
              <p className="text-sm text-nexus-text-secondary mt-1 font-medium">Site DQI Score</p>
            </div>
            <div className={`w-20 h-20 rounded-full border-4 flex items-center justify-center shadow-lg ${currentSite.dqi_score >= 85 ? 'border-success-400 bg-success-500/10 shadow-success-500/10' :
              currentSite.dqi_score >= 70 ? 'border-warning-400 bg-warning-500/10 shadow-warning-500/10' :
                'border-error-400 bg-error-500/10 shadow-error-500/10'
              }`}>
              {currentSite.dqi_score >= 85 ? (
                <CheckCircle2 className="w-10 h-10 text-success-400" />
              ) : (
                <AlertTriangle className={`w-10 h-10 ${currentSite.dqi_score >= 70 ? 'text-warning-400' : 'text-error-400'}`} />
              )}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Button className="bg-teal-600 hover:bg-teal-700 text-white" onClick={() => alert('Opening site report...')}>
            <FileText className="w-4 h-4 mr-2" />
            View Report
          </Button>
          <Button variant="outline" className="border-nexus-border text-white hover:bg-nexus-card" onClick={() => alert('Exporting site data...')}>
            <Download className="w-4 h-4 mr-2" />
            Export Data
          </Button>
          <Button variant="outline" className="border-nexus-border text-white hover:bg-nexus-card" onClick={() => alert('Scheduling monitoring visit...')}>
            <Calendar className="w-4 h-4 mr-2" />
            Schedule Visit
          </Button>
          <Button variant="outline" className="border-nexus-border text-white hover:bg-nexus-card" onClick={() => alert('Contacting CRA...')}>
            <MessageSquare className="w-4 h-4 mr-2" />
            Contact CRA
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="bg-nexus-card border-nexus-border hover:border-success-500/30 transition-colors">
          <CardContent className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{Number(currentSite.clean_rate).toFixed(1)}%</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Clean Patients</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <CheckCircle2 className="w-6 h-6 text-success-400" />
              </div>
            </div>
            <div className="mt-4">
              <Progress value={currentSite.clean_rate} className="h-1.5 bg-success-500/10" indicatorClassName="bg-success-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-nexus-card border-nexus-border hover:border-info-500/30 transition-colors">
          <CardContent className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{Number(currentSite.db_lock_ready).toFixed(1)}%</p>
                <p className="text-sm text-nexus-text-secondary mt-1">DB Lock Ready</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-info-500/20 flex items-center justify-center">
                <Activity className="w-6 h-6 text-info-400" />
              </div>
            </div>
            <div className="mt-4">
              <Progress value={currentSite.db_lock_ready} className="h-1.5 bg-info-500/10" indicatorClassName="bg-info-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-nexus-card border-nexus-border hover:border-warning-500/30 transition-colors">
          <CardContent className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{currentSite.open_issues}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Open Issues</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-warning-400" />
              </div>
            </div>
            <p className="text-xs text-nexus-text-secondary mt-4 flex items-center gap-1">
              <TrendingUp className="w-3 h-3 text-error-400" />
              +2 since last week
            </p>
          </CardContent>
        </Card>

        <Card className="bg-nexus-card border-nexus-border hover:border-purple-500/30 transition-colors">
          <CardContent className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{currentSite.patients_enrolled}/{currentSite.target_enrollment}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Enrolled</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Users className="w-6 h-6 text-purple-400" />
              </div>
            </div>
            <div className="mt-4">
              <Progress value={(currentSite.patients_enrolled / currentSite.target_enrollment) * 100} className="h-1.5 bg-purple-500/10" indicatorClassName="bg-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Overall Progress */}
      <Card className="glass-card border-nexus-border">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white">Overall Completion Progress</h3>
              <p className="text-sm text-nexus-text-secondary">Tracking towards database lock</p>
            </div>
            {targetDate && (
              <Badge variant="info" className="text-sm">
                Target: {targetDate}
              </Badge>
            )}
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-white">Data Entry</span>
                <span className="text-sm text-success-400">{completionProgress.data_entry}%</span>
              </div>
              <Progress value={completionProgress.data_entry} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-white">Query Resolution</span>
                <span className="text-sm text-warning-400">{completionProgress.query_resolution}%</span>
              </div>
              <Progress value={completionProgress.query_resolution} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-white">Medical Coding</span>
                <span className="text-sm text-info-400">{completionProgress.medical_coding}%</span>
              </div>
              <Progress value={completionProgress.medical_coding} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm text-white">SAE Processing</span>
                <span className="text-sm text-success-400">{completionProgress.sae_processing}%</span>
              </div>
              <Progress value={completionProgress.sae_processing} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs defaultValue="actions" className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border p-1">
          <TabsTrigger value="actions" className="data-[state=active]:bg-teal-600 data-[state=active]:text-white">
            Action Items
          </TabsTrigger>
          <TabsTrigger value="messages" className="data-[state=active]:bg-teal-600 data-[state=active]:text-white">
            Messages
            {messages.filter(m => m.unread).length > 0 && (
              <Badge className="ml-2 bg-error-500 text-white border-0 h-5 w-5 p-0 flex items-center justify-center rounded-full text-[10px]">
                {messages.filter(m => m.unread).length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="help" className="data-[state=active]:bg-teal-600 data-[state=active]:text-white">Help & Support</TabsTrigger>
          <TabsTrigger value="simulator" className="data-[state=active]:bg-teal-600 data-[state=active]:text-white">DQI Simulator</TabsTrigger>
          <TabsTrigger value="performance" className="data-[state=active]:bg-teal-600 data-[state=active]:text-white">Performance</TabsTrigger>
        </TabsList>

        {/* Action Items Tab */}
        <TabsContent value="actions">
          <Card className="bg-nexus-card border-nexus-border overflow-hidden">
            <CardHeader className="border-b border-nexus-border pb-4 bg-nexus-card/50">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white">Tasks & Actions</CardTitle>
                  <CardDescription className="text-nexus-text-secondary mt-1">Prioritized items requiring site attention</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="error" className="bg-error-500/10 text-error-400 border-error-500/20">
                    {actionItems.filter((a: any) => a.priority === 'Critical').length} Critical
                  </Badge>
                  <Badge variant="warning" className="bg-warning-500/10 text-warning-400 border-warning-500/20">
                    {actionItems.filter((a: any) => a.priority === 'High').length} High
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-nexus-border">
                {actionItems.map((item: any) => (
                  <div
                    key={item.id}
                    className={cn(
                      "p-5 hover:bg-nexus-card-hover transition-colors flex items-start justify-between gap-4",
                      item.status === 'completed' && "opacity-60 grayscale-[0.5]"
                    )}
                  >
                    <div className="flex items-start gap-4 flex-1">
                      <div className={cn(
                        "mt-1 p-2 rounded-lg",
                        item.status === 'completed' ? "bg-success-500/10 text-success-400" :
                          item.priority === 'Critical' ? "bg-error-500/10 text-error-400" :
                            item.priority === 'High' ? "bg-warning-500/10 text-warning-400" :
                              "bg-info-500/10 text-info-400"
                      )}>
                        {item.status === 'completed' ? <CheckCircle2 className="w-5 h-5" /> :
                          item.priority === 'Critical' ? <AlertTriangle className="w-5 h-5" /> :
                            <Clock className="w-5 h-5" />}
                      </div>
                      <div className="space-y-1">
                        <p className={cn("font-semibold text-white", item.status === 'completed' && "line-through text-nexus-text-secondary")}>
                          {item.title}
                        </p>
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="text-xs text-nexus-text-secondary flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            Due: {item.due || item.due_date}
                          </span>
                          <Badge className={cn(
                            "text-[10px] h-5 px-2 font-bold uppercase tracking-wider border-0",
                            item.priority === 'Critical' ? "bg-error-500 text-white shadow-lg shadow-error-500/20" :
                              item.priority === 'High' ? "bg-warning-500 text-white shadow-lg shadow-warning-500/20" :
                                "bg-info-500 text-white"
                          )}>
                            {item.priority}
                          </Badge>
                          {parseFloat(item.dqi_impact) > 0 && (
                            <span className="text-xs text-success-400 font-medium flex items-center gap-1">
                              <TrendingUp className="w-3 h-3" />
                              +{item.dqi_impact}% DQI impact
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {item.status !== 'completed' ? (
                        <Button
                          size="sm"
                          className="bg-teal-600 hover:bg-teal-700 h-9 px-4 text-xs font-semibold"
                          onClick={() => handleAction(item.id, 'Task')}
                        >
                          {item.status === 'in_progress' ? 'Complete' : 'Start Task'}
                        </Button>
                      ) : (
                        <Badge variant="outline" className="bg-success-500/10 text-success-400 border-success-500/20">
                          Completed
                        </Badge>
                      )}
                      <Button variant="ghost" size="icon" className="text-nexus-text-secondary hover:text-white" onClick={() => alert('Viewing task context...')}>
                        <HelpCircle className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Messages Tab */}
        <TabsContent value="messages">
          <Card className="bg-nexus-card border-nexus-border overflow-hidden">
            <CardHeader className="border-b border-nexus-border flex flex-row items-center justify-between pb-4 bg-nexus-card/50">
              <div>
                <CardTitle className="text-white">Communication Hub</CardTitle>
                <CardDescription className="text-nexus-text-secondary mt-1">Direct messaging with study team and monitors</CardDescription>
              </div>
              <Button className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white border-0" onClick={() => alert('Composing new message...')}>
                <Plus className="w-4 h-4 mr-2" />
                New Message
              </Button>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-nexus-border">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={cn(
                      "p-4 hover:bg-nexus-card-hover transition-all cursor-pointer border-l-4",
                      msg.unread ? "bg-purple-500/5 border-purple-500" : "bg-transparent border-transparent"
                    )}
                    onClick={() => markMessageRead(msg.id)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-4">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm",
                          msg.role === 'CRA' ? "bg-teal-500/20 text-teal-400" :
                            msg.role === 'Safety' ? "bg-error-500/20 text-error-400" :
                              "bg-purple-500/20 text-purple-400"
                        )}>
                          {msg.from.split(' ').map(n => n[0]).join('').slice(0, 2)}
                        </div>
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className={cn("font-semibold", msg.unread ? "text-white" : "text-nexus-text-secondary")}>
                              {msg.from}
                            </span>
                            <Badge variant="outline" className="text-[10px] h-4 py-0 bg-nexus-card border-nexus-border text-nexus-text-secondary capitalize">
                              {msg.role}
                            </Badge>
                            {msg.unread && <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />}
                          </div>
                          <p className={cn("text-sm font-medium", msg.unread ? "text-white" : "text-nexus-text-secondary")}>
                            {msg.subject}
                          </p>
                          <p className="text-xs text-nexus-text-muted line-clamp-1">{msg.body}</p>
                        </div>
                      </div>
                      <div className="text-right space-y-2">
                        <p className="text-[10px] text-nexus-text-muted uppercase tracking-tighter">{msg.date}</p>
                        <div className="flex items-center justify-end gap-1">
                          <Button variant="ghost" size="icon" className={cn("h-7 w-7", msg.starred ? "text-yellow-400" : "text-nexus-text-muted")} onClick={(e) => { e.stopPropagation(); alert('Starred message'); }}>
                            <Star className={cn("w-3.5 h-3.5", msg.starred && "fill-current")} />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-nexus-text-muted hover:text-white" onClick={(e) => { e.stopPropagation(); alert('Replying...'); }}>
                            <Send className="w-3.5 h-3.5" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-nexus-text-muted hover:text-error-400" onClick={(e) => { e.stopPropagation(); alert('Archiving...'); }}>
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Help & Support Tab */}
        <TabsContent value="help">
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-nexus-card border-nexus-border shadow-xl">
              <CardHeader className="pb-2">
                <CardTitle className="text-white">Primary CRA Contact</CardTitle>
                <CardDescription className="text-nexus-text-secondary">Assigned Clinical Research Associate for your site</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="flex items-center gap-5 p-4 rounded-xl bg-nexus-card-hover border border-nexus-border">
                  <div className="w-16 h-16 rounded-full bg-gradient-to-br from-teal-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
                    <span className="text-2xl font-bold text-white">
                      {(
                        (currentSite.cra_name || '')
                          .split(' ')
                          .filter(Boolean)
                          .slice(0, 2)
                          .map(s => s[0])
                          .join('')
                          .toUpperCase() || 'NA'
                      )}
                    </span>
                  </div>
                  <div>
                    <p className="text-xl font-bold text-white">{currentSite.cra_name}</p>
                    <p className="text-sm text-teal-400 font-medium">Assigned CRA</p>
                  </div>
                </div>
                <div className="mt-6 space-y-3">
                  <Button
                    variant="outline"
                    className="w-full justify-between h-12 border-nexus-border bg-nexus-card/50 hover:bg-nexus-card text-white group"
                    disabled={!currentSite.cra_email}
                    onClick={() => currentSite.cra_email && (window.location.href = `mailto:${currentSite.cra_email}`)}
                  >
                    <div className="flex items-center">
                      <Mail className="w-4 h-4 mr-3 text-teal-400" />
                      <span className="text-sm font-medium">{currentSite.cra_email || 'No email on file'}</span>
                    </div>
                    <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-all" />
                  </Button>
                  {currentSite.cra_phone && (
                    <Button
                      variant="outline"
                      className="w-full justify-between h-12 border-nexus-border bg-nexus-card/50 hover:bg-nexus-card text-white group"
                      onClick={() => window.location.href = `tel:${currentSite.cra_phone}`}
                    >
                      <div className="flex items-center">
                        <Phone className="w-4 h-4 mr-3 text-success-400" />
                        <span className="text-sm font-medium">{currentSite.cra_phone}</span>
                      </div>
                      <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-all" />
                    </Button>
                  )}
                </div>
                <Button className="w-full mt-6 bg-teal-600 hover:bg-teal-700 text-white h-11 font-bold shadow-lg shadow-teal-600/20">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Send Instant Message
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-nexus-card border-nexus-border">
              <CardHeader className="pb-2">
                <CardTitle className="text-white">Resource Center</CardTitle>
                <CardDescription className="text-nexus-text-secondary">Documentation and support tools</CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-3">
                {[
                  { label: 'View Protocol Document', icon: FileText, color: 'text-purple-400' },
                  { label: 'EDC User Guide', icon: HelpCircle, color: 'text-info-400' },
                  { label: 'Report Technical Issue', icon: AlertTriangle, color: 'text-error-400' },
                  { label: 'Request Training Session', icon: Calendar, color: 'text-warning-400' },
                  { label: 'Site Pharmacy Manual', icon: Building2, color: 'text-teal-400' },
                ].map((item, idx) => (
                  <Button
                    key={idx}
                    variant="outline"
                    className="w-full justify-start h-12 border-nexus-border text-white hover:bg-nexus-card/80 bg-nexus-card/30 group"
                    onClick={() => alert(`Opening ${item.label}...`)}
                  >
                    <item.icon className={cn("w-4 h-4 mr-3 transition-transform group-hover:scale-110", item.color)} />
                    <span className="text-sm font-medium">{item.label}</span>
                  </Button>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* DQI Simulator Tab */}
        <TabsContent value="simulator">
          <Card className="bg-nexus-card border-nexus-border shadow-2xl overflow-hidden">
            <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 p-8 border-b border-nexus-border">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-purple-500 rounded-xl shadow-lg shadow-purple-500/20">
                  <TrendingUp className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">DQI Improvement Simulator</h2>
                  <p className="text-nexus-text-secondary">Model the impact of resolving specific data issues in real-time</p>
                </div>
              </div>
            </div>
            <CardContent className="p-8">
              <div className="grid lg:grid-cols-2 gap-12">
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-bold uppercase tracking-widest text-nexus-text-secondary">Select Issues to Fix</h4>
                    <Button variant="ghost" size="sm" className="text-xs text-purple-400 h-7" onClick={() => setSelectedIssues([])}>
                      Reset All
                    </Button>
                  </div>

                  <div className="space-y-3">
                    {dqiIssues.map((issue: DQIIDIssue) => (
                      <div
                        key={issue.id}
                        onClick={() => toggleIssue(issue.id)}
                        className={cn(
                          "group p-4 rounded-xl border cursor-pointer transition-all duration-200",
                          selectedIssues.includes(issue.id)
                            ? "bg-purple-500/10 border-purple-500 shadow-lg shadow-purple-500/5 ring-1 ring-purple-500/50"
                            : "bg-nexus-card border-nexus-border hover:border-nexus-text-muted"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className={cn(
                              "w-6 h-6 rounded-md border-2 flex items-center justify-center transition-colors",
                              selectedIssues.includes(issue.id)
                                ? "bg-purple-500 border-purple-500"
                                : "border-nexus-text-muted group-hover:border-purple-400"
                            )}>
                              {selectedIssues.includes(issue.id) && <CheckCircle2 className="w-4 h-4 text-white" />}
                            </div>
                            <div>
                              <p className="font-bold text-white text-base leading-tight">{issue.name}</p>
                              <p className="text-xs text-nexus-text-secondary mt-1">
                                {issue.count} items • {issue.component}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <Badge className={cn(
                              "bg-transparent border-0 font-black text-sm",
                              selectedIssues.includes(issue.id) ? "text-purple-400" : "text-success-400"
                            )}>
                              +{issue.dqi_impact}%
                            </Badge>
                            <p className="text-[10px] text-nexus-text-muted mt-1 uppercase font-bold">{issue.priority}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-8">
                  <h4 className="text-sm font-bold uppercase tracking-widest text-nexus-text-secondary">Projected Impact Analysis</h4>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-6 bg-nexus-card/50 rounded-2xl border border-nexus-border text-center">
                      <p className="text-xs font-bold text-nexus-text-secondary uppercase mb-3">Current DQI</p>
                      <p className="text-4xl font-black text-nexus-text-muted">{currentDQI.toFixed(1)}%</p>
                    </div>
                    <div className="p-6 bg-nexus-card/80 rounded-2xl border border-purple-500/30 text-center relative overflow-hidden group">
                      <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent pointer-events-none" />
                      <p className="text-xs font-bold text-purple-400 uppercase mb-3">Projected DQI</p>
                      <p className="text-4xl font-black text-success-400 transition-all duration-500 group-hover:scale-110">{projectedDQI.toFixed(1)}%</p>
                    </div>
                  </div>

                  <div className="p-6 bg-success-500/5 rounded-2xl border border-success-500/20">
                    <div className="flex items-center justify-between mb-6">
                      <p className="text-sm font-medium text-success-400">Total Improvement Potential</p>
                      <div className="px-3 py-1 bg-success-500/20 rounded-full">
                        <span className="text-success-400 font-black text-lg">+{totalSelectedImpact.toFixed(1)} pts</span>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-nexus-text-secondary">Estimated Resolution Effort</span>
                        <span className="text-white font-bold">{totalSelectedEffort}</span>
                      </div>
                      <Progress value={totalSelectedImpact * 10} className="h-2 bg-success-500/10" indicatorClassName="bg-success-400" />
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h5 className="text-xs font-bold text-nexus-text-muted uppercase tracking-tighter">Resolution Breakdown</h5>
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
                      {selectedIssues.map((id: string) => {
                        const issue = dqiIssues.find((i: DQIIDIssue) => i.id === id);
                        return issue ? (
                          <div key={id} className="flex items-center justify-between p-3 bg-nexus-card/30 rounded-lg border border-nexus-border/50">
                            <span className="text-sm text-white">{issue.name}</span>
                            <div className="text-right">
                              <span className="text-xs text-success-400 font-bold">+{issue.dqi_impact}%</span>
                              <span className="mx-2 text-nexus-text-muted text-[10px]">|</span>
                              <span className="text-[10px] text-nexus-text-secondary">~{issue.effort}</span>
                            </div>
                          </div>
                        ) : null;
                      })}
                      {selectedIssues.length === 0 && (
                        <div className="text-center py-8 bg-nexus-card/20 rounded-xl border border-dashed border-nexus-border">
                          <p className="text-sm text-nexus-text-muted">No issues selected for simulation</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <Button
                    className="w-full h-14 bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white font-black text-lg shadow-xl shadow-teal-500/10 transition-all active:scale-[0.98]"
                    disabled={selectedIssues.length === 0 || createActionPlanMutation.isPending}
                    onClick={() => createActionPlanMutation.mutate(selectedIssues)}
                  >
                    <Play className="w-5 h-5 mr-3 fill-current" />
                    {createActionPlanMutation.isPending ? 'CREATING...' : 'CREATE ACTION PLAN'}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance">
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-nexus-card border-nexus-border shadow-lg">
              <CardHeader>
                <CardTitle className="text-white">Peer Benchmarking</CardTitle>
                <CardDescription className="text-nexus-text-secondary">Your site DQI vs. top performing sites in the study</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={sitePerformance}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" vertical={false} />
                      <XAxis
                        dataKey="name"
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1a1f2e',
                          border: '1px solid #2d3548',
                          borderRadius: '12px',
                          color: '#fff',
                          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
                        }}
                        itemStyle={{ color: '#14b8a6' }}
                        cursor={{ fill: '#2d3548', opacity: 0.4 }}
                      />
                      <Bar dataKey="dqi" name="DQI Score" radius={[4, 4, 0, 0]}>
                        {sitePerformance.map((entry: { name: string; dqi: number }, index: number) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.name === currentSite.site_id ? '#14b8a6' : '#8b5cf6'}
                            fillOpacity={entry.name === currentSite.site_id ? 1 : 0.6}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-nexus-card border-nexus-border">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-white">Global Site Network</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Performance overview across all regions</CardDescription>
                </div>
                <div className="relative w-48">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-nexus-text-secondary" />
                  <Input
                    placeholder="Search sites..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-8 bg-nexus-card border-nexus-border text-white h-8 text-xs"
                  />
                </div>
              </CardHeader>
              <CardContent className="p-0 px-1">
                <Table>
                  <TableHeader>
                    <TableRow className="border-nexus-border hover:bg-transparent">
                      <TableHead className="text-nexus-text-secondary text-[10px] uppercase font-bold">Site ID</TableHead>
                      <TableHead className="text-nexus-text-secondary text-[10px] uppercase font-bold">DQI Performance</TableHead>
                      <TableHead className="text-nexus-text-secondary text-[10px] uppercase font-bold">Status</TableHead>
                      <TableHead className="text-right"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sites?.filter((site: any) =>
                      site.site_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      site.name.toLowerCase().includes(searchQuery.toLowerCase())
                    ).slice(0, 8).map((site: any) => (
                      <TableRow key={site.site_id} className="border-nexus-border hover:bg-nexus-card-hover group">
                        <TableCell className="font-bold text-white text-xs">{site.site_id}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className={cn(
                              "text-xs font-bold",
                              (site.dqi_score || 0) >= 85 ? 'text-success-400' :
                                (site.dqi_score || 0) >= 70 ? 'text-warning-400' :
                                  'text-error-400'
                            )}>
                              {site.dqi_score?.toFixed(1) || '0.0'}%
                            </span>
                            <div className="w-12 h-1 bg-nexus-border rounded-full overflow-hidden">
                              <div
                                className={cn(
                                  "h-full rounded-full",
                                  (site.dqi_score || 0) >= 85 ? 'bg-success-400' :
                                    (site.dqi_score || 0) >= 70 ? 'bg-warning-400' :
                                      'bg-error-400'
                                )}
                                style={{ width: `${site.dqi_score || 0}%` }}
                              />
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge className={cn(
                            "text-[9px] h-4 py-0 font-bold",
                            site.status === 'active' ? "bg-success-500/10 text-success-400 border-success-500/20" : "bg-nexus-text-muted/10 text-nexus-text-muted border-nexus-text-muted/20"
                          )}>
                            {site.status?.toUpperCase() || 'INACTIVE'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-nexus-text-secondary hover:text-white"
                            onClick={() => setSelectedSiteId(site.site_id)}
                          >
                            Access Portal
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
