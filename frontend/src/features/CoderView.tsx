import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { codingApi } from '@/services/api';
import { useAppStore } from '@/stores/appStore';
import SanchalakLoader from '@/components/SanchalakLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
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
  Code,
  CheckCircle2,
  AlertTriangle,
  Search,
  ThumbsUp,
  ArrowUpRight,
  Stethoscope,
  Pill,
  Zap,
  TrendingUp,
  Loader2,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

interface CodingItem {
  item_id: string;
  patient_key: string;
  site_id: string;
  study_id: string;
  verbatim_term: string;
  dictionary_type: string;
  form_name: string;
  field_name: string;
  status: string;
  priority: string;
  created_at: string;
  confidence_score: number;
  coded_term?: string;
  coded_code?: string;
  suggested_term?: string;
  suggested_code?: string;
}

export default function CoderView() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('queue');
  const [selectedItems, setSelectedItems] = useState<string[]>([]);
  const [hiddenItemIds, setHiddenItemIds] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('nexus_hidden_coding');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const hideItem = (id: string) => {
    setHiddenItemIds(prev => {
      const next = [...prev, id];
      localStorage.setItem('nexus_hidden_coding', JSON.stringify(next));
      return next;
    });
  };

  const queryClient = useQueryClient();

  const { selectedStudy } = useAppStore();

  // Fetch coding queue
  const { data: codingQueue, isLoading: isQueueLoading, isError: isQueueError, error: queueError } = useQuery({
    queryKey: ['coding-queue', selectedStudy],
    queryFn: () => codingApi.getQueue({ limit: 500, study_id: selectedStudy }),
  });

  // Fetch coding stats
  const { data: codingStats, isLoading: isStatsLoading, isError: isStatsError, error: statsError } = useQuery({
    queryKey: ['coding-stats', selectedStudy],
    queryFn: () => codingApi.getStats(selectedStudy),
  });

  // Fetch escalated items separately (queue LIMIT may exclude older escalated items)
  const { data: escalatedQueue } = useQuery({
    queryKey: ['coding-escalated', selectedStudy],
    queryFn: () => codingApi.getQueue({ status: 'escalated', limit: 200, study_id: selectedStudy }),
  });

  // Fetch productivity data
  const { data: productivity } = useQuery({
    queryKey: ['coding-productivity'],
    queryFn: () => codingApi.getProductivity(30),
  });

  // Mutations
  const approveMutation = useMutation({
    mutationFn: ({ itemId, codedTerm, codedCode }: { itemId: string, codedTerm: string, codedCode: string }) => 
      codingApi.approve(itemId, codedTerm, codedCode),
    onSuccess: (_, variables) => {
      hideItem(variables.itemId);
      queryClient.invalidateQueries({ queryKey: ['coding-stats'] });
    },
  });

  const escalateMutation = useMutation({
    mutationFn: ({ itemId, reason }: { itemId: string, reason: string }) => 
      codingApi.escalate(itemId, reason),
    onSuccess: (_, variables) => {
      hideItem(variables.itemId);
      queryClient.invalidateQueries({ queryKey: ['coding-stats'] });
    },
  });

  const items = (codingQueue?.items || [])
    .filter((item: CodingItem) => !hiddenItemIds.includes(item.item_id));
  const pendingMeddra = codingQueue?.pending_meddra || codingStats?.meddra?.pending || 0;
  const pendingWhodrug = codingQueue?.pending_whodrug || codingStats?.whodrug?.pending || 0;
  const codedMeddra = codingStats?.meddra?.coded || 0;
  const codedWhodrug = codingStats?.whodrug?.coded || 0;
  const totalCoded = codingStats?.total_coded || 0;
  const todayCoded = codingStats?.today_coded || 0;
  const highConfReady = codingStats?.high_confidence_ready || 0;
  const escalated = (codingStats?.meddra?.escalated || 0) + (codingStats?.whodrug?.escalated || 0);

  // Filter items
  const filteredItems = items.filter((item: CodingItem) => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        item.verbatim_term?.toLowerCase().includes(query) ||
        item.patient_key?.toLowerCase().includes(query) ||
        item.site_id?.toLowerCase().includes(query)
      );
    }
    return true;
  });

  const pendingItems = filteredItems.filter((i: CodingItem) => i.status === 'pending');
  const escalatedItems = (escalatedQueue?.items || []).filter((item: CodingItem) => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        item.verbatim_term?.toLowerCase().includes(query) ||
        item.patient_key?.toLowerCase().includes(query) ||
        item.site_id?.toLowerCase().includes(query)
      );
    }
    return true;
  });
  const highConfItems = filteredItems.filter((i: CodingItem) => i.confidence_score >= 0.9);

  // Productivity chart data
  const productivityData = productivity?.daily_trend || [];

  // Toggle item selection
  const toggleItemSelection = (itemId: string) => {
    setSelectedItems(prev => 
      prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  };

  // Batch approve high confidence
  const handleBatchApprove = () => {
    highConfItems.forEach((item: CodingItem) => {
      approveMutation.mutate({
        itemId: item.item_id,
        codedTerm: item.suggested_term || item.verbatim_term || '',
        codedCode: item.suggested_code || ''
      });
    });
    setSelectedItems([]);
  };

  return (
    <div className="space-y-6">
      {(isQueueError || isStatsError) && (
        <div className="glass-card rounded-xl p-4 border border-error-500/30 bg-error-500/10">
          <p className="text-sm text-white font-semibold">Coding data failed to load</p>
          <p className="text-xs text-nexus-text-secondary mt-1">
            {String((statsError as any)?.message || (queueError as any)?.message || 'Unknown error')}
          </p>
        </div>
      )}

      {/* Header */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center">
              <Code className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Medical Coder Workbench</h1>
              <p className="text-nexus-text-secondary">MedDRA and WHODrug coding with AI assistance</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="warning" className="text-lg px-4 py-2">
              {pendingMeddra + pendingWhodrug} Pending
              {totalCoded > 0 ? ` • ${totalCoded.toLocaleString()} Coded` : ''}
            </Badge>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card className="kpi-card kpi-card-info">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{pendingMeddra}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">MedDRA Pending</p>
                <p className="text-[11px] text-nexus-text-muted mt-1">Coded: {codedMeddra.toLocaleString()}</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-info-500/20 flex items-center justify-center">
                <Stethoscope className="w-6 h-6 text-info-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-purple">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{pendingWhodrug}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">WHODrug Pending</p>
                <p className="text-[11px] text-nexus-text-muted mt-1">Coded: {codedWhodrug.toLocaleString()}</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Pill className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-success">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{todayCoded}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Coded Today</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-success-500/20 flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-success-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-warning">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{highConfReady}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">High Conf. Ready</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-warning-500/20 flex items-center justify-center">
                <Zap className="w-6 h-6 text-warning-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="kpi-card kpi-card-error">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{escalated}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Escalated</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-error-500/20 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-error-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-nexus-text-secondary" />
        <Input
          placeholder="Search by verbatim term, patient ID, or site..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 bg-nexus-card border-nexus-border text-white placeholder:text-nexus-text-secondary"
        />
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="queue" className="data-[state=active]:bg-indigo-600">
            Coding Queue ({pendingMeddra + pendingWhodrug})
          </TabsTrigger>
          <TabsTrigger value="escalated" className="data-[state=active]:bg-indigo-600">
            Escalations ({escalated})
          </TabsTrigger>
          <TabsTrigger value="productivity" className="data-[state=active]:bg-indigo-600">
            Productivity
          </TabsTrigger>
        </TabsList>

        {/* Coding Queue Tab */}
        <TabsContent value="queue">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white">AI-Assisted Coding Queue</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Items requiring medical coding review</CardDescription>
                </div>
                {highConfItems.length > 0 && (
                  <Button 
                    onClick={handleBatchApprove}
                    className="bg-gradient-to-r from-success-600 to-success-700 hover:from-success-700 hover:to-success-800"
                    disabled={approveMutation.isPending}
                  >
                    {approveMutation.isPending ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                    )}
                    Batch Approve High Confidence ({highConfItems.length})
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {(isQueueLoading || isStatsLoading) ? (
                <SanchalakLoader size="md" label="Loading coding queue..." className="py-8" />
              ) : pendingItems.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow className="border-nexus-border hover:bg-transparent">
                      <TableHead className="w-10"></TableHead>
                      <TableHead className="text-nexus-text-secondary">Item ID</TableHead>
                      <TableHead className="text-nexus-text-secondary">Dictionary</TableHead>
                      <TableHead className="text-nexus-text-secondary">Verbatim Term</TableHead>
                      <TableHead className="text-nexus-text-secondary">Suggested Code</TableHead>
                      <TableHead className="text-nexus-text-secondary">Patient</TableHead>
                      <TableHead className="text-nexus-text-secondary">Confidence</TableHead>
                      <TableHead className="text-right text-nexus-text-secondary">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pendingItems.slice(0, 20).map((item: CodingItem) => (
                      <TableRow key={item.item_id} className="border-nexus-border hover:bg-nexus-card/50">
                        <TableCell>
                          <Checkbox
                            checked={selectedItems.includes(item.item_id)}
                            onCheckedChange={() => toggleItemSelection(item.item_id)}
                          />
                        </TableCell>
                        <TableCell className="font-mono text-sm text-white">{item.item_id}</TableCell>
                        <TableCell>
                          <Badge variant={item.dictionary_type === 'MEDDRA' ? 'info' : 'secondary'}>
                            {item.dictionary_type}
                          </Badge>
                        </TableCell>
                        <TableCell className="max-w-xs">
                          <div>
                            <p className="text-white truncate">{item.verbatim_term}</p>
                            {item.suggested_term && (
                              <p className="text-xs text-success-400 mt-1">
                                Suggested: {item.suggested_term}
                              </p>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="font-mono text-sm text-nexus-text-secondary">
                          {item.suggested_code || '-'}
                        </TableCell>
                        <TableCell className="text-nexus-text-secondary">{item.patient_key}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Progress 
                              value={item.confidence_score * 100} 
                              className="w-16 h-2"
                            />
                            <span className={`text-xs font-medium ${
                              item.confidence_score >= 0.9 ? 'text-success-400' :
                              item.confidence_score >= 0.7 ? 'text-warning-400' :
                              'text-error-400'
                            }`}>
                              {(item.confidence_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button 
                              size="sm"
                              variant="outline"
                              className="h-8 px-3 border-success-500/50 text-success-400 hover:bg-success-500/10 hover:text-success-300"
                              onClick={() => approveMutation.mutate({
                                itemId: item.item_id,
                                codedTerm: item.suggested_term || item.verbatim_term || '',
                                codedCode: item.suggested_code || ''
                              })}
                              disabled={approveMutation.isPending}
                            >
                              <ThumbsUp className="w-4 h-4 mr-1" />
                              Approve
                            </Button>
                            <Button 
                              size="sm"
                              variant="outline"
                              className="h-8 px-3 border-error-500/50 text-error-400 hover:bg-error-500/10 hover:text-error-300"
                              onClick={() => escalateMutation.mutate({
                                itemId: item.item_id,
                                reason: 'Manual rejection/escalation'
                              })}
                              disabled={escalateMutation.isPending}
                            >
                              <AlertTriangle className="w-4 h-4 mr-1" />
                              Reject
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-8">
                  <Code className="w-12 h-12 mx-auto mb-4 text-nexus-text-secondary" />
                  <p className="text-nexus-text-secondary">No pending coding tasks</p>
                  {totalCoded > 0 && (
                    <p className="text-xs text-nexus-text-muted mt-1">{totalCoded.toLocaleString()} coded terms available in UPR</p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Escalations Tab */}
        <TabsContent value="escalated">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <CardTitle className="text-white">Escalated Items</CardTitle>
              <CardDescription className="text-nexus-text-secondary">Items requiring senior coder review</CardDescription>
            </CardHeader>
            <CardContent>
              {escalatedItems.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow className="border-nexus-border hover:bg-transparent">
                      <TableHead className="text-nexus-text-secondary">Item ID</TableHead>
                      <TableHead className="text-nexus-text-secondary">Dictionary</TableHead>
                      <TableHead className="text-nexus-text-secondary">Verbatim Term</TableHead>
                      <TableHead className="text-nexus-text-secondary">Patient</TableHead>
                      <TableHead className="text-nexus-text-secondary">Site</TableHead>
                      <TableHead className="text-nexus-text-secondary">Escalation Reason</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {escalatedItems.map((item: CodingItem) => (
                      <TableRow key={item.item_id} className="border-nexus-border hover:bg-nexus-card/50">
                        <TableCell className="font-mono text-sm text-white">{item.item_id}</TableCell>
                        <TableCell>
                          <Badge variant="secondary">{item.dictionary_type}</Badge>
                        </TableCell>
                        <TableCell className="max-w-xs truncate text-white">{item.verbatim_term}</TableCell>
                        <TableCell className="text-nexus-text-secondary">{item.patient_key}</TableCell>
                        <TableCell className="text-nexus-text-secondary">{item.site_id}</TableCell>
                        <TableCell className="text-warning-400">Low confidence match</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-8">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-nexus-text-secondary" />
                  <p className="text-nexus-text-secondary">No escalated items</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Productivity Tab */}
        <TabsContent value="productivity">
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white">Daily Coding Activity</CardTitle>
                <CardDescription className="text-nexus-text-secondary">Items coded per day over the last 30 days</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={productivityData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
                      <XAxis 
                        dataKey="date" 
                        tickFormatter={(v) => v.slice(5)} 
                        stroke="#64748b"
                        tick={{ fill: '#94a3b8', fontSize: 12 }}
                      />
                      <YAxis stroke="#64748b" tick={{ fill: '#94a3b8' }} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1a1f2e', 
                          border: '1px solid #2d3548',
                          borderRadius: '8px',
                          color: '#fff'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="coded" 
                        name="Items Coded" 
                        stroke="#8b5cf6" 
                        fill="#8b5cf6"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <CardTitle className="text-white">Coding Statistics</CardTitle>
                <CardDescription className="text-nexus-text-secondary">Performance metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-nexus-card rounded-lg border border-nexus-border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-nexus-text-secondary">Total Coded (All Time)</span>
                      <span className="text-xl font-bold text-white">{totalCoded.toLocaleString()}</span>
                    </div>
                    <Progress value={75} className="h-2" />
                  </div>
                  
                  <div className="p-4 bg-nexus-card rounded-lg border border-nexus-border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-nexus-text-secondary">Today's Progress</span>
                      <span className="text-xl font-bold text-success-400">{todayCoded}</span>
                    </div>
                    <Progress value={(todayCoded / 50) * 100} className="h-2" />
                    <p className="text-xs text-nexus-text-secondary mt-2">Target: 50 items/day</p>
                  </div>

                  <div className="p-4 bg-nexus-card rounded-lg border border-nexus-border">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-nexus-text-secondary">Auto-Code Rate</span>
                      <span className="text-xl font-bold text-info-400">82%</span>
                    </div>
                    <Progress value={82} className="h-2" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-success-500/10 rounded-lg border border-success-500/20 text-center">
                      <p className="text-2xl font-bold text-success-400">98.5%</p>
                      <p className="text-xs text-nexus-text-secondary">Accuracy Rate</p>
                    </div>
                    <div className="p-4 bg-info-500/10 rounded-lg border border-info-500/20 text-center">
                      <p className="text-2xl font-bold text-info-400">1.2 min</p>
                      <p className="text-xs text-nexus-text-secondary">Avg. Time/Item</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
