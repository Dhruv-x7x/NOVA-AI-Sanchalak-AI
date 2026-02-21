import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { reportsApi, sitesApi, studiesApi } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  FileText,
  Download,
  Loader2,
  CheckCircle2,
  Calendar,
  User,
  Building2,
  Activity,
  Shield,
  AlertTriangle,
  BarChart3,
  FileSpreadsheet,
  File,
  Eye,
  Clock,
  RefreshCw,
  Printer,
} from 'lucide-react';

interface ReportType {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
}

export default function Reports() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const [generatedReport, setGeneratedReport] = useState<string | null>(null);
  const [selectedSiteId, setSelectedSiteId] = useState<string>('all');
  const [selectedStudyId, setSelectedStudyId] = useState<string>('all');
  const [isDownloading, setIsDownloading] = useState(false);
  const [reportConfig, setReportConfig] = useState({
    craName: '',
    reportPeriod: '30',
    outputFormat: 'xlsx',
    includeCharts: true,
  });

  // Fetch report types from the API
  const { data: reportTypes, isLoading: typesLoading } = useQuery({
    queryKey: ['report-types'],
    queryFn: () => reportsApi.getTypes(),
  });

  // Fetch sites for the dropdown
  const { data: sitesData } = useQuery({
    queryKey: ['sites-list'],
    queryFn: () => sitesApi.list(),
  });

  // Fetch studies for the dropdown
  const { data: studiesData } = useQuery({
    queryKey: ['studies-list'],
    queryFn: () => studiesApi.list(),
  });

  const sites = sitesData?.sites || [];
  const studies = studiesData?.studies || [];

  const generateMutation = useMutation({
    mutationFn: (reportType: string) => reportsApi.generate({
      report_type: reportType,
      format: 'html',  // Always preview as HTML; Download button uses actual format
      site_id: selectedSiteId === 'all' ? undefined : selectedSiteId,
      study_id: selectedStudyId === 'all' ? undefined : selectedStudyId,
    }),
    onSuccess: (data) => {
      setGeneratedReport(data.content);
    },
  });

  const handleGenerateReport = () => {
    if (selectedReport) {
      generateMutation.mutate(selectedReport);
    }
  };

  const handleDownload = async () => {
    if (!selectedReport) return;
    setIsDownloading(true);

    try {
      const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
      const siteParam = selectedSiteId !== 'all' ? `&site_id=${selectedSiteId}` : '';
      const studyParam = selectedStudyId !== 'all' ? `&study_id=${selectedStudyId}` : '';
      const format = reportConfig.outputFormat;

      const token = localStorage.getItem('auth_token') || ''; // Assuming token is here, or use store
      const url = `${baseUrl}/reports/download/${selectedReport}?format=${format}${siteParam}${studyParam}`;

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      const extension = format === 'excel' || format === 'xlsx' ? 'xlsx' : format;
      link.setAttribute('download', `${selectedReport}_${new Date().toISOString().split('T')[0]}.${extension}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download error:', error);
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePrint = () => {
    const printWindow = window.open('', '_blank');
    if (printWindow && generatedReport) {
      printWindow.document.write(`
        <html>
          <head><title>Print Report</title></head>
          <body onload="window.print();window.close()">${generatedReport}</body>
        </html>
      `);
      printWindow.document.close();
    }
  };

  // Group reports by category
  const groupedReports = (reportTypes?.report_types || []).reduce((acc: Record<string, ReportType[]>, report: ReportType) => {
    const category = report.category || 'General';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(report);
    return acc;
  }, {});

  // Get icon for report type
  const getReportIcon = (iconName: string) => {
    const icons: Record<string, React.ComponentType<{ className?: string }>> = {
      'file-text': FileText,
      'activity': Activity,
      'shield': Shield,
      'alert-triangle': AlertTriangle,
      'bar-chart': BarChart3,
      'building': Building2,
      'user': User,
      'calendar': Calendar,
    };
    const IconComponent = icons[iconName] || FileText;
    return <IconComponent className="w-5 h-5" />;
  };

  // Recent reports (mock data)
  const recentReports = [
    { id: 1, name: 'CRA Weekly Summary', type: 'cra_weekly', generatedAt: '2024-01-25 10:30', format: 'PDF', status: 'completed' },
    { id: 2, name: 'Safety Summary', type: 'safety_summary', generatedAt: '2024-01-24 16:45', format: 'HTML', status: 'completed' },
    { id: 3, name: 'DQI Trend Report', type: 'dqi_trend', generatedAt: '2024-01-24 09:15', format: 'Excel', status: 'completed' },
    { id: 4, name: 'Site Performance', type: 'site_performance', generatedAt: '2024-01-23 14:00', format: 'PDF', status: 'completed' },
  ];

  const selectedReportDetails = (reportTypes?.report_types || []).find((r: ReportType) => r.id === selectedReport);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Report Generation Center</h1>
              <p className="text-nexus-text-secondary">Generate and download clinical trial reports</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="success" className="text-sm">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              All Systems Operational
            </Badge>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="generate" className="space-y-6">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="generate" className="data-[state=active]:bg-emerald-600">Generate Report</TabsTrigger>
          <TabsTrigger value="recent" className="data-[state=active]:bg-emerald-600">Recent Reports</TabsTrigger>
          <TabsTrigger value="scheduled" className="data-[state=active]:bg-emerald-600">Scheduled</TabsTrigger>
        </TabsList>

        {/* Generate Report Tab */}
        <TabsContent value="generate" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            {/* Report Types */}
            <div className="md:col-span-2">
              <Card className="glass-card border-nexus-border">
                <CardHeader>
                  <CardTitle className="text-white">Select Report Type</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Choose a report template to generate</CardDescription>
                </CardHeader>
                <CardContent>
                  {typesLoading ? (
                    <div className="text-center py-8 text-nexus-text-secondary">Loading report types...</div>
                  ) : (
                    <div className="space-y-6">
                      {Object.entries(groupedReports).map(([category, reports]) => (
                        <div key={category}>
                          <h4 className="text-sm font-medium text-nexus-text-secondary mb-3">{category}</h4>
                          <div className="grid gap-3 md:grid-cols-2">
                            {(reports as ReportType[]).map((report) => (
                              <div
                                key={report.id}
                                onClick={() => setSelectedReport(report.id)}
                                className={`p-4 rounded-lg border cursor-pointer transition-all ${selectedReport === report.id
                                    ? 'bg-emerald-500/10 border-emerald-500/50 ring-1 ring-emerald-500/30'
                                    : 'bg-nexus-card border-nexus-border hover:border-emerald-500/30'
                                  }`}
                              >
                                <div className="flex items-start gap-3">
                                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${selectedReport === report.id
                                      ? 'bg-emerald-500/20 text-emerald-400'
                                      : 'bg-nexus-card text-nexus-text-secondary'
                                    }`}>
                                    {getReportIcon(report.icon)}
                                  </div>
                                  <div className="flex-1">
                                    <h3 className="font-medium text-white">{report.name}</h3>
                                    <p className="text-sm text-nexus-text-secondary mt-1">{report.description}</p>
                                  </div>
                                  {selectedReport === report.id && (
                                    <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Configuration Panel */}
            <div className="space-y-6">
              <Card className="glass-card border-nexus-border">
                <CardHeader>
                  <CardTitle className="text-white">Configuration</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Customize your report</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-white">CRA Name</Label>
                    <Input
                      placeholder="Enter CRA name"
                      value={reportConfig.craName}
                      onChange={(e) => setReportConfig({ ...reportConfig, craName: e.target.value })}
                      className="bg-nexus-card border-nexus-border text-white"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-white">Filter by Study</Label>
                      <Select value={selectedStudyId} onValueChange={setSelectedStudyId}>
                        <SelectTrigger className="bg-nexus-card border-nexus-border text-white">
                          <SelectValue placeholder="All Studies" />
                        </SelectTrigger>
                        <SelectContent className="bg-nexus-card border-nexus-border text-white">
                          <SelectItem value="all">All Studies</SelectItem>
                          {studies.map((study: any) => (
                            <SelectItem key={study.study_id} value={study.study_id}>
                              {study.study_id}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-white">Filter by Site</Label>
                      <Select value={selectedSiteId} onValueChange={setSelectedSiteId}>
                        <SelectTrigger className="bg-nexus-card border-nexus-border text-white">
                          <SelectValue placeholder="All Sites" />
                        </SelectTrigger>
                        <SelectContent className="bg-nexus-card border-nexus-border text-white">
                          <SelectItem value="all">All Sites</SelectItem>
                          {sites.map((site: any) => (
                            <SelectItem key={site.site_id} value={site.site_id}>
                              {site.site_id}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-white">Report Period</Label>
                    <Select
                      value={reportConfig.reportPeriod}
                      onValueChange={(value) => setReportConfig({ ...reportConfig, reportPeriod: value })}
                    >
                      <SelectTrigger className="bg-nexus-card border-nexus-border text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-nexus-card border-nexus-border">
                        <SelectItem value="7">Last 7 days</SelectItem>
                        <SelectItem value="14">Last 14 days</SelectItem>
                        <SelectItem value="30">Last 30 days</SelectItem>
                        <SelectItem value="90">Last 90 days</SelectItem>
                        <SelectItem value="custom">Custom range</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-white">Output Format</Label>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { id: 'html', label: 'HTML', icon: FileText },
                        { id: 'pdf', label: 'PDF', icon: File },
                        { id: 'excel', label: 'Excel', icon: FileSpreadsheet },
                      ].map((format) => (
                        <Button
                          key={format.id}
                          variant={reportConfig.outputFormat === format.id ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setReportConfig({ ...reportConfig, outputFormat: format.id })}
                          className={reportConfig.outputFormat === format.id
                            ? 'bg-emerald-600 hover:bg-emerald-700'
                            : 'border-nexus-border text-nexus-text-secondary hover:text-white'
                          }
                        >
                          <format.icon className="w-3 h-3 mr-1" />
                          {format.label}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Selected Report Info */}
              {selectedReportDetails && (
                <Card className="glass-card border-nexus-border">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                        {getReportIcon(selectedReportDetails.icon)}
                      </div>
                      <div>
                        <p className="font-medium text-white">{selectedReportDetails.name}</p>
                        <p className="text-xs text-nexus-text-secondary">{selectedReportDetails.category}</p>
                      </div>
                    </div>
                    <p className="text-sm text-nexus-text-secondary mb-4">{selectedReportDetails.description}</p>

                    <div className="space-y-2">
                      <Button
                        onClick={handleGenerateReport}
                        disabled={generateMutation.isPending}
                        className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
                      >
                        {generateMutation.isPending ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <FileText className="w-4 h-4 mr-2" />
                            Generate Report
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* Generated Report Preview */}
          {generatedReport && (
            <Card className="glass-card border-nexus-border">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white">Report Preview</CardTitle>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handlePrint}
                      className="border-nexus-border text-nexus-text-secondary hover:text-white"
                    >
                      <Printer className="w-4 h-4 mr-2" />
                      Print
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleDownload}
                      disabled={isDownloading}
                      className="bg-emerald-600 hover:bg-emerald-700"
                    >
                      {isDownloading ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <Download className="w-4 h-4 mr-2" />
                      )}
                      Download {reportConfig.outputFormat.toUpperCase()}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div
                  className="prose prose-invert max-w-none p-6 bg-nexus-card rounded-lg border border-nexus-border overflow-auto max-h-[600px]"
                  dangerouslySetInnerHTML={{ __html: generatedReport }}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Recent Reports Tab */}
        <TabsContent value="recent">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white">Recent Reports</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Previously generated reports</CardDescription>
                </div>
                <Button variant="outline" size="sm" className="border-nexus-border text-nexus-text-secondary hover:text-white">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {recentReports.length > 0 ? (
                <div className="space-y-3">
                  {recentReports.map((report) => (
                    <div
                      key={report.id}
                      className="flex items-center justify-between p-4 bg-nexus-card rounded-lg border border-nexus-border hover:border-emerald-500/30 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                          <FileText className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                          <p className="font-medium text-white">{report.name}</p>
                          <div className="flex items-center gap-3 mt-1">
                            <span className="text-xs text-nexus-text-secondary flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {report.generatedAt}
                            </span>
                            <Badge variant="secondary">{report.format}</Badge>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="success">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          {report.status}
                        </Badge>
                        <Button variant="ghost" size="sm" className="text-nexus-text-secondary hover:text-white">
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button variant="ghost" size="sm" className="text-nexus-text-secondary hover:text-white">
                          <Download className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <FileText className="w-12 h-12 mx-auto mb-4 text-nexus-text-secondary" />
                  <p className="text-nexus-text-secondary">No recent reports</p>
                  <p className="text-sm text-nexus-text-secondary mt-1">Generate your first report to see it here</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Scheduled Reports Tab */}
        <TabsContent value="scheduled">
          <Card className="glass-card border-nexus-border">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-white">Scheduled Reports</CardTitle>
                  <CardDescription className="text-nexus-text-secondary">Automated report generation schedule</CardDescription>
                </div>
                <Button className="bg-emerald-600 hover:bg-emerald-700">
                  <Calendar className="w-4 h-4 mr-2" />
                  Schedule New Report
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <Calendar className="w-12 h-12 mx-auto mb-4 text-nexus-text-secondary" />
                <p className="text-white mb-2">No scheduled reports</p>
                <p className="text-sm text-nexus-text-secondary">Set up automated report generation to receive reports on a regular schedule</p>
                <Button className="mt-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700">
                  <Calendar className="w-4 h-4 mr-2" />
                  Create Schedule
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
