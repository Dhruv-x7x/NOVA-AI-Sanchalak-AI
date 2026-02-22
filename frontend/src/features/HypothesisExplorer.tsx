import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '@/services/api';
import A6onLoader from '@/components/a6on_iLoader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Lightbulb,
  AlertCircle,
  TrendingUp,
  Eye,
  CheckCircle2,
} from 'lucide-react';

export default function HypothesisExplorer() {
  const { data: patterns, isLoading: patternsLoading } = useQuery({
    queryKey: ['pattern-alerts'],
    queryFn: () => analyticsApi.getPatterns(),
  });

  if (patternsLoading) {
    return <A6onLoader size="lg" label="Loading hypotheses..." fullPage />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Hypothesis Explorer</h1>
        <p className="text-gray-500">AI-generated insights and pattern detection</p>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-50 rounded-lg">
                <Lightbulb className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{patterns?.total || 0}</p>
                <p className="text-xs text-gray-500">Patterns Detected</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-warning-50 rounded-lg">
                <AlertCircle className="w-5 h-5 text-warning-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {patterns?.alerts?.filter((a: { severity: string }) => a.severity === 'high').length || 0}
                </p>
                <p className="text-xs text-gray-500">High Severity</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-success-50 rounded-lg">
                <TrendingUp className="w-5 h-5 text-success-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">-</p>
                <p className="text-xs text-gray-500">Actionable Insights</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Pattern Alerts */}
      <Card>
        <CardHeader>
          <CardTitle>Pattern Alerts</CardTitle>
          <CardDescription>AI-detected patterns and anomalies</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {patterns?.alerts?.map((alert: {
              pattern_id: string;
              pattern_name: string;
              severity: string;
              match_count: number;
              sites_affected: number;
              status: string;
              alert_message?: string;
            }) => (
              <div key={alert.pattern_id} className="p-4 border rounded-lg">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${alert.severity === 'high' ? 'bg-error-50' :
                        alert.severity === 'medium' ? 'bg-warning-50' : 'bg-primary-50'
                      }`}>
                      <AlertCircle className={`w-5 h-5 ${alert.severity === 'high' ? 'text-error-600' :
                          alert.severity === 'medium' ? 'text-warning-600' : 'text-primary'
                        }`} />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-100">{alert.pattern_name}</h4>
                      <p className="text-sm text-gray-500 mt-1">{alert.alert_message || 'Pattern detected'}</p>
                      <div className="flex gap-4 mt-2 text-sm text-gray-500">
                        <span>Matches: {alert.match_count}</span>
                        <span>Sites: {alert.sites_affected}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={
                      alert.severity === 'high' ? 'error' :
                        alert.severity === 'medium' ? 'warning' : 'info'
                    }>
                      {alert.severity}
                    </Badge>
                    <Button variant="ghost" size="sm">
                      <Eye className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}

            {(!patterns?.alerts || patterns.alerts.length === 0) && (
              <div className="text-center py-8 text-gray-500">
                <Lightbulb className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No patterns detected</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Hypothesis Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Generated Hypotheses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              <Lightbulb className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>AI-generated hypotheses will appear here</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Validated Insights</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8 text-gray-500">
              <CheckCircle2 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>Validated insights will appear here</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
