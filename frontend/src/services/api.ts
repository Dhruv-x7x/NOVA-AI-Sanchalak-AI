import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/stores/authStore';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = useAuthStore.getState().accessToken;
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    // If 401 and not a retry, and NOT a login/refresh request
    const isAuthRequest = originalRequest.url?.includes('/auth/login') || originalRequest.url?.includes('/auth/refresh');

    if (error.response?.status === 401 && !originalRequest._retry && !isAuthRequest) {
      originalRequest._retry = true;

      const refreshToken = useAuthStore.getState().refreshToken;

      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token, user } = response.data;
          useAuthStore.getState().setTokens(access_token, refresh_token, user);

          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
          }

          return api(originalRequest);
        } catch (refreshError) {
          // Refresh failed, logout
          useAuthStore.getState().logout();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        // No refresh token, logout
        useAuthStore.getState().logout();
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  login: async (username: string, password: string) => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },

  logout: async () => {
    try {
      await api.post('/auth/logout');
    } catch {
      // Ignore logout errors
    }
  },

  getMe: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
};

// Patients API
export const patientsApi = {
  list: async (params?: { page?: number; page_size?: number; site_id?: string; study_id?: string; status?: string }) => {
    const response = await api.get('/patients', { params });
    return response.data;
  },

  search: async (query: string, limit = 20) => {
    const response = await api.get('/patients/search', { params: { q: query, limit } });
    return response.data;
  },

  get: async (patientKey: string) => {
    const response = await api.get(`/patients/${encodeURIComponent(patientKey)}`);
    return response.data;
  },

  getDQI: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/patients/dqi', { params });
    return response.data;
  },

  getCleanStatus: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/patients/clean-status', { params });
    return response.data;
  },

  getDBLockStatus: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/patients/dblock-status', { params });
    return response.data;
  },

  getIssues: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/patients/issues', { params });
    return response.data;
  },

  getRiskExplanation: async (patientKey: string) => {
    const response = await api.get(`/patients/${encodeURIComponent(patientKey)}/risk-explanation`);
    return response.data;
  },
};

// Sites API
export const sitesApi = {
  list: async (params?: { country?: string; region?: string; status?: string; study_id?: string }) => {
    const response = await api.get('/sites', { params });
    return response.data;
  },

  get: async (siteId: string) => {
    const response = await api.get(`/sites/${siteId}`);
    return response.data;
  },

  getBenchmarks: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/sites/benchmarks', { params });
    return response.data;
  },

  getSmartQueue: async (studyId?: string, limit = 50) => {
    const params: Record<string, string | number> = { limit };
    if (studyId && studyId !== 'all') params.study_id = studyId;
    const response = await api.get('/sites/smart-queue', { params });
    return response.data;
  },

  getActivityLogs: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') params.study_id = studyId;
    const response = await api.get('/sites/activity-logs', { params });
    return response.data;
  },

  getRegions: async () => {
    const response = await api.get('/sites/regions');
    return response.data;
  },

  getPatients: async (siteId: string, params?: { page?: number; page_size?: number }) => {
    const response = await api.get(`/sites/${siteId}/patients`, { params });
    return response.data;
  },

  getPortalData: async (siteId: string) => {
    const response = await api.get(`/sites/${siteId}/portal`);
    return response.data;
  },

  getDQIIssues: async (siteId: string) => {
    const response = await api.get(`/sites/${siteId}/dqi-issues`);
    return response.data;
  },

  createActionPlan: async (siteId: string, issueIds: string[]) => {
    const response = await api.post(`/sites/${siteId}/action-plan`, { issue_ids: issueIds });
    return response.data;
  },
};

// Studies API
export const studiesApi = {
  list: async (params?: { phase?: string; status?: string; therapeutic_area?: string }) => {
    const response = await api.get('/studies', { params });
    return response.data;
  },

  get: async (studyId: string) => {
    const response = await api.get(`/studies/${studyId}`);
    return response.data;
  },

  getSites: async (studyId: string) => {
    const response = await api.get(`/studies/${studyId}/sites`);
    return response.data;
  },

  getPatients: async (studyId: string) => {
    const response = await api.get(`/studies/${studyId}/patients`);
    return response.data;
  },
};

// Analytics API
export const analyticsApi = {
  getPortfolio: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/portfolio', { params });
    return response.data;
  },

  getDQIDistribution: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/dqi-distribution', { params });
    return response.data;
  },

  getCleanStatusSummary: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/clean-status-summary', { params });
    return response.data;
  },

  getDBLockSummary: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/dblock-summary', { params });
    return response.data;
  },

  getCascade: async (limit = 100, studyId?: string) => {
    const params: Record<string, any> = { limit };
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/cascade', { params });
    return response.data;
  },

  getPatterns: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/patterns', { params });
    return response.data;
  },

  getRegional: async () => {
    const response = await api.get('/analytics/regional');
    return response.data;
  },

  getSiteComparison: async (metric = 'dqi_score') => {
    const response = await api.get('/analytics/site-comparison', { params: { metric } });
    return response.data;
  },

  getRegionalPerformance: async () => {
    const response = await api.get('/analytics/regional-performance');
    return response.data;
  },

  getBottlenecks: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/bottlenecks', { params });
    return response.data;
  },

  getQualityMatrix: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/quality-matrix', { params });
    return response.data;
  },

  getResolutionStats: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/resolution-stats', { params });
    return response.data;
  },

  getLabReconciliation: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/lab-reconciliation', { params });
    return response.data;
  },

  recordResolution: async (data: { template_id: string; duration: number; success: boolean }) => {
    const response = await api.post('/analytics/resolution-stats/record', data);
    return response.data;
  },

  getRecommendations: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/analytics/recommendations', { params });
    return response.data;
  },

  refreshCache: async () => {
    const response = await api.post('/analytics/refresh');
    return response.data;
  },
};

// Intelligence API
export const intelligenceApi = {
  getHypotheses: async (params?: { sample_size?: number; study_id?: string }) => {
    const response = await api.get('/intelligence/hypotheses', { params });
    return response.data;
  },

  runSwarm: async (data: { query: string; context: Record<string, any> }) => {
    const response = await api.post('/intelligence/swarm/run', data);
    return response.data;
  },

  getAnomalies: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/intelligence/anomalies', { params });
    return response.data;
  },

  autoFix: async (params: { issue_id: number; entity_id: string }) => {
    const response = await api.post('/intelligence/auto-fix', null, { params });
    return response.data;
  },

  runAssistant: async (query: string) => {
    const response = await api.post('/intelligence/assistant/query', { query });
    return response.data;
  },
  resetAssistant: async () => {
    const response = await api.post('/intelligence/assistant/reset');
    return response.data;
  },
};

// Issues API
export const issuesApi = {
  list: async (params?: { status?: string; priority?: string; site_id?: string; study_id?: string; limit?: number }) => {
    const response = await api.get('/issues', { params });
    return response.data;
  },

  getSummary: async (studyId?: string) => {
    const params: Record<string, string> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/issues/summary', { params });
    return response.data;
  },

  getPatientSummary: async () => {
    const response = await api.get('/issues/patient-summary');
    return response.data;
  },

  create: async (data: { patient_key: string; site_id: string; issue_type: string; priority: string; description: string }) => {
    const response = await api.post('/issues', data);
    return response.data;
  },

  update: async (issueId: string | number, data: { status?: string; priority?: string; resolution_notes?: string; assigned_to?: string }) => {
    const response = await api.put(`/issues/${issueId}`, data);
    return response.data;
  },

  resolve: async (issueId: string | number, resolutionNotes?: string, reasonForChange?: string) => {
    const response = await api.post(`/issues/${issueId}/resolve`, null, {
      params: {
        resolution_notes: resolutionNotes,
        reason_for_change: reasonForChange || 'Data resolution'
      }
    });
    return response.data;
  },

  escalate: async (issueId: string | number, reason: string) => {
    const response = await api.post(`/issues/${issueId}/escalate`, null, { params: { escalation_reason: reason } });
    return response.data;
  },

  reject: async (issueId: string | number, reason: string) => {
    const response = await api.post(`/issues/${issueId}/reject`, null, { params: { reason } });
    return response.data;
  },

  analyze: async (issueId: string | number) => {
    const response = await api.get(`/issues/${issueId}/analysis`);
    return response.data;
  },
};

// Reports API
export const reportsApi = {
  getTypes: async () => {
    const response = await api.get('/reports/types');
    return response.data;
  },

  generate: async (data: { report_type: string; site_id?: string; study_id?: string; date_range_days?: number; format?: string }) => {
    const response = await api.post('/reports/generate', data);
    return response.data;
  },

  generateGet: async (reportType: string, params?: { site_id?: string; study_id?: string }) => {
    const response = await api.get(`/reports/generate/${reportType}`, { params });
    return response.data;
  },

  preview: async (reportType: string, siteId?: string) => {
    const response = await api.get(`/reports/preview/${reportType}`, { params: { site_id: siteId } });
    return response.data;
  },
};

// ML API
export const mlApi = {
  getModels: async (params?: { status?: string; model_type?: string }) => {
    const response = await api.get('/ml/models', { params });
    return response.data;
  },

  getModel: async (modelId: number) => {
    const response = await api.get(`/ml/models/${modelId}`);
    return response.data;
  },

  getSummary: async () => {
    const response = await api.get('/ml/summary');
    return response.data;
  },

  approve: async (modelId: number, data: { approved_by: string; notes?: string }) => {
    const response = await api.post(`/ml/models/${modelId}/approve`, data);
    return response.data;
  },

  deploy: async (modelId: number) => {
    const response = await api.post(`/ml/models/${modelId}/deploy`);
    return response.data;
  },

  retire: async (modelId: number, reason?: string) => {
    const response = await api.post(`/ml/models/${modelId}/retire`, null, { params: { reason } });
    return response.data;
  },

  getDriftReports: async (modelId?: string | number) => {
    const response = await api.get('/ml/drift-reports', { params: { model_id: modelId } });
    return response.data;
  },

  getAuditLog: async (modelId?: number, limit = 50) => {
    const response = await api.get('/ml/audit-log', { params: { model_id: modelId, limit } });
    return response.data;
  },

  runDriftCheck: async () => {
    const response = await api.post('/ml/run-drift-check');
    return response.data;
  },
};

// Coding API (MedDRA/WHODrug)
export const codingApi = {
  getQueue: async (params?: { dictionary?: string; status?: string; site_id?: string; study_id?: string; limit?: number }) => {
    const response = await api.get('/coding/queue', { params });
    return response.data;
  },

  getMedDRAPending: async (siteId?: string, limit = 50) => {
    const response = await api.get('/coding/meddra/pending', { params: { site_id: siteId, limit } });
    return response.data;
  },

  getWHODrugPending: async (siteId?: string, limit = 50) => {
    const response = await api.get('/coding/whodrug/pending', { params: { site_id: siteId, limit } });
    return response.data;
  },

  getStats: async (studyId?: string) => {
    const response = await api.get('/coding/stats', { params: { study_id: studyId } });
    return response.data;
  },

  approve: async (itemId: string, codedTerm: string, codedCode: string) => {
    const response = await api.post(`/coding/approve/${itemId}`, null, {
      params: { coded_term: codedTerm, coded_code: codedCode }
    });
    return response.data;
  },

  escalate: async (itemId: string, reason: string) => {
    const response = await api.post(`/coding/escalate/${itemId}`, null, { params: { reason } });
    return response.data;
  },

  search: async (dictionary: string, term: string, limit = 20) => {
    const response = await api.get(`/coding/search/${dictionary}`, { params: { term, limit } });
    return response.data;
  },

  getProductivity: async (periodDays = 30) => {
    const response = await api.get('/coding/productivity', { params: { period_days: periodDays } });
    return response.data;
  },
};

// Safety API
export const safetyApi = {
  getOverview: async (studyId?: string) => {
    const response = await api.get('/safety/overview', { params: { study_id: studyId } });
    return response.data;
  },

  getSAECases: async (params?: { status?: string; seriousness?: string; site_id?: string; limit?: number }) => {
    const response = await api.get('/safety/sae-cases', { params });
    return response.data;
  },

  getSLAStatus: async (studyId?: string) => {
    const response = await api.get('/safety/sla-status', { params: { study_id: studyId } });
    return response.data;
  },

  getSignals: async (params?: { min_strength?: number; status?: string; limit?: number }) => {
    const response = await api.get('/safety/signals', { params });
    return response.data;
  },

  getTimeline: async (days = 90, studyId?: string) => {
    const response = await api.get('/safety/timeline', { params: { days, study_id: studyId } });
    return response.data;
  },

  getNarrative: async (saeId: string) => {
    const response = await api.get(`/safety/narratives/${saeId}`);
    return response.data;
  },

  updateSAEStatus: async (saeId: string, newStatus: string, notes?: string) => {
    const response = await api.post(`/safety/sae/${saeId}/update-status`, null, {
      params: { new_status: newStatus, notes }
    });
    return response.data;
  },

  getPatternAlerts: async (severity?: string, limit = 20) => {
    const response = await api.get('/safety/pattern-alerts', { params: { severity, limit } });
    return response.data;
  },
};

// Graph API (Neo4j/Cascade)
export const graphApi = {
  getNodes: async (nodeType?: string, limit = 100) => {
    const response = await api.get('/graph/nodes', { params: { node_type: nodeType, limit } });
    return response.data;
  },

  getEdges: async (params?: { source_type?: string; target_type?: string; relationship?: string; limit?: number }) => {
    const response = await api.get('/graph/edges', { params });
    return response.data;
  },

  getCascadePath: async (issueId: string, maxDepth = 3) => {
    const response = await api.get(`/graph/cascade-path/${issueId}`, { params: { max_depth: maxDepth } });
    return response.data;
  },

  getDependencies: async (entityId: string, entityType: string, direction = 'both') => {
    const response = await api.get(`/graph/dependencies/${entityId}`, {
      params: { entity_type: entityType, direction }
    });
    return response.data;
  },

  getCascadeAnalysis: async (params?: { site_id?: string; study_id?: string; min_impact?: number; limit?: number }) => {
    const response = await api.get('/graph/cascade-analysis', { params });
    return response.data;
  },

  getVisualizationData: async (params?: { site_id?: string; study_id?: string; include_issues?: boolean; include_patients?: boolean; limit?: number }) => {
    const response = await api.get('/graph/visualization-data', { params });
    return response.data;
  },

  getCascadeIssueTypes: async (params?: { study_id?: string; site_id?: string }) => {
    const response = await api.get('/graph/cascade-issue-types', { params });
    return response.data;
  },

  getCascadeTopology: async (params?: { study_id?: string; site_id?: string }) => {
    const response = await api.get('/graph/cascade-topology', { params });
    return response.data;
  },

  seedCascadeGraph: async () => {
    const response = await api.post('/graph/cascade-seed');
    return response.data;
  },
};

// Simulation API (Digital Twin)
export const simulationApi = {
  getScenarios: async () => {
    const response = await api.get('/simulation/scenarios');
    return response.data;
  },

  run: async (data: { scenario_type: string; parameters: Record<string, unknown>; iterations?: number; confidence_level?: number }, studyId?: string) => {
    const response = await api.post('/simulation/run', data, { params: { study_id: studyId } });
    return response.data;
  },

  compare: async (scenarios: Array<{ name: string; description?: string; parameters: Record<string, unknown> }>, iterations = 1000) => {
    const response = await api.post('/simulation/compare', scenarios, { params: { iterations } });
    return response.data;
  },

  getCurrentState: async (studyId?: string) => {
    const response = await api.get('/simulation/current-state', { params: { study_id: studyId } });
    return response.data;
  },

  getProjections: async (metric: string, horizonDays = 90, studyId?: string) => {
    const response = await api.get('/simulation/projections', {
      params: { metric, horizon_days: horizonDays, study_id: studyId }
    });
    return response.data;
  },

  whatIf: async (intervention: string, magnitude: number, studyId?: string) => {
    const response = await api.get('/simulation/what-if-analysis', {
      params: { intervention, magnitude, study_id: studyId }
    });
    return response.data;
  },

  getDBLockProjection: async (studyId?: string) => {
    const params: Record<string, any> = {};
    if (studyId && studyId !== 'all') {
      params.study_id = studyId;
    }
    const response = await api.get('/simulation/db-lock-projection', {
      params
    });
    return response.data;
  },

  runWhatIf: async (scenarioType: string, entityId: string) => {
    const response = await api.post('/simulation/what-if', null, {
      params: { scenario_type: scenarioType, entity_id: entityId }
    });
    return response.data;
  },
};

// Integration API (UPR and Data Sources)
export const integrationApi = {
  getSourcesStatus: async () => {
    const response = await api.get('/integration/sources/status');
    return response.data;
  },

  getMetrics: async () => {
    const response = await api.get('/integration/metrics');
    return response.data;
  },

  getUnifiedPatientRecord: async (params?: { patient_key?: string; study_id?: string; site_id?: string; limit?: number }) => {
    const response = await api.get('/integration/unified-patient-record', { params });
    return response.data;
  },

  getSyncHistory: async (sourceId?: string, hours = 24) => {
    const response = await api.get('/integration/sync-history', { params: { source_id: sourceId, hours } });
    return response.data;
  },

  triggerSync: async (sourceId?: string) => {
    const response = await api.post('/integration/trigger-sync', null, { params: { source_id: sourceId } });
    return response.data;
  },
};

// Dashboards API (Role-based dashboards)
export const dashboardsApi = {
  getSummary: async (studyId?: string) => {
    const response = await api.get('/dashboards/summary', { params: { study_id: studyId } });
    return response.data;
  },

  getMain: async (studyId?: string) => {
    const response = await api.get('/dashboards/main', { params: { study_id: studyId } });
    return response.data;
  },

  getCRA: async (studyId?: string, siteId?: string) => {
    const response = await api.get('/dashboards/cra', { params: { study_id: studyId, site_id: siteId } });
    return response.data;
  },

  getDataManager: async (studyId?: string) => {
    const response = await api.get('/dashboards/data_manager', { params: { study_id: studyId } });
    return response.data;
  },

  getSafety: async (studyId?: string) => {
    const response = await api.get('/dashboards/safety', { params: { study_id: studyId } });
    return response.data;
  },

  getStudyLead: async (studyId?: string) => {
    const response = await api.get('/dashboards/study_lead', { params: { study_id: studyId } });
    return response.data;
  },

  getSite: async (siteId: string) => {
    const response = await api.get('/dashboards/site', { params: { site_id: siteId } });
    return response.data;
  },

  getCoder: async (studyId?: string) => {
    const response = await api.get('/dashboards/coder', { params: { study_id: studyId } });
    return response.data;
  },
};

// Agents API (6-Agent Orchestration)
export const agentsApi = {
  list: async () => {
    const response = await api.get('/agents');
    return response.data;
  },

  getSupervisorStatus: async () => {
    const response = await api.get('/agents/supervisor/status');
    return response.data;
  },

  supervisorAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/supervisor/act', { query, context });
    return response.data;
  },

  diagnosticAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/diagnostic/act', { query, context });
    return response.data;
  },

  forecasterAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/forecaster/act', { query, context });
    return response.data;
  },

  resolverAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/resolver/act', { query, context });
    return response.data;
  },

  executorAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/executor/act', { query, context });
    return response.data;
  },

  communicatorAct: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/communicator/act', { query, context });
    return response.data;
  },

  getAgentStatus: async (agentId: string) => {
    const response = await api.get(`/agents/${agentId}/status`);
    return response.data;
  },

  orchestrate: async (query: string, context?: Record<string, unknown>) => {
    const response = await api.post('/agents/orchestrate', { query, context });
    return response.data;
  },
};

// Collaboration API (Investigation Rooms, Tagging, Escalation)
export const collaborationApi = {
  listRooms: async (params?: { status?: string; room_type?: string; limit?: number }) => {
    const response = await api.get('/collaboration/rooms', { params });
    return response.data;
  },

  createRoom: async (data: { title: string; room_type?: string; description?: string; related_entity_id?: string; related_entity_type?: string; participants?: string[]; priority?: string }) => {
    const response = await api.post('/collaboration/rooms', data);
    return response.data;
  },

  getRoom: async (roomId: string) => {
    const response = await api.get(`/collaboration/rooms/${roomId}`);
    return response.data;
  },

  getRoomMessages: async (roomId: string) => {
    const response = await api.get(`/collaboration/rooms/${roomId}/messages`);
    return response.data;
  },

  postMessage: async (roomId: string, data: { content: string }) => {
    const response = await api.post(`/collaboration/rooms/${roomId}/messages`, data);
    return response.data;
  },

  tagUser: async (roomId: string, userId: string, message: string, context?: Record<string, unknown>) => {
    const response = await api.post(`/collaboration/rooms/${roomId}/tag`, { user_id: userId, message, context });
    return response.data;
  },

  escalate: async (roomId: string, escalationLevel: string, reason: string, urgency = 'standard') => {
    const response = await api.post(`/collaboration/rooms/${roomId}/escalate`, { escalation_level: escalationLevel, reason, urgency });
    return response.data;
  },

  addEvidence: async (roomId: string, evidence: Record<string, unknown>) => {
    const response = await api.post(`/collaboration/rooms/${roomId}/evidence`, evidence);
    return response.data;
  },

  castVote: async (roomId: string, vote: { type: string; option: string; comment?: string }) => {
    const response = await api.post(`/collaboration/rooms/${roomId}/vote`, vote);
    return response.data;
  },

  resolve: async (roomId: string, resolution: Record<string, unknown>) => {
    const response = await api.post(`/collaboration/rooms/${roomId}/resolve`, resolution);
    return response.data;
  },

  listWorkspaces: async () => {
    const response = await api.get('/collaboration/workspaces');
    return response.data;
  },
};

// Digital Twin API (Status and State)
export const digitalTwinApi = {
  getStatus: async () => {
    const response = await api.get('/digital-twin/status');
    return response.data;
  },

  getState: async (studyId?: string, includeProjections = false) => {
    const response = await api.get('/digital-twin/state', { params: { study_id: studyId, include_projections: includeProjections } });
    return response.data;
  },

  getSnapshots: async (hours = 24) => {
    const response = await api.get('/digital-twin/snapshots', { params: { hours } });
    return response.data;
  },

  getChangeDetection: async (threshold = 0.05) => {
    const response = await api.get('/digital-twin/change-detection', { params: { threshold } });
    return response.data;
  },

  simulate: async (scenarioType: string, entityId: string, action = 'simulate') => {
    const response = await api.post('/digital-twin/simulate', null, { params: { scenario_type: scenarioType, entity_id: entityId, action } });
    return response.data;
  },

  getScenarios: async (category?: string) => {
    const response = await api.get('/digital-twin/scenarios', { params: { category } });
    return response.data;
  },

  getEntities: async (type: string) => {
    const response = await api.get(`/digital-twin/entities/${type}`);
    return response.data;
  },

  predict: async (scenarioId: string, entityType: string, entityId: string, params?: Record<string, any>) => {
    const response = await api.post('/digital-twin/predict', params || {}, {
      params: { scenario_id: scenarioId, entity_type: entityType, entity_id: entityId }
    });
    return response.data;
  },

  getResourceRecommendations: async (region?: string) => {
    const response = await api.get('/digital-twin/resource-recommendations', { params: { region } });
    return response.data;
  },
};

export default api;
