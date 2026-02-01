import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { TooltipProvider } from '@/components/ui/tooltip';
import Layout from '@/components/Layout';
import StartupLoader from '@/components/StartupLoader';
import LoginPage from '@/features/LoginPage';
import ExecutiveOverview from '@/features/ExecutiveOverview';
import StudyLead from '@/features/StudyLead';
import DMHub from '@/features/DMHub';
import CRAView from '@/features/CRAView';
import CoderView from '@/features/CoderView';
import SafetyView from '@/features/SafetyView';
import SitePortal from '@/features/SitePortal';
import Reports from '@/features/Reports';
import MLGovernance from '@/features/MLGovernance';
import CascadeExplorer from '@/features/CascadeExplorer';
import HypothesisExplorer from '@/features/HypothesisExplorer';
import CollaborationHub from '@/features/CollaborationHub';
import AIAssistant from '@/features/AIAssistant';
import SettingsPage from '@/features/SettingsPage';
import VisualizationDashboard from '@/features/VisualizationDashboard';


function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[#0a0a0c]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
          <p className="text-indigo-400 font-mono text-xs uppercase tracking-widest animate-pulse">Initializing NEXUS...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Layout>{children}</Layout>;
}

function RoleBasedRedirect() {
  const { user } = useAuthStore();

  if (!user) return <Navigate to="/login" replace />;

  switch (user.role) {
    case 'lead':
      return <Navigate to="/study-lead" replace />;
    case 'dm':
      return <Navigate to="/dm-hub" replace />;
    case 'cra':
      return <Navigate to="/cra-view" replace />;
    case 'coder':
      return <Navigate to="/coder-view" replace />;
    case 'safety':
      return <Navigate to="/safety-view" replace />;
    case 'executive':
      return <Navigate to="/executive" replace />;
    default:
      return <Navigate to="/executive" replace />;
  }
}

export default function App() {
  const { isAuthenticated } = useAuthStore();

  return (
    <StartupLoader>
      <TooltipProvider>
        <Routes>
          <Route
            path="/login"
            element={
              isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
            }
          />

          <Route
            path="/"
            element={
              <ProtectedRoute>
                <RoleBasedRedirect />
              </ProtectedRoute>
            }
          />

          <Route
            path="/executive"
            element={
              <ProtectedRoute>
                <ExecutiveOverview />
              </ProtectedRoute>
            }
          />

          <Route
            path="/study-lead"
            element={
              <ProtectedRoute>
                <StudyLead />
              </ProtectedRoute>
            }
          />

          <Route
            path="/dm-hub"
            element={
              <ProtectedRoute>
                <DMHub />
              </ProtectedRoute>
            }
          />

          <Route
            path="/cra-view"
            element={
              <ProtectedRoute>
                <CRAView />
              </ProtectedRoute>
            }
          />

          <Route
            path="/coder-view"
            element={
              <ProtectedRoute>
                <CoderView />
              </ProtectedRoute>
            }
          />

          <Route
            path="/safety-view"
            element={
              <ProtectedRoute>
                <SafetyView />
              </ProtectedRoute>
            }
          />

          <Route
            path="/site-portal"
            element={
              <ProtectedRoute>
                <SitePortal />
              </ProtectedRoute>
            }
          />

          <Route
            path="/reports"
            element={
              <ProtectedRoute>
                <Reports />
              </ProtectedRoute>
            }
          />

          <Route
            path="/ml-governance"
            element={
              <ProtectedRoute>
                <MLGovernance />
              </ProtectedRoute>
            }
          />

          <Route
            path="/cascade-explorer"
            element={
              <ProtectedRoute>
                <CascadeExplorer />
              </ProtectedRoute>
            }
          />

          <Route
            path="/hypothesis-explorer"
            element={
              <ProtectedRoute>
                <HypothesisExplorer />
              </ProtectedRoute>
            }
          />

          <Route
            path="/collaboration-hub"
            element={
              <ProtectedRoute>
                <CollaborationHub />
              </ProtectedRoute>
            }
          />

          <Route
            path="/ai-assistant"
            element={
              <ProtectedRoute>
                <AIAssistant />
              </ProtectedRoute>
            }
          />

          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <SettingsPage />
              </ProtectedRoute>
            }
          />

          <Route
            path="/visualization"
            element={
              <ProtectedRoute>
                <VisualizationDashboard />
              </ProtectedRoute>
            }
          />

          {/* Catch all - redirect to home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </TooltipProvider>
    </StartupLoader>
  );
}
