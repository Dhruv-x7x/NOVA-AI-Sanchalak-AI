import { lazy, Suspense, useEffect, Component, ReactNode } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { TooltipProvider } from '@/components/ui/tooltip';
import Layout from '@/components/Layout';
import LoginPage from '@/features/LoginPage';
import SanchalakLoader from '@/components/SanchalakLoader';

class ErrorBoundary extends Component<{children: ReactNode}, {hasError: boolean}> {
  constructor(props: {children: ReactNode}) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-screen flex items-center justify-center bg-[#0a0a0c] text-white p-10">
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Module Loading Error</h2>
            <p className="text-nexus-text-secondary mb-6">Failed to load the requested intelligence module. This may be due to a network issue or a system update.</p>
            <button 
              onClick={() => window.location.reload()} 
              className="px-6 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-500 font-bold"
            >
              Reload Sanchalak AI
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// Lazy-loaded feature pages — each becomes a separate JS chunk
const ExecutiveOverview = lazy(() => import('@/features/ExecutiveOverview'));
const StudyLead = lazy(() => import('@/features/StudyLead'));
const DMHub = lazy(() => import('@/features/DMHub'));
const CRAView = lazy(() => import('@/features/CRAView'));
const CoderView = lazy(() => import('@/features/CoderView'));
const SafetyView = lazy(() => import('@/features/SafetyView'));
const SitePortal = lazy(() => import('@/features/SitePortal'));
const Reports = lazy(() => import('@/features/Reports'));
const MLGovernance = lazy(() => import('@/features/MLGovernance'));
const CascadeExplorer = lazy(() => import('@/features/CascadeExplorer'));
const HypothesisExplorer = lazy(() => import('@/features/HypothesisExplorer'));
const CollaborationHub = lazy(() => import('@/features/CollaborationHub'));
const AIAssistant = lazy(() => import('@/features/AIAssistant'));
const SettingsPage = lazy(() => import('@/features/SettingsPage'));
const DigitalTwin = lazy(() => import('@/features/DigitalTwin'));

function PageLoader() {
  return <SanchalakLoader size="lg" label="Loading module..." fullPage />;
}

// Prefetch common route chunks immediately after authentication
function prefetchRoutes() {
  const routes = [
    () => import('@/features/ExecutiveOverview'),
    () => import('@/features/StudyLead'),
    () => import('@/features/DMHub'),
    () => import('@/features/CRAView'),
    () => import('@/features/CoderView'),
    () => import('@/features/SafetyView'),
    () => import('@/features/SitePortal'),
    () => import('@/features/Reports'),
    () => import('@/features/MLGovernance'),
    () => import('@/features/CascadeExplorer'),
    () => import('@/features/AIAssistant'),
  ];
  setTimeout(() => {
    routes.forEach((importFn) => importFn().catch(() => {}));
  }, 100);
}


function ProtectedRoute({ children, allowedRoles }: { children: React.ReactNode; allowedRoles?: string[] }) {
  const { isAuthenticated, isLoading, user } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated) {
      prefetchRoutes();
    }
  }, [isAuthenticated]);

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[#0a0a0c]">
        <SanchalakLoader size="xl" label="Initializing Sanchalak AI..." />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && user && !allowedRoles.includes(user.role)) {
    return <Navigate to="/" replace />;
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
    <TooltipProvider>
      <ErrorBoundary>
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
            <ProtectedRoute allowedRoles={['executive', 'lead']}>
              <Suspense fallback={<PageLoader />}>
                <ExecutiveOverview />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/study-lead"
          element={
            <ProtectedRoute allowedRoles={['lead']}>
              <Suspense fallback={<PageLoader />}>
                <StudyLead />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/dm-hub"
          element={
            <ProtectedRoute allowedRoles={['dm', 'lead']}>
              <Suspense fallback={<PageLoader />}>
                <DMHub />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/cra-view"
          element={
            <ProtectedRoute allowedRoles={['cra', 'lead']}>
              <Suspense fallback={<PageLoader />}>
                <CRAView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/coder-view"
          element={
            <ProtectedRoute allowedRoles={['coder', 'dm', 'lead']}>
              <Suspense fallback={<PageLoader />}>
                <CoderView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/safety-view"
          element={
            <ProtectedRoute allowedRoles={['safety', 'lead']}>
              <Suspense fallback={<PageLoader />}>
                <SafetyView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/site-portal"
          element={
            <ProtectedRoute allowedRoles={['cra', 'lead', 'executive']}>
              <Suspense fallback={<PageLoader />}>
                <SitePortal />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/reports"
          element={
            <ProtectedRoute allowedRoles={['lead', 'executive', 'dm', 'coder', 'safety']}>
              <Suspense fallback={<PageLoader />}>
                <Reports />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/ml-governance"
          element={
            <ProtectedRoute allowedRoles={['lead', 'executive']}>
              <Suspense fallback={<PageLoader />}>
                <MLGovernance />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/cascade-explorer"
          element={
            <ProtectedRoute allowedRoles={['lead', 'executive', 'dm']}>
              <Suspense fallback={<PageLoader />}>
                <CascadeExplorer />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/digital-twin"
          element={
            <ProtectedRoute allowedRoles={['lead', 'executive']}>
              <Suspense fallback={<PageLoader />}>
                <DigitalTwin />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/hypothesis-explorer"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <HypothesisExplorer />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/collaboration-hub"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <CollaborationHub />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/ai-assistant"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <AIAssistant />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <SettingsPage />
              </Suspense>
            </ProtectedRoute>
          }
        />

        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </ErrorBoundary>
    </TooltipProvider>
  );
}
