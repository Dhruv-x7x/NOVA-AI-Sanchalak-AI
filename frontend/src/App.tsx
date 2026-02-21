import { lazy, Suspense, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { TooltipProvider } from '@/components/ui/tooltip';
import Layout from '@/components/Layout';
import LoginPage from '@/features/LoginPage';
import SanchalakLoader from '@/components/SanchalakLoader';

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
// This eliminates the lazy-load delay when navigating to a page for the first time
function prefetchRoutes() {
  // Fire all imports in parallel — browser handles them without blocking the main thread
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
  // Small delay to let the current route render first, then prefetch the rest
  setTimeout(() => {
    routes.forEach((importFn) => importFn().catch(() => {}));
  }, 100);
}


function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();

  // Prefetch all common route chunks once authenticated
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
              <Suspense fallback={<PageLoader />}>
                <ExecutiveOverview />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/study-lead"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <StudyLead />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/dm-hub"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <DMHub />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/cra-view"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <CRAView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/coder-view"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <CoderView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/safety-view"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <SafetyView />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/site-portal"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <SitePortal />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/reports"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <Reports />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/ml-governance"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <MLGovernance />
              </Suspense>
            </ProtectedRoute>
          }
        />

        <Route
          path="/cascade-explorer"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <CascadeExplorer />
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

        <Route
          path="/digital-twin"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <DigitalTwin />
              </Suspense>
            </ProtectedRoute>
          }
        />

        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </TooltipProvider>
  );
}
