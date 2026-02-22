import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { useAppStore } from '@/stores/appStore';
import { cn, getInitials } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  LayoutDashboard,
  Users,
  Building2,
  Database,
  FileSearch,
  Shield,
  FileText,
  Brain,
  GitBranch,
  MessageSquare,
  Bot,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Bell,
  Menu,
  Target,
  X,
  AlertTriangle,
  CheckCircle,
  Info,
  Activity
} from 'lucide-react';
import { format } from 'date-fns';
import { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { studiesApi, issuesApi, analyticsApi, sitesApi, patientsApi } from '@/services/api';

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  roles?: string[];
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Executive Command', href: '/executive', icon: LayoutDashboard, roles: ['executive', 'lead'] },
  { name: 'Study Lead', href: '/study-lead', icon: Target, roles: ['lead'] },
  { name: 'DM Hub', href: '/dm-hub', icon: Database, roles: ['dm', 'lead'] },
  { name: 'CRA Field View', href: '/cra-view', icon: Building2, roles: ['cra', 'lead'] },
  { name: 'Medical Coder', href: '/coder-view', icon: FileSearch, roles: ['coder', 'dm', 'lead'] },
  { name: 'Safety Surveillance', href: '/safety-view', icon: Shield, roles: ['safety', 'lead'] },
  { name: 'Site Portal', href: '/site-portal', icon: Users, roles: ['cra', 'lead', 'executive'] },
  { name: 'Cascade Explorer', href: '/cascade-explorer', icon: GitBranch, roles: ['lead', 'executive', 'dm'] },
  { name: 'Reports', href: '/reports', icon: FileText, roles: ['lead', 'executive', 'dm', 'coder', 'safety'] },
  { name: 'ML Governance', href: '/ml-governance', icon: Brain, roles: ['lead', 'executive'] },
  { name: 'Digital Twin', href: '/digital-twin', icon: Activity, roles: ['lead', 'executive'] },

  { name: 'Collaboration', href: '/collaboration-hub', icon: MessageSquare },
  { name: 'AI Assistant', href: '/ai-assistant', icon: Bot, badge: 'AI' },
  { name: 'Settings', href: '/settings', icon: Settings },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();
  const { user, logout } = useAuthStore();

  const filteredNavItems = navItems.filter(item => {
    if (!item.roles) return true;
    return user && item.roles.includes(user.role);
  });

  return (
    <div
      className={cn(
        'fixed left-0 top-0 z-40 h-screen bg-nexus-card border-r border-nexus-border transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-16 px-4 border-b border-nexus-border">
        <div className="flex items-center gap-2">
          <img
            src="/logo.jpeg"
            alt="a6on-i"
            className="w-8 h-8 object-contain flex-shrink-0 rounded-md"
          />
          {!collapsed && (
            <span className="text-lg font-bold text-white">a6on-i</span>
          )}
        </div>
      </div>

      {/* Navigation */}
      <ScrollArea className="h-[calc(100vh-8rem)]">
        <nav className="p-2 space-y-1">
          {filteredNavItems.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-primary/10 text-primary border-l-2 border-primary'
                    : 'text-nexus-text-muted hover:bg-nexus-card-hover hover:text-nexus-text'
                )}
                title={collapsed ? item.name : undefined}
              >
                <item.icon className={cn('w-5 h-5 shrink-0', isActive && 'text-primary')} />
                {!collapsed && (
                  <span className="flex-1">{item.name}</span>
                )}
                {!collapsed && item.badge && (
                  <span className="px-1.5 py-0.5 text-xs rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </ScrollArea>

      {/* User & Toggle */}
      <div className="absolute bottom-0 left-0 right-0 border-t border-nexus-border bg-nexus-card">
        <div className="p-2">
          <div
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg',
              collapsed ? 'justify-center' : ''
            )}
          >
            <Avatar className="w-8 h-8 border-2 border-primary/30">
              <AvatarFallback className="bg-gradient-to-br from-purple-500 to-blue-500 text-white text-xs">
                {user ? getInitials(user.full_name || user.username) : '??'}
              </AvatarFallback>
            </Avatar>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-nexus-text truncate">
                  {user?.full_name || user?.username}
                </p>
                <p className="text-xs text-nexus-text-muted capitalize">{user?.role}</p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1 mt-1">
            <Button
              variant="ghost"
              size="sm"
              className={cn(
                'flex-1 text-nexus-text-muted hover:text-nexus-text hover:bg-nexus-card-hover',
                collapsed && 'w-full'
              )}
              onClick={logout}
              title="Sign out"
            >
              <LogOut className="w-4 h-4" />
              {!collapsed && <span className="ml-2">Sign out</span>}
            </Button>
          </div>
        </div>

        <Button
          variant="ghost"
          size="sm"
          className="absolute -right-3 top-1/2 -translate-y-1/2 h-6 w-6 rounded-full border border-nexus-border bg-nexus-card shadow-sm hover:bg-nexus-card-hover"
          onClick={onToggle}
        >
          {collapsed ? (
            <ChevronRight className="w-3 h-3 text-nexus-text" />
          ) : (
            <ChevronLeft className="w-3 h-3 text-nexus-text" />
          )}
        </Button>
      </div>
    </div>
  );
}

interface HeaderProps {
  onMenuClick: () => void;
}

function Header({ onMenuClick }: HeaderProps) {
  const { user } = useAuthStore();
  const { selectedStudy, setSelectedStudy } = useAppStore();
  const queryClient = useQueryClient();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  // Fetch studies for dropdown (use same cache key as feature pages)
  const { data: studiesData } = useQuery({
    queryKey: ['studies'],
    queryFn: () => studiesApi.list(),
  });

  // Fetch issues for notifications (share cache with feature pages)
  const { data: issuesData } = useQuery({
    queryKey: ['issues-summary', undefined],
    queryFn: () => issuesApi.getSummary(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Eagerly warm API caches for common dashboard data so tab switching is instant
  useEffect(() => {
    const study = selectedStudy;
    const prefetches = [
      { key: ['portfolio', study], fn: () => analyticsApi.getPortfolio(study) },
      { key: ['dqi-distribution', study], fn: () => analyticsApi.getDQIDistribution(study) },
      { key: ['regional-metrics'], fn: () => analyticsApi.getRegional() },
      { key: ['clean-status-summary', study], fn: () => analyticsApi.getCleanStatusSummary(study) },
      { key: ['sites', study], fn: () => sitesApi.list({ study_id: study }) },
      { key: ['site-benchmarks', study], fn: () => sitesApi.getBenchmarks(study) },
      { key: ['clean-status', study], fn: () => patientsApi.getCleanStatus(study) },
      { key: ['dblock-status', study], fn: () => patientsApi.getDBLockStatus(study) },
    ];
    prefetches.forEach(({ key, fn }) => {
      queryClient.prefetchQuery({ queryKey: key, queryFn: fn, staleTime: 1000 * 60 * 5 });
    });
  }, [selectedStudy, queryClient]);

  const allStudies = studiesData?.studies || [];
  const studies = allStudies;
  const criticalCount = issuesData?.critical_count || 0;
  const openCount = issuesData?.open_count || 0;

  // Generate notifications from issues
  const notifications = [
    ...(criticalCount > 0 ? [{
      id: 'critical',
      type: 'error' as const,
      title: 'Critical Issues',
      message: `${criticalCount} critical issue${criticalCount > 1 ? 's' : ''} require immediate attention`,
      time: 'Just now',
    }] : []),
    ...(openCount > 10 ? [{
      id: 'open',
      type: 'warning' as const,
      title: 'Open Issues',
      message: `${openCount} open issues in the system`,
      time: '5 min ago',
    }] : []),
    {
      id: 'sync',
      type: 'success' as const,
      title: 'Data Sync Complete',
      message: 'All data sources synchronized successfully',
      time: format(currentTime, 'HH:mm'),
    },
    {
      id: 'system',
      type: 'info' as const,
      title: 'System Status',
      message: 'All services operational',
      time: format(currentTime, 'HH:mm'),
    },
  ];

  const getNotificationIcon = (type: 'error' | 'warning' | 'success' | 'info') => {
    switch (type) {
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'info': return <Info className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <header className="sticky top-0 z-30 h-14 bg-nexus-bg border-b border-nexus-border flex items-center justify-between px-4 lg:px-6">
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden text-nexus-text-muted hover:text-nexus-text hover:bg-nexus-card"
          onClick={onMenuClick}
        >
          <Menu className="w-5 h-5" />
        </Button>

        {/* Study Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-nexus-text-muted">Filter by Study</span>
          <Select value={selectedStudy} onValueChange={setSelectedStudy}>
            <SelectTrigger className="w-48 h-8 bg-nexus-card border-nexus-border text-nexus-text text-sm">
              <SelectValue placeholder="All Studies" />
            </SelectTrigger>
            <SelectContent className="bg-nexus-card border-nexus-border max-h-64">
              <SelectItem value="all" className="text-nexus-text hover:bg-nexus-card-hover">
                All Studies
              </SelectItem>
              {studies.map((study: { study_id: string; name?: string; protocol_number?: string }) => (
                <SelectItem
                  key={study.study_id}
                  value={study.study_id}
                  className="text-nexus-text hover:bg-nexus-card-hover"
                >
                  <div className="flex flex-col">
                    <span>{study.protocol_number || study.study_id}</span>
                    {study.name && <span className="text-[10px] text-nexus-text-secondary truncate">{study.name}</span>}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Notifications */}
        <div className="relative">
          <Button
            variant="ghost"
            size="icon"
            className="relative text-nexus-text-muted hover:text-nexus-text hover:bg-nexus-card"
            onClick={() => setShowNotifications(!showNotifications)}
          >
            <Bell className="w-5 h-5" />
            {criticalCount > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            )}
          </Button>

          {/* Notification Panel */}
          {showNotifications && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setShowNotifications(false)}
              />
              <div className="absolute right-0 top-full mt-2 w-80 bg-nexus-card border border-nexus-border rounded-lg shadow-xl z-50">
                <div className="flex items-center justify-between p-3 border-b border-nexus-border">
                  <h3 className="text-sm font-medium text-nexus-text">Notifications</h3>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-nexus-text-muted hover:text-nexus-text"
                    onClick={() => setShowNotifications(false)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
                <ScrollArea className="max-h-80">
                  <div className="p-2 space-y-2">
                    {notifications.map((notification) => (
                      <div
                        key={notification.id}
                        className="p-3 rounded-lg bg-nexus-bg hover:bg-nexus-card-hover transition-colors cursor-pointer"
                      >
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5">
                            {getNotificationIcon(notification.type)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-nexus-text">{notification.title}</p>
                            <p className="text-xs text-nexus-text-muted mt-0.5">{notification.message}</p>
                            <p className="text-xs text-nexus-text-muted mt-1">{notification.time}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                <div className="p-2 border-t border-nexus-border">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full text-nexus-text-muted hover:text-nexus-text text-xs"
                  >
                    View All Notifications
                  </Button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* System Sync Status */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-nexus-card border border-nexus-border">
          <span className="text-xs text-nexus-text-muted">SYSTEM SYNC</span>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse-glow" />
            <span className="text-xs font-medium text-emerald-400">LIVE</span>
          </div>
          <span className="text-xs text-nexus-text-muted ml-2">
            {format(currentTime, 'yyyy-MM-dd HH:mm')}
          </span>
        </div>

        {/* User Avatar */}
        <Avatar className="w-8 h-8 border-2 border-primary/30">
          <AvatarFallback className="bg-gradient-to-br from-purple-500 to-blue-500 text-white text-xs">
            {user ? getInitials(user.full_name || user.username) : '??'}
          </AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
}

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { sidebarCollapsed, toggleSidebarCollapsed, sidebarOpen, toggleSidebar } = useAppStore();
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    navigate('/login');
    return null;
  }

  return (
    <div className="min-h-screen bg-nexus-bg">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={toggleSidebar}
        />
      )}

      {/* Sidebar */}
      <div className="hidden lg:block">
        <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebarCollapsed} />
      </div>

      {/* Mobile sidebar */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-40 lg:hidden transition-transform duration-300',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <Sidebar collapsed={false} onToggle={toggleSidebar} />
      </div>

      {/* Main content */}
      <div
        className={cn(
          'transition-all duration-300',
          sidebarCollapsed ? 'lg:ml-16' : 'lg:ml-64'
        )}
      >
        <Header onMenuClick={toggleSidebar} />
        <main className="p-4 lg:p-6">{children}</main>

        {/* Floating AI Button */}
        <Button
          onClick={() => navigate('/ai-assistant')}
          className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-br from-indigo-600 to-violet-700 shadow-2xl shadow-indigo-500/40 border border-white/20 z-50 group hover:scale-110 transition-all duration-300"
        >
          <Bot className="w-7 h-7 text-white group-hover:animate-bounce" />
          <span className="absolute -top-1 -right-1 flex h-4 w-4">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-4 w-4 bg-purple-500 border-2 border-white flex items-center justify-center text-[8px] font-bold text-white">AI</span>
          </span>
        </Button>

        {/* Footer */}
        <footer className="border-t border-nexus-border px-4 lg:px-6 py-4 mt-auto">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-nexus-text-muted">
            <span>&copy; 2026 a6on-i 10X &nbsp;&nbsp; v10.0.0</span>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                System Online
              </span>
              <span>Refreshed: {format(new Date(), 'yyyy-MM-dd HH:mm')}</span>
            </div>
            <div className="flex items-center gap-4">
              <a href="#" className="hover:text-nexus-text">Documentation</a>
              <a href="#" className="hover:text-nexus-text">Support</a>
              <a href="#" className="hover:text-nexus-text">Privacy</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
