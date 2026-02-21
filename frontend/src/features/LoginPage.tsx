import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';
import { authApi } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, AlertCircle, Eye, EyeOff, Lock, User, ShieldCheck, Database, LayoutDashboard, Building2, FileSearch, Shield } from 'lucide-react';

export default function LoginPage() {
  const navigate = useNavigate();
  const { setTokens, isAuthenticated } = useAuthStore();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await authApi.login(username, password);
      setTokens(response.access_token, response.refresh_token, response.user);
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Invalid username or password');
    } finally {
      setIsLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const setDemoUser = (user: string, pass: string) => {
    setUsername(user);
    setPassword(pass);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] flex items-center justify-center p-4 relative overflow-hidden">
      {/* DNA Helix Video Background */}
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover opacity-30 z-0"
      >
        <source src="/dna_helix_video_animation_landing_page.mp4" type="video/mp4" />
      </video>
      {/* Dark overlay for readability */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a0c]/60 via-[#0a0a0c]/40 to-[#0a0a0c]/80 z-0" />

      <div className="w-full max-w-md z-10">
        {/* Logo */}
        <div className="text-center mb-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <img
            src="/logo_with_text.png"
            alt="Sanchalak AI"
            className="h-24 mx-auto mb-6 drop-shadow-2xl hover:scale-105 transition-transform duration-500"
          />
          <p className="text-gray-400 font-medium text-sm tracking-wide">Orchestrating the Future of Clinical Trials</p>
        </div>

        {/* Login Card */}
        <Card className="bg-nexus-card/50 backdrop-blur-xl border-nexus-border shadow-2xl animate-in fade-in zoom-in-95 duration-500 delay-150">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center text-white">Identity Access</CardTitle>
            <CardDescription className="text-center text-gray-400">
              Secure biometric-grade authentication required
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-sm animate-in fade-in shake duration-300">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="username" className="text-gray-300">Workstation ID</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    id="username"
                    type="text"
                    placeholder="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    className="pl-10 bg-nexus-bg/50 border-nexus-border text-white focus:ring-indigo-500/50"
                    autoFocus
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label htmlFor="password" className="text-gray-300">Access Key</Label>
                  <button
                    type="button"
                    onClick={() => alert("Please contact your Protocol Administrator to reset your workstation access key.")}
                    className="text-xs text-indigo-400 hover:text-indigo-300"
                  >
                    Forgot key?
                  </button>
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="pl-10 pr-10 bg-nexus-bg/50 border-nexus-border text-white focus:ring-indigo-500/50"
                  />
                  <button
                    type="button"
                    onClick={togglePasswordVisibility}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 focus:outline-none"
                  >
                    {showPassword ? (
                      <EyeOff className="w-4 h-4" />
                    ) : (
                      <Eye className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                className="w-full bg-gradient-to-r from-indigo-600 to-violet-700 hover:from-indigo-500 hover:to-violet-600 text-white font-bold h-11 transition-all duration-300 shadow-lg shadow-indigo-500/20"
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    Authenticating...
                  </div>
                ) : (
                  'Establish Connection'
                )}
              </Button>
            </form>

            {/* Quick Access Roles */}
            <div className="mt-8">
              <div className="relative mb-6">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t border-nexus-border" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-[#121214] px-2 text-gray-500 font-medium tracking-widest">Protocol Shortcuts</span>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2">
                <button
                  onClick={() => setDemoUser('testuser', 'testpassword')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300"
                >
                  <Activity className="w-5 h-5 text-indigo-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-indigo-300 uppercase">Test</span>
                </button>
                <button
                  onClick={() => setDemoUser('lead', 'lead123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300"
                >
                  <ShieldCheck className="w-5 h-5 text-indigo-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-indigo-300 uppercase">Lead</span>
                </button>
                <button
                  onClick={() => setDemoUser('dm', 'dm123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-emerald-500/10 hover:border-emerald-500/30 transition-all duration-300"
                >
                  <Database className="w-5 h-5 text-emerald-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-emerald-300 uppercase">Data</span>
                </button>
                <button
                  onClick={() => setDemoUser('exec', 'exec123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-amber-500/10 hover:border-amber-500/30 transition-all duration-300"
                >
                  <LayoutDashboard className="w-5 h-5 text-amber-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-amber-300 uppercase">Exec</span>
                </button>
                <button
                  onClick={() => setDemoUser('cra', 'cra123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-blue-500/10 hover:border-blue-500/30 transition-all duration-300"
                >
                  <Building2 className="w-5 h-5 text-blue-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-blue-300 uppercase">CRA</span>
                </button>
                <button
                  onClick={() => setDemoUser('coder', 'coder123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-purple-500/10 hover:border-purple-500/30 transition-all duration-300"
                >
                  <FileSearch className="w-5 h-5 text-purple-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-purple-300 uppercase">Coder</span>
                </button>
                <button
                  onClick={() => setDemoUser('safety', 'safety123')}
                  className="group flex flex-col items-center gap-2 p-3 rounded-xl bg-nexus-bg/40 border border-nexus-border hover:bg-red-500/10 hover:border-red-500/30 transition-all duration-300"
                >
                  <Shield className="w-5 h-5 text-red-400 group-hover:scale-110 transition-transform" />
                  <span className="text-[10px] font-bold text-gray-400 group-hover:text-red-300 uppercase">Safety</span>
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Status Footer */}
        <div className="mt-8 flex items-center justify-between text-[10px] text-gray-500 font-mono tracking-tighter uppercase px-2">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            NODE-ALPHA-7 ACTIVE
          </div>
          <div>ESTABLISHING ENCRYPTED TUNNEL...</div>
          <div>v10.0.0-PROD</div>
        </div>
      </div>
    </div>
  );
}

