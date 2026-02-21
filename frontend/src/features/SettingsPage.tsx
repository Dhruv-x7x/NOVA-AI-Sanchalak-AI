import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  User,
  Bell,
  Shield,
  Database,
  Palette,
} from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export default function SettingsPage() {
  const { user } = useAuthStore();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-gray-500">Manage your account and preferences</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Sidebar Navigation */}
        <Card className="h-fit">
          <CardContent className="p-4">
            <nav className="space-y-1">
              {[
                { icon: User, label: 'Profile', active: true },
                { icon: Bell, label: 'Notifications' },
                { icon: Shield, label: 'Security' },
                { icon: Database, label: 'Data' },
                { icon: Palette, label: 'Appearance' },
              ].map((item) => (
                <Button
                  key={item.label}
                  variant={item.active ? 'secondary' : 'ghost'}
                  className="w-full justify-start"
                >
                  <item.icon className="w-4 h-4 mr-2" />
                  {item.label}
                </Button>
              ))}
            </nav>
          </CardContent>
        </Card>

        {/* Main Content */}
        <div className="md:col-span-2 space-y-6">
          {/* Profile Section */}
          <Card>
            <CardHeader>
              <CardTitle>Profile</CardTitle>
              <CardDescription>Your personal information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" value={user?.username || ''} disabled />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" value={user?.email || ''} disabled />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input id="fullName" value={user?.full_name || ''} disabled />
              </div>

              <div className="space-y-2">
                <Label>Role</Label>
                <div>
                  <Badge variant="info" className="capitalize">{user?.role}</Badge>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Permissions</Label>
                <div className="flex flex-wrap gap-2">
                  {user?.permissions?.map((perm) => (
                    <Badge key={perm} variant="secondary">{perm}</Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notifications Section */}
          <Card>
            <CardHeader>
              <CardTitle>Notifications</CardTitle>
              <CardDescription>Configure your notification preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Email Notifications</p>
                  <p className="text-sm text-gray-500">Receive email updates about issues</p>
                </div>
                <Select defaultValue="daily">
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="realtime">Real-time</SelectItem>
                    <SelectItem value="daily">Daily</SelectItem>
                    <SelectItem value="weekly">Weekly</SelectItem>
                    <SelectItem value="off">Off</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Critical Alerts</p>
                  <p className="text-sm text-gray-500">Get notified for critical issues</p>
                </div>
                <Select defaultValue="realtime">
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="realtime">Real-time</SelectItem>
                    <SelectItem value="off">Off</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Autonomy Section */}
          <Card>
            <CardHeader>
              <CardTitle>AI Autonomy & Governance</CardTitle>
              <CardDescription>Control human-in-the-loop requirements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Human-in-the-Loop Enforcement</p>
                  <p className="text-sm text-gray-500">Require approval for critical resolution actions</p>
                </div>
                <Select defaultValue="always">
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="always">Always</SelectItem>
                    <SelectItem value="high_risk">High Risk Only</SelectItem>
                    <SelectItem value="never">Never (Full Autonomy)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Agent Risk Threshold</p>
                  <p className="text-sm text-gray-500">Confidence level required for autonomous execution</p>
                </div>
                <Badge variant="outline" className="text-nexus-text-secondary">0.85</Badge>
              </div>
            </CardContent>
          </Card>

          {/* System Info */}
          <Card>
            <CardHeader>
              <CardTitle>System Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 text-sm">
                <div>
                  <p className="text-gray-500">Version</p>
                  <p className="font-medium">1.0.0</p>
                </div>
                <div>
                  <p className="text-gray-500">API Status</p>
                  <Badge variant="success">Connected</Badge>
                </div>
                <div>
                  <p className="text-gray-500">Database</p>
                  <Badge variant="success">Healthy</Badge>
                </div>
                <div>
                  <p className="text-gray-500">Last Sync</p>
                  <p className="font-medium">Just now</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
