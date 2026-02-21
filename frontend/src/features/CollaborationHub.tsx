import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collaborationApi } from '@/services/api';
import { useAuthStore } from '@/stores/authStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  MessageSquare,
  Users,
  Send,
  Plus,
  Search,
  ArrowUpCircle,
  AtSign,
  X,
  Loader2,
} from 'lucide-react';
import { getInitials, cn } from '@/lib/utils';

// Move static data outside or to the top to avoid ReferenceErrors
const TEAM_MEMBERS = [
  { id: '1', name: 'Dr. Sarah Chen', role: 'Study Lead', status: 'online', lastActive: 'Now' },
  { id: '2', name: 'Michael Thompson', role: 'CRA', status: 'online', lastActive: 'Now' },
  { id: '3', name: 'Emily Rodriguez', role: 'Data Manager', status: 'online', lastActive: '2 min ago' },
  { id: '4', name: 'James Wilson', role: 'Medical Coder', status: 'away', lastActive: '15 min ago' },
  { id: '5', name: 'Maria Santos', role: 'Safety Officer', status: 'online', lastActive: 'Now' },
  { id: '6', name: 'David Kim', role: 'Executive', status: 'offline', lastActive: '1 hour ago' },
];

const ESCALATIONS = [
  { id: 'ESC-001', title: 'Critical SAE - Expedited Report Due', severity: 'critical', status: 'in_progress', slaRemaining: '11h 15m' },
  { id: 'ESC-002', title: 'Site BR-201 Enrollment Hold', severity: 'high', status: 'acknowledged', slaRemaining: '40h 45m' },
  { id: 'ESC-003', title: 'Data Lock Deadline Risk', severity: 'high', status: 'pending', slaRemaining: '12h 45m' },
];

export default function CollaborationHub() {
  const [activeTab, setActiveTab] = useState('rooms');
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [messageInput, setMessageInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const messageEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const { user } = useAuthStore();

  const { data: roomsData, isLoading: roomsLoading } = useQuery({
    queryKey: ['collaboration-rooms'],
    queryFn: () => collaborationApi.listRooms(),
  });

  const rooms = roomsData?.rooms || [];
  const selectedRoom = rooms.find((r: any) => r.room_id === selectedRoomId);

  const { data: messagesData, isLoading: messagesLoading } = useQuery({
    queryKey: ['room-messages', selectedRoomId],
    queryFn: async () => {
      if (!selectedRoomId) return { messages: [] };
      const apiObj = collaborationApi as any;
      if (apiObj.getRoomMessages) {
        return await apiObj.getRoomMessages(selectedRoomId);
      }
      return { messages: [] };
    },
    enabled: !!selectedRoomId,
    refetchInterval: 20000, 
  });

  const messages = messagesData?.messages || [];

  const postMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      if (!selectedRoomId) throw new Error('No room selected');
      const apiObj = collaborationApi as any;
      if (apiObj.postMessage) {
        return await apiObj.postMessage(selectedRoomId, { content });
      }
      throw new Error('API method not implemented');
    },
    onSuccess: () => {
      setMessageInput('');
      queryClient.invalidateQueries({ queryKey: ['room-messages', selectedRoomId] });
    }
  });

  const handleSend = () => {
    if (!messageInput.trim() || postMessageMutation.isPending) return;
    postMessageMutation.mutate(messageInput);
  };

  const getPriorityColor = (priority: string) => {
    switch (String(priority).toLowerCase()) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      default: return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    }
  };

  const getStatusColor = (status: string) => {
    switch (String(status).toLowerCase()) {
      case 'active': return 'bg-green-500/20 text-green-400';
      case 'escalated': return 'bg-red-500/20 text-red-400';
      case 'resolved': return 'bg-gray-500/20 text-gray-400';
      default: return 'bg-blue-500/20 text-blue-400';
    }
  };

  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-xl p-6 border border-nexus-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Collaboration Hub</h1>
              <p className="text-nexus-text-secondary">Investigation Rooms, @Tagging & Escalation Pipeline</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-nexus-text-secondary" />
              <Input 
                placeholder="Search rooms..." 
                className="pl-9 w-64 bg-nexus-card border-nexus-border"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button className="bg-gradient-to-r from-blue-500 to-cyan-500">
              <Plus className="w-4 h-4 mr-2" />
              New Room
            </Button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card className="glass-card border-nexus-border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{rooms.filter((r:any) => r.status === 'active').length}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Active Rooms</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center text-blue-400">
                <MessageSquare className="w-6 h-6" />
              </div>
            </div>
        </Card>
        <Card className="glass-card border-nexus-border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{ESCALATIONS.length}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Open Escalations</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center text-red-400">
                <ArrowUpCircle className="w-6 h-6" />
              </div>
            </div>
        </Card>
        <Card className="glass-card border-nexus-border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">{TEAM_MEMBERS.filter(m => m.status === 'online').length}</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Team Online</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center text-green-400">
                <Users className="w-6 h-6" />
              </div>
            </div>
        </Card>
        <Card className="glass-card border-nexus-border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-3xl font-bold text-white">12</p>
                <p className="text-sm text-nexus-text-secondary mt-1">Unread Mentions</p>
              </div>
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400">
                <AtSign className="w-6 h-6" />
              </div>
            </div>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="bg-nexus-card border border-nexus-border">
          <TabsTrigger value="rooms" className="data-[state=active]:bg-blue-600">Investigation Rooms</TabsTrigger>
          <TabsTrigger value="escalations" className="data-[state=active]:bg-blue-600">Escalation Pipeline</TabsTrigger>
          <TabsTrigger value="team" className="data-[state=active]:bg-blue-600">Team Workspace</TabsTrigger>
        </TabsList>

        <TabsContent value="rooms" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-nexus-text-secondary uppercase tracking-wider">Active Investigations</h3>
              <div className="space-y-3 max-h-[calc(100vh-350px)] overflow-y-auto pr-2 custom-scrollbar">
                {roomsLoading ? (
                  <div className="flex justify-center py-8"><Loader2 className="w-6 h-6 animate-spin text-blue-500" /></div>
                ) : rooms.length > 0 ? rooms.map((room: any) => (
                  <Card 
                    key={room.room_id}
                    className={cn(
                      "glass-card border-nexus-border cursor-pointer transition-all hover:border-blue-500/50",
                      selectedRoomId === room.room_id && "border-blue-500 bg-blue-500/5"
                    )}
                    onClick={() => setSelectedRoomId(room.room_id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <Badge className={getPriorityColor(room.priority)}>{room.priority}</Badge>
                        <Badge className={getStatusColor(room.status)}>{room.status}</Badge>
                      </div>
                      <h4 className="font-medium text-white text-sm mb-1">{room.title}</h4>
                      <p className="text-xs text-nexus-text-secondary line-clamp-2">{room.description || 'No description provided.'}</p>
                      <div className="flex items-center justify-between text-xs mt-3">
                        <span className="text-nexus-text-secondary capitalize">{room.type}</span>
                        <span className="text-nexus-text-secondary">{new Date(room.created_at).toLocaleDateString()}</span>
                      </div>
                    </CardContent>
                  </Card>
                )) : (
                    <div className="p-8 text-center border-2 border-dashed border-nexus-border rounded-xl">
                        <p className="text-nexus-text-secondary text-sm">No investigations found.</p>
                    </div>
                )}
              </div>
            </div>

            <div className="md:col-span-2">
              {selectedRoom ? (
                <Card className="glass-card border-nexus-border h-[calc(100vh-320px)] flex flex-col">
                  <div className="p-4 border-b border-nexus-border flex items-center justify-between shrink-0">
                    <div>
                        <CardTitle className="text-white text-lg">{selectedRoom.title}</CardTitle>
                        <p className="text-xs text-nexus-text-secondary">{selectedRoom.room_id} • {selectedRoom.type}</p>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setSelectedRoomId(null)}><X className="w-4 h-4" /></Button>
                  </div>
                  
                  <CardContent className="p-0 flex-1 flex flex-col overflow-hidden">
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                      {messagesLoading ? (
                        <div className="flex justify-center py-8"><Loader2 className="w-6 h-6 animate-spin text-blue-500" /></div>
                      ) : messages.map((m: any) => (
                        <div key={m.message_id} className={cn("flex gap-3", m.user_id === user?.user_id ? "flex-row-reverse" : "flex-row")}>
                          <Avatar className="w-8 h-8 shrink-0">
                            <AvatarFallback className="text-xs bg-gradient-to-br from-blue-500 to-purple-500 text-white">
                              {getInitials(m.sender || 'Unknown')}
                            </AvatarFallback>
                          </Avatar>
                          <div className={cn("max-w-[80%]", m.user_id === user?.user_id ? "text-right" : "text-left")}>
                            <p className="text-[10px] text-nexus-text-secondary mb-1">{m.sender}</p>
                            <div className={cn(
                              "p-3 rounded-xl text-sm inline-block text-left",
                              m.user_id === user?.user_id ? "bg-blue-600 text-white" : "bg-nexus-card-hover border border-nexus-border text-nexus-text"
                            )}>
                              {m.content}
                            </div>
                          </div>
                        </div>
                      ))}
                      <div ref={messageEndRef} />
                    </div>
                    
                    <div className="p-4 border-t border-nexus-border shrink-0">
                        <div className="flex gap-2">
                            <Button 
                              variant="ghost" 
                              size="icon" 
                              className="text-nexus-text-secondary hover:text-primary"
                              onClick={() => setMessageInput(prev => prev + '@')}
                            >
                              <AtSign className="w-4 h-4" />
                            </Button>
                            <Textarea 
                              placeholder="Type a message..."
                              value={messageInput}
                              onChange={(e) => setMessageInput(e.target.value)}
                              className="min-h-[60px] bg-nexus-card border-nexus-border"
                              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                            />
                            <Button onClick={handleSend} disabled={!messageInput.trim() || postMessageMutation.isPending} className="h-auto px-6 bg-blue-600">
                                {postMessageMutation.isPending ? <Loader2 className="animate-spin" /> : <Send className="w-4 h-4" />}
                            </Button>
                        </div>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="glass-card border-nexus-border h-full flex items-center justify-center">
                  <div className="text-center p-8 opacity-50">
                    <MessageSquare className="w-16 h-16 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-white mb-2">Select an Investigation Room</h3>
                    <p className="text-sm">Choose a room from the list to start collaborating</p>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="escalations" className="space-y-4">
           <Card className="glass-card border-nexus-border">
                <CardHeader>
                    <CardTitle className="text-white flex items-center gap-2">
                        <ArrowUpCircle className="w-5 h-5 text-red-400" />
                        Escalation Pipeline
                    </CardTitle>
                    <CardDescription>Track and manage escalated issues with SLA monitoring</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {ESCALATIONS.map((e) => (
                        <div key={e.id} className="p-4 bg-nexus-card rounded-lg border border-nexus-border flex justify-between items-center">
                            <div>
                                <div className="flex items-center gap-2 mb-1">
                                    <Badge className={e.severity === 'critical' ? 'bg-red-500/20 text-red-400' : 'bg-orange-500/20 text-orange-400'}>{e.severity}</Badge>
                                    <span className="text-xs text-nexus-text-secondary">{e.id}</span>
                                </div>
                                <h4 className="font-medium text-white">{e.title}</h4>
                            </div>
                            <div className="text-right">
                                <p className="text-lg font-bold text-red-400">{e.slaRemaining}</p>
                                <p className="text-[10px] text-nexus-text-secondary uppercase">SLA Remaining</p>
                            </div>
                        </div>
                    ))}
                </CardContent>
           </Card>
        </TabsContent>

        <TabsContent value="team" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-1">
                <Card className="glass-card border-nexus-border">
                    <CardHeader>
                        <CardTitle className="text-white">Team Members</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {TEAM_MEMBERS.map((m) => (
                            <div key={m.id} className="flex items-center gap-3 p-3 bg-nexus-card rounded-lg border border-nexus-border">
                                <Avatar><AvatarFallback className="bg-blue-500 text-white text-xs">{getInitials(m.name)}</AvatarFallback></Avatar>
                                <div className="flex-1">
                                    <p className="font-medium text-white text-sm">{m.name}</p>
                                    <p className="text-xs text-nexus-text-secondary">{m.role}</p>
                                </div>
                                <Badge variant="outline" className={cn(m.status === 'online' ? "border-green-500 text-green-400" : "border-nexus-border text-nexus-text-secondary")}>
                                    {m.status}
                                </Badge>
                            </div>
                        ))}
                    </CardContent>
                </Card>
            </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
