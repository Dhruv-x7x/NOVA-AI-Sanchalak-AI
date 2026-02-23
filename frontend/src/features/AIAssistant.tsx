import { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { intelligenceApi } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Bot,
  Send,
  User,
  Sparkles,
  Loader2,
  Info,
  ChevronRight,
  ChevronUp,
  Activity,
  Eraser,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  agent_chain?: string[];
  steps?: Array<{ agent: string; thought: string; action: string; observation?: string }>;
  tools_used?: string[];
  confidence?: number;
  recommendations?: Array<{ action: string; impact: string }>;
}

export default function AIAssistant() {
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('nexus_ai_chat');
    return saved ? JSON.parse(saved) : [
      {
        content: 'Hello! I\'m your a6on-i Assistant. I use a swarm of 6 specialized agents to analyze your clinical trial data in real-time.\n\nHow can I help you today?',
      },
    ];
  });
  const [input, setInput] = useState('');
  const [expandedTraces, setExpandedTraces] = useState<Record<number, boolean>>({});
  const scrollRef = useRef<HTMLDivElement>(null);

  const toggleTrace = (idx: number) => {
    setExpandedTraces(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  useEffect(() => {
    localStorage.setItem('nexus_ai_chat', JSON.stringify(messages));
  }, [messages]);

  const assistantMutation = useMutation({
    mutationFn: (query: string) => {
      // Send last 6 turns of conversation history for multi-turn context
      const history = messages
        .filter((m): m is Message & { role: 'user' | 'assistant' } => m.role === 'user' || m.role === 'assistant')
        .slice(-6)
        .map(m => ({ role: m.role, content: m.content }));
      return intelligenceApi.runAssistant(query, history);
    },
    onSuccess: (data) => {
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.summary,
        agent_chain: data.agent_chain,
        steps: data.steps,
        tools_used: data.tools_used,
        confidence: data.confidence,
        recommendations: data.recommendations
      };
      setMessages((prev) => [...prev, assistantMessage]);
    },
    onError: (error: any) => {
      setMessages((prev) => [...prev, {
        role: 'assistant',
        content: ` bottleneck encountered: ${error.message || 'Orchestrator timeout'}`
      }]);
    }
  });

  const handleSend = async () => {
    if (!input.trim() || assistantMutation.isPending) return;
    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    const query = input;
    setInput('');
    assistantMutation.mutate(query);  // history is included via mutationFn closure
  };

  const handleClear = () => {
    setMessages([{ role: 'assistant', content: 'Conversation cleared. Ready for new analysis.' }]);
    localStorage.removeItem('nexus_ai_chat');
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, assistantMutation.isPending]);

  const suggestedQuestions = [
    'Analyze current DQI trends',
    'Which sites have high risk patients?',
    'Forecast database lock readiness',
    'Summarize safety events from this week',
  ];

  const lastAssistantMessage = [...messages].reverse().find(m => m.role === 'assistant');

  return (
    <div className="h-[calc(100vh-140px)] flex flex-col space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
            <Bot className="w-5 h-5 text-indigo-400" />
          </div>
          a6on-i Assistant
        </h1>
        <Button variant="outline" size="sm" onClick={handleClear} className="border-nexus-border hover:bg-error-500/10 hover:text-error-400 h-9">
          <Eraser className="w-4 h-4 mr-2" /> New Conversation
        </Button>
      </div>

      <div className="flex-1 grid gap-6 md:grid-cols-4 overflow-hidden">
        {/* Chat Area */}
        <Card className="md:col-span-3 bg-nexus-card border-nexus-border flex flex-col overflow-hidden shadow-2xl">
          <CardContent className="p-0 flex-1 flex flex-col overflow-hidden">
            <ScrollArea className="flex-1">
              <div className="p-6 space-y-6">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={cn(
                      "flex gap-3 w-full",
                      message.role === 'user' ? "flex-row-reverse" : "flex-row"
                    )}
                  >
                    <div
                      className={cn(
                        "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-md self-end mb-1",
                        message.role === 'user'
                          ? "bg-indigo-500 text-white order-2"
                          : "bg-nexus-card-hover border border-nexus-border text-indigo-400"
                      )}
                    >
                      {message.role === 'user' ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <Bot className="w-4 h-4" />
                      )}
                    </div>
                    <div
                      className={cn(
                        "max-w-[80%] p-4 rounded-2xl shadow-sm transition-all duration-200",
                        message.role === 'user'
                          ? "bg-indigo-600/90 text-white rounded-br-none border border-white/5 hover:bg-indigo-600"
                          : "bg-nexus-card-hover border border-nexus-border text-nexus-text rounded-bl-none hover:border-indigo-500/30"
                      )}
                    >
                      <div className="text-[14px] leading-relaxed font-normal prose prose-invert prose-sm max-w-none prose-headings:text-indigo-300 prose-headings:text-sm prose-headings:font-bold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-table:text-xs prose-th:bg-white/10 prose-th:px-3 prose-th:py-1.5 prose-th:text-indigo-300 prose-th:font-semibold prose-td:px-3 prose-td:py-1.5 prose-td:border-white/10 prose-li:my-0.5 prose-strong:text-white prose-ul:my-1">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                      </div>

                      {/* Technical Details Toggle */}
                      {message.role === 'assistant' && (message.agent_chain || message.steps) && (
                        <div className="mt-3 pt-3 border-t border-nexus-border/20">
                          <button
                            onClick={() => toggleTrace(index)}
                            className="flex items-center gap-2 text-[9px] font-bold text-indigo-400/80 uppercase tracking-wider hover:text-indigo-300 transition-colors"
                          >
                            <Activity className="w-2.5 h-2.5" />
                            {expandedTraces[index] ? 'Hide Trace' : 'Clinical Intelligence Trace'}
                            {expandedTraces[index] ? <ChevronUp className="w-2.5 h-2.5" /> : <ChevronRight className="w-2.5 h-2.5" />}
                          </button>

                          {expandedTraces[index] && (
                            <div className="mt-4 space-y-4 animate-in fade-in slide-in-from-top-1 duration-200 bg-black/40 p-4 rounded-xl border border-white/5 shadow-inner">
                              {message.steps?.map((step: any, i: number) => (
                                <div key={i} className="relative pl-4 border-l border-indigo-500/30 pb-4 last:pb-0">
                                  <div className="absolute -left-[5px] top-0 w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.8)]" />
                                  <p className="text-[9px] font-black text-indigo-300 uppercase tracking-tighter mb-1">{step.agent}</p>
                                  <p className="text-[11px] text-white/80 italic leading-snug font-medium">"{step.thought}"</p>
                                  {step.observation && (
                                    <p className="text-[10px] text-emerald-400 mt-1 font-bold">↳ {step.observation}</p>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {assistantMutation.isPending && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-nexus-card-hover border border-nexus-border flex items-center justify-center text-indigo-400 self-end mb-1">
                      <Bot className="w-4 h-4 animate-spin duration-1000" />
                    </div>
                    <div className="bg-nexus-card-hover border border-nexus-border p-4 rounded-2xl rounded-bl-none shadow-sm flex flex-col gap-3 min-w-[180px]">
                      <div className="flex items-center gap-2">
                        <div className="flex gap-1">
                          <div className="w-1 h-1 bg-indigo-400 rounded-full animate-bounce delay-0"></div>
                          <div className="w-1 h-1 bg-indigo-400 rounded-full animate-bounce delay-150"></div>
                          <div className="w-1 h-1 bg-indigo-400 rounded-full animate-bounce delay-300"></div>
                        </div>
                        <p className="text-nexus-text-secondary text-[10px] font-bold uppercase tracking-wider">Swarm Analysis in progress</p>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={scrollRef} />
              </div>
            </ScrollArea>

            <div className="p-4 bg-nexus-card border-t border-nexus-border/50">
              <div className="flex gap-2 bg-nexus-bg/50 border border-nexus-border p-1.5 rounded-full focus-within:border-indigo-500/40 focus-within:ring-1 focus-within:ring-indigo-500/20 transition-all">
                <Input
                  placeholder="Ask a6on-i Assistant..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  className="flex-1 bg-transparent border-0 focus-visible:ring-0 text-[14px] text-white py-2 px-4 h-10"
                />
                <Button
                  onClick={handleSend}
                  disabled={assistantMutation.isPending || !input.trim()}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-full w-10 h-10 p-0 flex-shrink-0"
                >
                  {assistantMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Sidebar */}
        <div className="hidden md:flex flex-col gap-4 overflow-y-auto custom-scrollbar">
          <Card className="bg-nexus-card border-nexus-border shadow-lg">
            <CardHeader className="pb-3 border-b border-white/5">
              <CardTitle className="text-[11px] font-black text-white flex items-center gap-2 uppercase tracking-widest">
                <Sparkles className="w-4 h-4 text-yellow-400" />
                Quick Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-4">
              <div className="space-y-2">
                {suggestedQuestions.map((question, index) => (
                  <button
                    key={index}
                    className="w-full text-left p-3 rounded-xl border border-nexus-border bg-white/5 hover:bg-indigo-500/10 hover:border-indigo-500/30 text-nexus-text-secondary text-xs font-semibold transition-all group"
                    onClick={() => setInput(question)}
                  >
                    <div className="flex items-center gap-2">
                      <Activity className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity text-indigo-400" />
                      {question}
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {lastAssistantMessage?.recommendations && lastAssistantMessage.recommendations.length > 0 && (
            <Card className="bg-nexus-card border-nexus-border shadow-lg animate-in slide-in-from-right-4">
              <CardHeader className="pb-3 border-b border-white/5">
                <CardTitle className="text-[11px] font-black text-amber-400 flex items-center gap-2 uppercase tracking-widest">
                  <Zap className="w-4 h-4" />
                  Strategic Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-3">
                {lastAssistantMessage.recommendations.map((rec, i) => (
                  <div key={i} className="p-3 bg-white/5 rounded-lg border border-white/5 space-y-2">
                    <p className="text-xs font-bold text-white/90 leading-snug">{rec.action}</p>
                    <Badge className={cn(
                      "text-[9px] font-bold uppercase",
                      rec.impact === 'High' ? "bg-red-500/20 text-red-400 border-red-500/30" : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                    )}>
                      {rec.impact} Impact
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          <Card className="bg-nexus-card border-nexus-border shadow-lg">
            <CardHeader className="pb-3 border-b border-white/5">
              <CardTitle className="text-[11px] font-black text-white flex items-center gap-2 uppercase tracking-widest">
                <Info className="w-4 h-4 text-indigo-400" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-5 space-y-4">
              {[
                { label: 'Agent Supervisor', status: 'Active' },
                { label: 'Diagnostic Engine', status: 'Active' },
                { label: 'Forecasting Node', status: 'Active' },
                { label: 'Resolution Genome', status: 'Ready' },
              ].map((cap, idx) => (
                <div key={idx} className="flex items-center justify-between text-[11px] font-bold">
                  <span className="text-nexus-text-secondary">{cap.label}</span>
                  <span className="text-success-400">{cap.status}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
