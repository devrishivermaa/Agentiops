import React, { useEffect, useRef } from 'react';
import { useStore } from '../store';
import { Terminal, Clock, AlertCircle, CheckCircle, Info } from 'lucide-react';

export const EventLog: React.FC = () => {
  const events = useStore((state) => state.events);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const getIcon = (severity: string) => {
    switch (severity) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default: return <Info className="w-4 h-4 text-blue-500" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface border-t border-surfaceHighlight md:border-t-0 md:border-l w-full md:w-80 shrink-0 shadow-xl">
      <div className="p-4 border-b border-surfaceHighlight flex items-center gap-2 bg-surfaceHighlight/50">
        <Terminal className="w-4 h-4 text-zinc-400" />
        <h3 className="text-sm font-semibold text-zinc-200">System Events</h3>
        <span className="ml-auto text-xs text-zinc-500">{events.length} events</span>
      </div>
      
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-2 space-y-1 font-mono text-xs"
      >
        {events.length === 0 ? (
          <div className="text-center text-zinc-600 mt-10 p-4">
            Waiting for pipeline events...
          </div>
        ) : (
          events.map((event) => (
            <div 
              key={event.id} 
              className={`p-2 rounded border border-surfaceHighlight/50 flex gap-2 animate-in fade-in slide-in-from-bottom-2 duration-300
                ${event.severity === 'error' ? 'bg-red-950/20 border-red-900/50' : 'bg-zinc-900/50'}
              `}
            >
              <div className="mt-0.5 shrink-0">
                {getIcon(event.severity)}
              </div>
              <div className="flex flex-col gap-1 min-w-0">
                <div className="flex items-center gap-2 text-zinc-500">
                  <Clock className="w-3 h-3" />
                  <span>{event.timestamp.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' })}</span>
                  {event.agentId && (
                     <span className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 text-[10px] truncate max-w-[80px]">
                       {event.agentId}
                     </span>
                  )}
                </div>
                <span className={`break-words leading-relaxed ${
                    event.severity === 'error' ? 'text-red-300' :
                    event.severity === 'success' ? 'text-green-300' : 'text-zinc-300'
                }`}>
                  {event.message}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};