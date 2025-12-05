import React, { useState, useRef, useEffect } from "react";
import { useStore } from "../store";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  MessageSquare,
  Send,
  Trash2,
  Loader2,
  AlertCircle,
  Bot,
  User,
  FileText,
  ChevronDown,
  ChevronUp,
  Sparkles,
  BookOpen,
} from "lucide-react";
import { ChatMessage, ChatSource, SessionStatus } from "../types";

// Markdown components styling
const markdownComponents = {
  p: ({ children }: any) => (
    <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
  ),
  h1: ({ children }: any) => (
    <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0 text-zinc-100">
      {children}
    </h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-base font-semibold mb-2 mt-3 first:mt-0 text-zinc-100">
      {children}
    </h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0 text-zinc-200">
      {children}
    </h3>
  ),
  ul: ({ children }: any) => (
    <ul className="list-disc list-inside mb-2 space-y-1 text-zinc-300">
      {children}
    </ul>
  ),
  ol: ({ children }: any) => (
    <ol className="list-decimal list-inside mb-2 space-y-1 text-zinc-300">
      {children}
    </ol>
  ),
  li: ({ children }: any) => <li className="text-sm">{children}</li>,
  code: ({ inline, children }: any) =>
    inline ? (
      <code className="bg-zinc-700/50 px-1.5 py-0.5 rounded text-xs font-mono text-violet-300">
        {children}
      </code>
    ) : (
      <pre className="bg-zinc-900/80 rounded-lg p-3 my-2 overflow-x-auto border border-zinc-700/50">
        <code className="text-xs font-mono text-zinc-300">{children}</code>
      </pre>
    ),
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-2 border-violet-500/50 pl-3 my-2 text-zinc-400 italic">
      {children}
    </blockquote>
  ),
  a: ({ href, children }: any) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-violet-400 hover:text-violet-300 underline underline-offset-2"
    >
      {children}
    </a>
  ),
  table: ({ children }: any) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border border-zinc-700/50 rounded-lg overflow-hidden">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }: any) => (
    <thead className="bg-zinc-800/50">{children}</thead>
  ),
  th: ({ children }: any) => (
    <th className="px-3 py-2 text-left font-medium text-zinc-300 border-b border-zinc-700/50">
      {children}
    </th>
  ),
  td: ({ children }: any) => (
    <td className="px-3 py-2 text-zinc-400 border-b border-zinc-700/30">
      {children}
    </td>
  ),
  strong: ({ children }: any) => (
    <strong className="font-semibold text-zinc-100">{children}</strong>
  ),
  em: ({ children }: any) => (
    <em className="italic text-zinc-300">{children}</em>
  ),
  hr: () => <hr className="my-3 border-zinc-700/50" />,
};

// Source citation component
function SourceCitation({
  sources,
  isExpanded,
  onToggle,
}: {
  sources: ChatSource[];
  isExpanded: boolean;
  onToggle: () => void;
}) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 border-t border-zinc-700/50 pt-3">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-300 transition-colors"
      >
        <BookOpen size={14} />
        <span>{sources.length} sources</span>
        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-2">
              {sources.map((source, idx) => (
                <div
                  key={idx}
                  className="bg-zinc-800/50 rounded-lg p-3 text-xs border border-zinc-700/30"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-zinc-400 flex items-center gap-1">
                      <FileText size={12} />
                      {source.doc_id}
                    </span>
                    <span className="text-emerald-400 font-mono">
                      {(source.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className="text-zinc-300 leading-relaxed">{source.text}</p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Single message bubble
function MessageBubble({ message }: { message: ChatMessage }) {
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-600/20 flex items-center justify-center flex-shrink-0 border border-violet-500/30">
          <Bot size={16} className="text-violet-400" />
        </div>
      )}

      <div
        className={`max-w-[80%] ${
          isUser
            ? "bg-primary/20 border-primary/30"
            : "bg-zinc-800/50 border-zinc-700/30"
        } rounded-2xl px-4 py-3 border`}
      >
        {message.isLoading ? (
          <div className="flex items-center gap-2 text-zinc-400">
            <Loader2 size={16} className="animate-spin" />
            <span className="text-sm">Thinking...</span>
          </div>
        ) : (
          <>
            {isUser ? (
              <p className="text-sm text-zinc-100 whitespace-pre-wrap leading-relaxed">
                {message.content}
              </p>
            ) : (
              <div className="text-sm text-zinc-100 prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
            {message.sources && message.sources.length > 0 && (
              <SourceCitation
                sources={message.sources}
                isExpanded={sourcesExpanded}
                onToggle={() => setSourcesExpanded(!sourcesExpanded)}
              />
            )}
          </>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-blue-500/20 flex items-center justify-center flex-shrink-0 border border-primary/30">
          <User size={16} className="text-primary" />
        </div>
      )}
    </motion.div>
  );
}

// Empty state
function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/10 to-purple-600/10 flex items-center justify-center mb-6 border border-violet-500/20">
        <Sparkles size={32} className="text-violet-400" />
      </div>
      <h3 className="text-xl font-semibold text-zinc-100 mb-2">
        Ask questions about your document
      </h3>
      <p className="text-zinc-400 max-w-md text-sm leading-relaxed">
        Use RAG (Retrieval Augmented Generation) to get intelligent answers
        based on the processed document content.
      </p>
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg">
        {[
          "What are the main findings?",
          "Summarize the key points",
          "What methodology was used?",
          "What are the conclusions?",
        ].map((suggestion) => (
          <button
            key={suggestion}
            className="px-4 py-2 text-xs bg-zinc-800/50 border border-zinc-700/50 rounded-lg text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 transition-all text-left"
          >
            "{suggestion}"
          </button>
        ))}
      </div>
    </div>
  );
}

// Error banner
function ErrorBanner({
  error,
  onDismiss,
}: {
  error: string;
  onDismiss: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="mx-4 mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center justify-between"
    >
      <div className="flex items-center gap-2 text-red-400">
        <AlertCircle size={16} />
        <span className="text-sm">{error}</span>
      </div>
      <button
        onClick={onDismiss}
        className="text-red-400 hover:text-red-300 text-xs underline"
      >
        Dismiss
      </button>
    </motion.div>
  );
}

// Main RAG Chat component
export function RagChat() {
  const { chat, sendChatMessage, clearChat, pipeline, session } = useStore();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat.messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    if (!input.trim() || chat.isLoading) return;

    const question = input.trim();
    setInput("");
    await sendChatMessage(question);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const dismissError = () => {
    useStore.setState((state) => ({
      chat: { ...state.chat, error: null },
    }));
  };

  const isReady =
    pipeline.status === "completed" ||
    session.status === SessionStatus.COMPLETED;

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-zinc-900 to-background">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-purple-600/20 flex items-center justify-center border border-violet-500/30">
            <MessageSquare size={20} className="text-violet-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">
              Document Chat
            </h2>
            <p className="text-xs text-zinc-500">
              {isReady
                ? "Ask questions about the processed document"
                : "Complete document processing to enable chat"}
            </p>
          </div>
        </div>

        {chat.messages.length > 0 && (
          <button
            onClick={clearChat}
            className="flex items-center gap-2 px-3 py-1.5 text-xs text-zinc-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors border border-transparent hover:border-red-500/30"
          >
            <Trash2 size={14} />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Error Banner */}
      <AnimatePresence>
        {chat.error && (
          <ErrorBanner error={chat.error} onDismiss={dismissError} />
        )}
      </AnimatePresence>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {chat.messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="space-y-4 max-w-3xl mx-auto">
            {chat.messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-zinc-800/50 px-4 py-4 bg-zinc-900/50 backdrop-blur">
        <div className="max-w-3xl mx-auto">
          {!isReady ? (
            <div className="flex items-center justify-center gap-2 py-3 text-zinc-500 text-sm">
              <AlertCircle size={16} />
              <span>Document processing must be completed to use chat</span>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about the document..."
                  disabled={chat.isLoading}
                  className="w-full px-4 py-3 pr-12 bg-zinc-800/50 border border-zinc-700/50 rounded-xl text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30 transition-all disabled:opacity-50"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || chat.isLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-lg bg-primary/20 text-primary hover:bg-primary/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {chat.isLoading ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <Send size={16} />
                  )}
                </button>
              </div>
            </div>
          )}

          <p className="text-center text-xs text-zinc-600 mt-3">
            Responses are generated using RAG from the processed document
          </p>
        </div>
      </div>
    </div>
  );
}
