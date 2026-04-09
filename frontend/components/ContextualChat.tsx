"use client";

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Bot, User } from "lucide-react";
import { GlassCard } from "@/components/ui/GlassCard";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ContextualChatProps {
  featureType: string;
  originalResult: string;
}

export function ContextualChat({ featureType, originalResult }: ContextualChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput("");
    
    const newMessages = [...messages, { role: "user" as const, content: userMsg }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Determine the API URL (use local if dev, otherwise use your deployed URL)
      const apiUrl = process.env.NODE_ENV === "development" 
        ? "http://localhost:8000/chat/followup" 
        : "https://healthai-b6y2.onrender.com/chat/followup";

      const response = await axios.post(apiUrl, {
        query: userMsg,
        feature_type: featureType,
        original_result: originalResult,
        history: messages.map(m => ({ role: m.role, content: m.content }))
      });

      setMessages([
        ...newMessages,
        { role: "assistant", content: response.data.response }
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages([
        ...newMessages,
        { role: "assistant", content: "Sorry, I encountered an error processing your request." }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-8">
      <GlassCard className="flex flex-col h-[400px]">
        <div className="flex items-center gap-2 mb-4 pb-4 border-b border-glass-border">
          <Bot className="text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Ask Follow-up Questions</h3>
        </div>

        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="text-gray-400 text-center flex flex-col items-center justify-center h-full">
              <Bot className="w-12 h-12 mb-2 opacity-50" />
              <p>I can help you customize or understand this {featureType} better.</p>
              <p className="text-sm mt-1">What would you like to know?</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === "user" ? "bg-blue-500/20 text-blue-400" : "bg-purple-500/20 text-purple-400"
                  }`}
                >
                  {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div
                  className={`px-4 py-3 rounded-2xl max-w-[85%] ${
                    msg.role === "user"
                      ? "bg-blue-600/20 text-blue-100 rounded-tr-sm"
                      : "bg-glass-100 text-gray-200 rounded-tl-sm prose prose-invert prose-sm"
                  }`}
                >
                  {msg.role === "user" ? (
                    msg.content
                  ) : (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center shrink-0">
                <Bot size={16} />
              </div>
              <div className="bg-glass-100 px-4 py-3 rounded-2xl rounded-tl-sm flex items-center gap-2">
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></span>
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-2 mt-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            placeholder="Type your question..."
            className="flex-1 bg-glass-100 border border-glass-border rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl px-4 py-3 transition-colors flex items-center justify-center"
          >
            <Send size={20} />
          </button>
        </div>
      </GlassCard>
    </div>
  );
}
