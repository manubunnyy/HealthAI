"use client";

import React, { useState, useRef, useEffect } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassInput } from '@/components/ui/GlassInput';
import { Send, Upload, Bot, User, FileText } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    type?: 'text' | 'diet_plan';
    data?: any;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [files, setFiles] = useState<FileList | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await axios.post('http://localhost:8000/chat/query', { query: userMessage.content });

            const assistantMessage: Message = {
                role: 'assistant',
                content: response.data.synthesis,
                type: response.data.diet_plan ? 'diet_plan' : 'text',
                data: response.data.diet_plan
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error. Please try again." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFiles(e.target.files);
            const formData = new FormData();
            Array.from(e.target.files).forEach(file => {
                formData.append('files', file);
            });

            try {
                await axios.post('http://localhost:8000/chat/upload', formData);
                setMessages(prev => [...prev, { role: 'assistant', content: `Successfully processed ${e.target.files?.length} document(s). You can now ask questions about them.` }]);
            } catch (error) {
                console.error('Error uploading files:', error);
                setMessages(prev => [...prev, { role: 'assistant', content: "Failed to upload documents." }]);
            }
        }
    };

    return (
        <div className="max-w-4xl mx-auto h-[calc(100vh-8rem)] flex flex-col">
            <header className="mb-6">
                <h1 className="text-3xl font-bold text-white mb-2">Health Assistant</h1>
                <p className="text-gray-400">Chat with AI about your health concerns and documents</p>
            </header>

            <GlassCard className="flex-1 flex flex-col overflow-hidden mb-4">
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.length === 0 && (
                        <div className="text-center text-gray-500 mt-20">
                            <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
                            <p>Start a conversation or upload a health document.</p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] rounded-2xl p-4 ${msg.role === 'user'
                                ? 'bg-blue-600/80 text-white rounded-tr-none'
                                : 'bg-glass-200 text-gray-100 rounded-tl-none'
                                }`}>
                                <div className="flex items-center gap-2 mb-1 opacity-50 text-xs">
                                    {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                                    <span>{msg.role === 'user' ? 'You' : 'AI Assistant'}</span>
                                </div>
                                <div className="prose prose-invert max-w-none prose-sm">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                </div>

                                {msg.type === 'diet_plan' && msg.data && (
                                    <div className="mt-4 bg-glass-100 rounded-xl p-4 border border-yellow-500/30">
                                        <h3 className="text-yellow-400 font-bold mb-2">🍽️ {msg.data.title}</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            <div>
                                                <h4 className="text-green-400 text-sm font-semibold mb-1">Vegetarian</h4>
                                                <ul className="text-sm space-y-1">
                                                    <li>🍳 {msg.data.diet_plan.breakfast}</li>
                                                    <li>🥗 {msg.data.diet_plan.lunch}</li>
                                                    <li>🍲 {msg.data.diet_plan.dinner}</li>
                                                </ul>
                                            </div>
                                        </div>
                                        <p className="mt-3 text-xs text-gray-400 italic">Note: {msg.data.diet_plan.notes}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                <div className="p-4 border-t border-glass-border bg-glass-100">
                    <div className="flex gap-2">
                        <label className="cursor-pointer">
                            <input type="file" multiple className="hidden" onChange={handleFileUpload} accept=".pdf,.txt,.png,.jpg" />
                            <div className="p-3 bg-glass-200 hover:bg-glass-300 rounded-xl transition-colors text-gray-300">
                                <Upload size={20} />
                            </div>
                        </label>
                        <GlassInput
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Type your health question..."
                            className="flex-1"
                        />
                        <GlassButton onClick={handleSend} disabled={isLoading || !input.trim()}>
                            <Send size={20} />
                        </GlassButton>
                    </div>
                </div>
            </GlassCard>
        </div>
    );
}
