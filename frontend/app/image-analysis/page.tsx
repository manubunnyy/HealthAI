"use client";

import React, { useState } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Upload, ImageIcon, Sparkles } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ImageAnalysisPage() {
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedImage(file);
            setPreviewUrl(URL.createObjectURL(file));
            setAnalysis(null);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedImage) return;
        setIsLoading(true);

        const formData = new FormData();
        formData.append('file', selectedImage);

        try {
            const response = await axios.post('https://healthai-b6y2.onrender.com/image/analyze', formData);
            setAnalysis(response.data.analysis);
        } catch (error) {
            console.error('Error analyzing image:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-6xl mx-auto h-[calc(100vh-8rem)]">
            <header className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-white mb-2">Medical Image Analysis</h1>
                <p className="text-gray-400">AI-powered diagnostics for medical imaging</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
                <GlassCard className="flex flex-col items-center justify-center">
                    {previewUrl ? (
                        <div className="relative w-full h-full max-h-[500px] rounded-xl overflow-hidden mb-4 bg-black/50 flex items-center justify-center">
                            <img src={previewUrl} alt="Medical scan" className="max-w-full max-h-full object-contain" />
                        </div>
                    ) : (
                        <div className="text-center p-8 border-2 border-dashed border-glass-border rounded-xl w-full h-full flex flex-col items-center justify-center text-gray-400">
                            <ImageIcon className="w-16 h-16 mb-4 opacity-50" />
                            <p>Upload X-ray, MRI, CT, or Ultrasound</p>
                        </div>
                    )}

                    <div className="flex gap-4 w-full mt-auto pt-4">
                        <label className="flex-1">
                            <input type="file" accept="image/*" className="hidden" onChange={handleImageSelect} />
                            <div className="w-full py-3 bg-glass-200 hover:bg-glass-300 rounded-xl text-center cursor-pointer transition-colors text-white">
                                Select Image
                            </div>
                        </label>
                        <GlassButton onClick={handleAnalyze} disabled={!selectedImage || isLoading} className="flex-1">
                            {isLoading ? 'Analyzing...' : 'Analyze Scan'}
                        </GlassButton>
                    </div>
                </GlassCard>

                <GlassCard className="overflow-hidden flex flex-col">
                    <h3 className="text-xl font-semibold mb-4 text-white flex items-center gap-2 shrink-0">
                        <Sparkles className="text-purple-400" size={20} />
                        Diagnostic Report
                    </h3>
                    <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                        {analysis ? (
                            <div className="prose prose-invert max-w-none">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis}</ReactMarkdown>
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-500">
                                Analysis results will appear here
                            </div>
                        )}
                    </div>
                </GlassCard>
            </div>
        </div>
    );
}
