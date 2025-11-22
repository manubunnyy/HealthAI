"use client";

import React from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { Scan, Sparkles, Zap, Shield, Loader2 } from 'lucide-react';

export default function AnomalyDetectionPage() {
    return (
        <div className="max-w-4xl mx-auto">
            <header className="mb-8 text-center">
                <div className="flex items-center justify-center gap-3 mb-4">
                    <Scan className="w-12 h-12 text-teal-400" />
                    <h1 className="text-4xl font-bold text-white">Anomaly Detection</h1>
                </div>
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-teal-500/20 text-teal-300 rounded-full border border-teal-500/30 mb-4">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="font-semibold">Coming Soon</span>
                </div>
                <p className="text-gray-400 max-w-2xl mx-auto">
                    Advanced AI-powered anomaly detection for medical data analysis
                </p>
            </header>

            <GlassCard className="mb-8">
                <div className="text-center py-12">
                    <div className="relative inline-block mb-6">
                        <div className="absolute inset-0 bg-teal-500/20 blur-3xl rounded-full animate-pulse" />
                        <Scan className="relative w-24 h-24 text-teal-400 mx-auto" />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-4">Under Development</h2>
                    <p className="text-gray-300 mb-8 max-w-xl mx-auto">
                        We're building an advanced anomaly detection system that will help identify unusual patterns in your health data using cutting-edge AI technology.
                    </p>
                </div>
            </GlassCard>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <GlassCard className="text-center p-6">
                    <Sparkles className="w-10 h-10 text-yellow-400 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-white mb-2">Smart Detection</h3>
                    <p className="text-sm text-gray-400">
                        AI algorithms to detect anomalies in medical reports and test results
                    </p>
                </GlassCard>

                <GlassCard className="text-center p-6">
                    <Zap className="w-10 h-10 text-blue-400 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-white mb-2">Real-time Analysis</h3>
                    <p className="text-sm text-gray-400">
                        Instant detection and alerts for potential health concerns
                    </p>
                </GlassCard>

                <GlassCard className="text-center p-6">
                    <Shield className="w-10 h-10 text-green-400 mx-auto mb-3" />
                    <h3 className="text-lg font-semibold text-white mb-2">Privacy First</h3>
                    <p className="text-sm text-gray-400">
                        All analysis done with complete data privacy and security
                    </p>
                </GlassCard>
            </div>

            <GlassCard className="bg-gradient-to-br from-teal-500/10 to-cyan-500/10 border-teal-500/30">
                <div className="text-center p-8">
                    <h3 className="text-xl font-bold text-white mb-3">Stay Tuned!</h3>
                    <p className="text-gray-300 mb-4">
                        This feature is currently in development and will be available soon.
                    </p>
                    <div className="flex items-center justify-center gap-2 text-teal-300">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span className="font-medium">Building something amazing...</span>
                    </div>
                </div>
            </GlassCard>
        </div>
    );
}
