"use client";

import React, { useState } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassInput } from '@/components/ui/GlassInput';
import { Activity, Heart, Scale, AlertCircle } from 'lucide-react';
import axios from 'axios';

export default function PredictionPage() {
    const [activeType, setActiveType] = useState<'diabetes' | 'heart' | 'obesity'>('diabetes');
    const [formData, setFormData] = useState({
        weight: '',
        height: '',
        glucose_fasting: '',
        glucose_post_meal: '',
        heart_rate: '',
        bp_systolic: '',
        bp_diastolic: ''
    });
    const [result, setResult] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleAnalyze = async () => {
        setIsLoading(true);
        try {
            // Convert string inputs to numbers
            const payload: any = {
                prediction_type: activeType,
                weight: parseFloat(formData.weight),
                height: parseFloat(formData.height)
            };

            if (activeType === 'diabetes') {
                payload.glucose_fasting = parseFloat(formData.glucose_fasting);
                payload.glucose_post_meal = parseFloat(formData.glucose_post_meal);
            } else if (activeType === 'heart') {
                payload.heart_rate = parseFloat(formData.heart_rate);
                payload.bp_systolic = parseFloat(formData.bp_systolic);
                payload.bp_diastolic = parseFloat(formData.bp_diastolic);
            }

            const response = await axios.post('http://localhost:8000/prediction/analyze', payload);
            setResult(response.data);
        } catch (error) {
            console.error('Error analyzing health:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto">
            <header className="mb-8 text-center">
                <h1 className="text-3xl font-bold text-white mb-2">Health Predictor</h1>
                <p className="text-gray-400">Assess your health risks with AI-powered analysis</p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <button
                    onClick={() => { setActiveType('diabetes'); setResult(null); }}
                    className={`p-4 rounded-xl border transition-all flex flex-col items-center gap-2 ${activeType === 'diabetes' ? 'bg-blue-600/20 border-blue-500 text-white' : 'bg-glass-100 border-glass-border text-gray-400 hover:bg-glass-200'}`}
                >
                    <Activity size={24} />
                    <span>Diabetes Risk</span>
                </button>
                <button
                    onClick={() => { setActiveType('heart'); setResult(null); }}
                    className={`p-4 rounded-xl border transition-all flex flex-col items-center gap-2 ${activeType === 'heart' ? 'bg-red-600/20 border-red-500 text-white' : 'bg-glass-100 border-glass-border text-gray-400 hover:bg-glass-200'}`}
                >
                    <Heart size={24} />
                    <span>Heart Health</span>
                </button>
                <button
                    onClick={() => { setActiveType('obesity'); setResult(null); }}
                    className={`p-4 rounded-xl border transition-all flex flex-col items-center gap-2 ${activeType === 'obesity' ? 'bg-green-600/20 border-green-500 text-white' : 'bg-glass-100 border-glass-border text-gray-400 hover:bg-glass-200'}`}
                >
                    <Scale size={24} />
                    <span>Weight Status</span>
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <GlassCard>
                    <h3 className="text-xl font-semibold mb-6 text-white">Enter Your Metrics</h3>
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Weight (kg)</label>
                                <GlassInput name="weight" type="number" value={formData.weight} onChange={handleInputChange} placeholder="70" />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Height (m)</label>
                                <GlassInput name="height" type="number" value={formData.height} onChange={handleInputChange} placeholder="1.75" />
                            </div>
                        </div>

                        {activeType === 'diabetes' && (
                            <>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">Fasting Glucose (mg/dL)</label>
                                    <GlassInput name="glucose_fasting" type="number" value={formData.glucose_fasting} onChange={handleInputChange} placeholder="95" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">Post-Meal Glucose (mg/dL)</label>
                                    <GlassInput name="glucose_post_meal" type="number" value={formData.glucose_post_meal} onChange={handleInputChange} placeholder="140" />
                                </div>
                            </>
                        )}

                        {activeType === 'heart' && (
                            <>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">Heart Rate (bpm)</label>
                                    <GlassInput name="heart_rate" type="number" value={formData.heart_rate} onChange={handleInputChange} placeholder="72" />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm text-gray-400 mb-1">Systolic BP</label>
                                        <GlassInput name="bp_systolic" type="number" value={formData.bp_systolic} onChange={handleInputChange} placeholder="120" />
                                    </div>
                                    <div>
                                        <label className="block text-sm text-gray-400 mb-1">Diastolic BP</label>
                                        <GlassInput name="bp_diastolic" type="number" value={formData.bp_diastolic} onChange={handleInputChange} placeholder="80" />
                                    </div>
                                </div>
                            </>
                        )}

                        <GlassButton onClick={handleAnalyze} disabled={isLoading} className="w-full mt-4">
                            {isLoading ? 'Analyzing...' : 'Analyze Health Risks'}
                        </GlassButton>
                    </div>
                </GlassCard>

                <GlassCard>
                    <h3 className="text-xl font-semibold mb-6 text-white">Analysis Results</h3>
                    {result ? (
                        <div className="space-y-6">
                            <div className="text-center p-4 bg-glass-100 rounded-xl">
                                <div className="text-sm text-gray-400 mb-1">BMI Score</div>
                                <div className="text-4xl font-bold text-white mb-1">{result.bmi}</div>
                                <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${result.bmi_category === 'normal' ? 'bg-green-500/20 text-green-300' :
                                        result.bmi_category === 'overweight' ? 'bg-yellow-500/20 text-yellow-300' :
                                            'bg-red-500/20 text-red-300'
                                    }`}>
                                    {result.bmi_category.toUpperCase()}
                                </div>
                            </div>

                            <div>
                                <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                                    <AlertCircle size={16} /> Key Insights
                                </h4>
                                <ul className="space-y-2">
                                    {result.insights.map((insight: string, idx: number) => (
                                        <li key={idx} className="text-sm text-gray-300 bg-glass-100 p-3 rounded-lg border-l-2 border-blue-500">
                                            {insight}
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <div>
                                <h4 className="text-sm font-semibold text-gray-300 mb-3">Recommendations</h4>
                                <ul className="space-y-2">
                                    {result.recommendations.map((rec: string, idx: number) => (
                                        <li key={idx} className="text-sm text-gray-400 flex items-start gap-2">
                                            <span className="text-green-400 mt-1">•</span>
                                            {rec}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                            <Activity className="w-16 h-16 mb-4 opacity-20" />
                            <p>Enter your metrics to see analysis</p>
                        </div>
                    )}
                </GlassCard>
            </div>
        </div>
    );
}
