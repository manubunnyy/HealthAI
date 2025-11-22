import React from 'react';
import { Loader2 } from 'lucide-react';

interface ProgressBarProps {
    progress: number; // 0-100
    message?: string;
    className?: string;
}

export function ProgressBar({ progress, message, className = '' }: ProgressBarProps) {
    return (
        <div className={`w-full ${className}`}>
            <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-400 flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {message || 'Processing...'}
                </span>
                <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-glass-100 rounded-full h-2 overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                />
            </div>
        </div>
    );
}
