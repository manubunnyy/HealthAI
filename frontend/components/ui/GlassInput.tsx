import React from 'react';
import { cn } from '@/lib/utils';

interface GlassInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    className?: string;
}

export function GlassInput({ className, ...props }: GlassInputProps) {
    return (
        <input
            className={cn(
                "bg-glass-100 backdrop-blur-sm border border-glass-border rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all placeholder-gray-400 text-white w-full",
                className
            )}
            {...props}
        />
    );
}
