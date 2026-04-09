import React from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
}

export function GlassCard({ children, className, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "bg-glass-100 backdrop-blur-lg border border-glass-border rounded-2xl shadow-xl p-6",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
