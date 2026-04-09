import React from "react";
import { cn } from "@/lib/utils";

interface GlassButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  className?: string;
  variant?: "primary" | "secondary" | "danger";
}

export function GlassButton({
  children,
  className,
  variant = "primary",
  ...props
}: GlassButtonProps) {
  const variants = {
    primary: "bg-blue-600/80 hover:bg-blue-600 text-white",
    secondary: "bg-glass-200 hover:bg-glass-300 text-white",
    danger: "bg-red-500/80 hover:bg-red-500 text-white",
  };

  return (
    <button
      className={cn(
        "backdrop-blur-md border border-glass-border rounded-xl px-6 py-3 transition-all duration-300 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-lg",
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
