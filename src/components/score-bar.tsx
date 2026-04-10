"use client";

import { motion } from "framer-motion";

interface ScoreBarProps {
  label: string;
  score: number; // 0 to 1
  color?: string;
  showPercentage?: boolean;
}

export function ScoreBar({ label, score, color, showPercentage = true }: ScoreBarProps) {
  // Determine color based on score if not provided
  const getBarColor = () => {
    if (color) return color;
    if (score > 0.7) return "bg-emerald-500";
    if (score > 0.4) return "bg-amber-500";
    return "bg-rose-500";
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-end">
        <span className="text-xs font-semibold text-white/60 uppercase tracking-wider">{label}</span>
        {showPercentage && (
          <span className={`text-sm font-bold ${getBarColor().replace('bg-', 'text-')}`}>
            {Math.round(score * 100)}%
          </span>
        )}
      </div>
      <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${score * 100}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full rounded-full ${getBarColor()} shadow-[0_0_10px_rgba(0,0,0,0.5)]`}
        />
      </div>
    </div>
  );
}
