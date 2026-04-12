"use client";

import { ScoreBar } from "@/components/score-bar";
import { MetricCardData } from "./metric-types";

interface MetricCardProps {
  metric: MetricCardData;
  onOpen: (metric: MetricCardData) => void;
}

export function MetricCard({ metric, onOpen }: MetricCardProps) {
  const getColor = () => {
    const value = metric.invertColor ? 1 - metric.score : metric.score;
    if (value >= 0.68) return "bg-emerald-500";
    if (value >= 0.42) return "bg-amber-500";
    return "bg-rose-500";
  };

  const getTextColor = () => {
    const value = metric.invertColor ? 1 - metric.score : metric.score;
    if (value >= 0.68) return "text-emerald-400";
    if (value >= 0.42) return "text-amber-400";
    return "text-rose-400";
  };

  return (
    <button
      type="button"
      onClick={() => onOpen(metric)}
      className="glass h-full min-h-[208px] w-full text-left rounded-2xl border border-white/10 p-5 group transition-all hover:border-white/25 hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
      aria-label={`Open ${metric.label} explanation`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="p-2 rounded-lg bg-white/5">{metric.icon}</div>
        <span className={`text-2xl font-black font-mono ${getTextColor()}`}>{Math.round(metric.score * 100)}%</span>
      </div>
      <p className="text-xs font-bold text-white/85 mb-1">{metric.label}</p>
      <p className="text-[10px] text-white/40 leading-tight min-h-[30px]">{metric.description}</p>
      <div className="mt-4">
        <ScoreBar label="" score={metric.score} showPercentage={false} color={getColor()} />
      </div>
      <p className="text-[10px] text-white/35 mt-3">Tap for detailed interpretation</p>
    </button>
  );
}
