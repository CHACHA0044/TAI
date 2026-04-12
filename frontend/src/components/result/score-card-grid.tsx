"use client";

import { MetricCardData } from "./metric-types";
import { MetricCard } from "./metric-card";

interface ScoreCardGridProps {
  metrics: MetricCardData[];
  onOpenMetric: (metric: MetricCardData) => void;
}

function layoutRows(metrics: MetricCardData[]): MetricCardData[][] {
  if (metrics.length <= 3) return [metrics];
  if (metrics.length === 4) return [metrics.slice(0, 3), metrics.slice(3, 4)];
  if (metrics.length === 5) return [metrics.slice(0, 3), metrics.slice(3, 5)];

  const rows: MetricCardData[][] = [];
  for (let i = 0; i < metrics.length; i += 3) {
    rows.push(metrics.slice(i, i + 3));
  }
  return rows;
}

function getCardWidthClass(rowLength: number, totalMetrics: number) {
  if (totalMetrics <= 2) return "w-full md:w-[22rem]";
  if (rowLength === 1) return "w-full md:w-[21rem]";
  if (rowLength === 2) return "w-full md:w-[19.5rem]";
  return "w-full md:w-[18.75rem]";
}

export function ScoreCardGrid({ metrics, onOpenMetric }: ScoreCardGridProps) {
  const rows = layoutRows(metrics);

  return (
    <div className="space-y-6">
      {rows.map((row, rowIndex) => (
        <div
          key={`row-${rowIndex}`}
          className={`flex flex-wrap gap-6 ${row.length < 3 ? "justify-center" : "justify-start"}`}
        >
          {row.map((metric) => (
            <div key={metric.key} className={`flex ${getCardWidthClass(row.length, metrics.length)}`}>
              <MetricCard metric={metric} onOpen={onOpenMetric} />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
