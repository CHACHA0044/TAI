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
  if (rowLength === 1) return "w-full md:w-[22rem]";
  if (rowLength === 2) return "w-full md:w-[20rem]";
  return "w-full md:w-[18rem]";
}

export function ScoreCardGrid({ metrics, onOpenMetric }: ScoreCardGridProps) {
  const rows = layoutRows(metrics);

  return (
    <div className="space-y-4">
      {rows.map((row, rowIndex) => (
        <div key={`${rowIndex}-${row.length}`} className="flex flex-wrap md:flex-nowrap justify-center gap-4">
          {row.map((metric) => (
            <div key={metric.key} className={getCardWidthClass(row.length, metrics.length)}>
              <MetricCard metric={metric} onOpen={onOpenMetric} />
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
