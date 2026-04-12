"use client";

import { useEffect, useMemo, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X, ExternalLink } from "lucide-react";
import { MetricCardData } from "./metric-types";

interface MetricModalProps {
  metric: MetricCardData | null;
  isOpen: boolean;
  onClose: () => void;
}

export function MetricModal({ metric, isOpen, onClose }: MetricModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  const sourceLinks = useMemo(() => metric?.sources?.filter(Boolean) ?? [], [metric]);

  useEffect(() => {
    if (!isOpen || !dialogRef.current) return;

    const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    first?.focus();

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key === "Tab" && focusable.length > 0) {
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && metric && (
        <motion.div
          className="fixed inset-0 z-[120] flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          aria-hidden={!isOpen}
        >
          <div className="absolute inset-0 bg-black/65 backdrop-blur-sm" />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="metric-modal-title"
            ref={dialogRef}
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto glass rounded-3xl border border-white/20 p-6 sm:p-8 shadow-[0_20px_80px_rgba(0,0,0,0.6)]"
            initial={{ opacity: 0, scale: 0.96, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 12 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <p className="text-[10px] uppercase tracking-[0.2em] text-white/35 font-black">Metric deep-dive</p>
                <h3 id="metric-modal-title" className="text-2xl font-black text-white mt-1">
                  {metric.label}
                </h3>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="p-2 rounded-lg bg-white/5 border border-white/10 text-white/70 hover:text-white hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                aria-label="Close metric dialog"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-5 text-sm">
              <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-2">
                <p className="text-[10px] uppercase tracking-[0.18em] text-emerald-400/90 font-black">What this score means</p>
                <p className="text-white/85">{metric.meaning}</p>
              </section>

              <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-2">
                <p className="text-[10px] uppercase tracking-[0.18em] text-sky-400/90 font-black">Why the model assigned it</p>
                <p className="text-white/80">{metric.whyAssigned}</p>
              </section>

              <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-2">
                <p className="text-[10px] uppercase tracking-[0.18em] text-purple-400/90 font-black">Signals that contributed</p>
                {metric.indicators.length > 0 ? (
                  <ul className="space-y-1 text-white/75 list-disc list-inside">
                    {metric.indicators.map((indicator) => (
                      <li key={indicator}>{indicator}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-white/60">No granular indicators were provided by the engine for this metric.</p>
                )}
              </section>

              <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-2">
                <p className="text-[10px] uppercase tracking-[0.18em] text-amber-400/90 font-black">How to interpret it</p>
                <p className="text-white/80">{metric.interpretation}</p>
              </section>

              <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-2">
                <p className="text-[10px] uppercase tracking-[0.18em] text-fuchsia-400/90 font-black">Confidence context</p>
                <p className="text-white/80">{metric.confidenceContext}</p>
              </section>

              {(metric.evidence || sourceLinks.length > 0) && (
                <section className="rounded-2xl bg-white/[0.03] border border-white/10 p-4 space-y-3">
                  <p className="text-[10px] uppercase tracking-[0.18em] text-cyan-400/90 font-black">Evidence & sources</p>
                  {metric.evidence && <p className="text-white/80">{metric.evidence}</p>}
                  {sourceLinks.length > 0 && (
                    <div className="grid gap-2">
                      {sourceLinks.map((source) => {
                        const isLink = source.startsWith("http://") || source.startsWith("https://");
                        return (
                          <a
                            key={source}
                            href={isLink ? source : undefined}
                            target={isLink ? "_blank" : undefined}
                            rel={isLink ? "noopener noreferrer" : undefined}
                            className="text-xs p-2 rounded-lg bg-white/5 border border-white/10 text-white/80 break-all inline-flex items-center gap-2"
                          >
                            <span className="flex-1">{source}</span>
                            {isLink && <ExternalLink className="w-3 h-3 text-cyan-300" />}
                          </a>
                        );
                      })}
                    </div>
                  )}
                </section>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
