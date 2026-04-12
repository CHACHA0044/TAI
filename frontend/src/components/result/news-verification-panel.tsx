"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Newspaper,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Link,
} from "lucide-react";
import { NewsVerification, NewsArticle } from "@/lib/types";

// ---------------------------------------------------------------------------
// NewsVerificationPanel
// ---------------------------------------------------------------------------

interface NewsVerificationPanelProps {
  nv: NewsVerification;
}

export function NewsVerificationPanel({ nv }: NewsVerificationPanelProps) {
  const [expanded, setExpanded] = useState(false);

  const { Icon, color, border } = (() => {
    if (!nv.available) {
      return { Icon: AlertCircle, color: "text-white/40", border: "border-white/10" };
    }
    if (nv.contradiction_detected) {
      return { Icon: XCircle, color: "text-rose-400", border: "border-rose-500/30" };
    }
    const score = nv.corroboration_score ?? 0;
    if (score >= 0.65) {
      return { Icon: CheckCircle2, color: "text-emerald-400", border: "border-emerald-500/30" };
    }
    if (score >= 0.40) {
      return { Icon: AlertCircle, color: "text-amber-400", border: "border-amber-500/30" };
    }
    return { Icon: XCircle, color: "text-rose-400", border: "border-rose-500/30" };
  })();

  return (
    <div className={`glass rounded-3xl border ${border} p-6 space-y-4`}>
      {/* Header */}
      <div className="flex items-center gap-2">
        <Newspaper className={`w-4 h-4 ${color}`} />
        <h3 className="text-sm font-bold text-white/80 uppercase tracking-wider">
          News Verification
        </h3>
        {nv.source_count > 0 && (
          <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-white/50 font-mono">
            {nv.source_count} source{nv.source_count !== 1 ? "s" : ""} checked
          </span>
        )}
      </div>

      {/* Verification summary badge */}
      <div className={`flex items-center gap-3 p-3 rounded-xl bg-black/30 border ${border}`}>
        <Icon className={`w-5 h-5 flex-shrink-0 ${color}`} />
        <p className={`text-sm font-semibold ${color}`}>
          {nv.verification_label || nv.message}
        </p>
      </div>

      {/* Corroboration score bar */}
      {nv.corroboration_score !== null && nv.corroboration_score !== undefined && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">
              Corroboration strength
            </p>
            <p className="text-sm font-black font-mono text-white">
              {Math.round(nv.corroboration_score * 100)}%
            </p>
          </div>
          <div className="h-2 rounded-full bg-white/5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                nv.contradiction_detected
                  ? "bg-rose-500"
                  : nv.corroboration_score >= 0.65
                  ? "bg-emerald-500"
                  : nv.corroboration_score >= 0.40
                  ? "bg-amber-400"
                  : "bg-rose-500"
              }`}
              style={{ width: `${Math.round(nv.corroboration_score * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Quota exhausted message */}
      {!nv.available && (
        <p className="text-xs text-white/40 italic">{nv.message}</p>
      )}

      {/* Article cards */}
      {nv.available && nv.articles.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40 hover:text-white/70 transition-colors flex items-center gap-1"
          >
            {expanded ? "▲" : "▼"}{" "}
            {expanded ? "Hide" : "Show"} matched articles ({nv.articles.length})
          </button>

          <AnimatePresence initial={false}>
            {expanded && (
              <motion.div
                key="article-list"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-2 overflow-hidden"
              >
                {nv.articles.map((article, i) => (
                  <ArticleCard key={i} article={article} />
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ArticleCard
// ---------------------------------------------------------------------------

function ArticleCard({ article }: { article: NewsArticle }) {
  const domain = (() => {
    try {
      return new URL(article.url).hostname.replace(/^www\./, "");
    } catch {
      return article.source || "Unknown";
    }
  })();

  const date = (() => {
    if (!article.published_at) return null;
    try {
      return new Date(article.published_at).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return null;
    }
  })();

  return (
    <div className="rounded-xl border border-white/8 bg-white/[0.02] p-3 space-y-1.5">
      <div className="flex items-start gap-2">
        <Link className="w-3.5 h-3.5 text-sky-400 flex-shrink-0 mt-0.5" />
        {article.url ? (
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-semibold text-sky-300 hover:text-sky-100 line-clamp-2 leading-snug"
          >
            {article.title}
          </a>
        ) : (
          <p className="text-xs font-semibold text-white/80 line-clamp-2 leading-snug">
            {article.title}
          </p>
        )}
      </div>
      {article.description && (
        <p className="text-[11px] text-white/55 line-clamp-2 leading-relaxed pl-5">
          {article.description}
        </p>
      )}
      <div className="flex items-center gap-2 pl-5">
        <span className="text-[10px] font-bold text-white/40 uppercase tracking-wide">
          {domain}
        </span>
        {date && <span className="text-[10px] text-white/25">· {date}</span>}
      </div>
    </div>
  );
}
