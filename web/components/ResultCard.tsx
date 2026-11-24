"use client";

import { useState } from "react";
import { Evidence, Result, imageUrl } from "@/lib/api";

const SWATCH: Record<string, string> = {
  red: "#dc2626", maroon: "#7f1d1d", orange: "#ea580c", yellow: "#eab308", green: "#16a34a",
  teal: "#0d9488", blue: "#2563eb", navy: "#1e3a8a", purple: "#7c3aed", pink: "#ec4899",
  brown: "#78350f", beige: "#d6c3a1", black: "#111", white: "#fafafa", gray: "#9ca3af",
};

const BOX_COLORS = ["#f97316", "#22d3ee", "#a3e635", "#f472b6"];

function EvidenceChip({ e, i }: { e: Evidence; i: number }) {
  const ok = e.score >= 0.75;
  const weak = e.score > 0 && !ok;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] ${
        ok ? "border-good/40 text-good" : weak ? "border-line text-muted" : "border-line text-muted line-through"
      }`}
      title={`${e.source} evidence, score ${e.score.toFixed(2)}`}
    >
      <span className="h-2 w-2 rounded-full" style={{ background: BOX_COLORS[i % BOX_COLORS.length] }} />
      {e.colors.map((c) => (
        <span key={c} className="h-2.5 w-2.5 rounded-sm border border-black/10" style={{ background: SWATCH[c] }} />
      ))}
      {e.colors.join("/")} {e.garment}
      <span className="opacity-60">· {e.source === "none" ? "not found" : e.source}</span>
    </span>
  );
}

export default function ResultCard({ r, rank, compact = false }: { r: Result; rank: number; compact?: boolean }) {
  const [hover, setHover] = useState(false);
  const boxes = (r.evidence || [])
    .map((e, i) => ({ e, i }))
    .filter(({ e }) => e.detail?.box && e.score > 0);

  return (
    <article
      className="group overflow-hidden rounded-2xl border border-line bg-card transition hover:shadow-lg"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl(r.path)} alt={r.caption} className="block w-full" loading="lazy" />
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
          {boxes.map(({ e, i }) => {
            const [x0, y0, x1, y1] = e.detail.box as number[];
            return (
              <rect key={i} x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none"
                stroke={BOX_COLORS[i % BOX_COLORS.length]} strokeWidth={hover ? 3 : 2}
                vectorEffect="non-scaling-stroke" opacity={hover ? 1 : 0.8} />
            );
          })}
        </svg>
        <span className="absolute left-2 top-2 rounded-full bg-black/70 px-2 py-0.5 text-xs font-medium text-white">
          #{rank}
        </span>
        <span className="absolute right-2 top-2 rounded-full bg-white/85 px-2 py-0.5 text-[11px] text-black">
          {r.scene_top}
        </span>
      </div>
      {!compact && (
        <div className="space-y-2 p-3">
          {r.evidence && r.evidence.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {r.evidence.map((e, i) => <EvidenceChip key={i} e={e} i={i} />)}
            </div>
          )}
          {r.features && (
            <div className="grid grid-cols-5 gap-1">
              {Object.entries(r.features).map(([k, v]) => (
                <div key={k} title={`${k}: ${v.toFixed(2)}`}>
                  <div className="h-1 overflow-hidden rounded bg-line">
                    <div className="h-full bg-accent" style={{ width: `${Math.max(3, v * 100)}%` }} />
                  </div>
                  <div className="mt-0.5 truncate text-[10px] text-muted">{k}</div>
                </div>
              ))}
            </div>
          )}
          <p className="line-clamp-2 text-xs text-muted group-hover:line-clamp-none">{r.caption}</p>
        </div>
      )}
    </article>
  );
}
