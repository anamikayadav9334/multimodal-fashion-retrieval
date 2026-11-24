"use client";

import { useEffect, useState } from "react";
import { getJSON } from "@/lib/api";

type M = Record<string, number>;
type EvalOut = { mean: Record<string, M>; per_query: Record<string, { query: string; type: string } & Record<string, M>> };

const NAMES: Record<string, string> = {
  baseline: "Vanilla CLIP", fashion: "FashionSigLIP only", hybrid: "Ours (hybrid)",
  "-binding": "Ours − binding", "-text": "Ours − caption/BM25", "-scene": "Ours − scene", rules: "Ours, rule parser",
};

function Bar({ v, strong }: { v: number; strong?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 flex-1 overflow-hidden rounded bg-line">
        <div className={`h-full ${strong ? "bg-accent" : "bg-muted/60"}`} style={{ width: `${v * 100}%` }} />
      </div>
      <span className="w-9 text-right tabular-nums text-xs">{v.toFixed(2)}</span>
    </div>
  );
}

export default function EvalPage() {
  const [d, setD] = useState<EvalOut | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    // bundled static copy first (instant, works while the free API sleeps); API as fallback
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH || ""}/results.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .catch(() => getJSON<EvalOut>("/api/eval"))
      .then(setD)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="py-10 text-muted">{err}</p>;
  if (!d) return <p className="py-10 text-muted">Loading…</p>;

  return (
    <div className="py-10">
      <h1 className="font-display text-4xl">Evaluation</h1>
      <p className="mt-2 max-w-3xl text-sm text-muted">
        The 5 assignment queries. Pooled top-10 of each system, relevance graded 0/1/2 by looking at the images
        (not the captions our system uses). P@5 counts only full matches; nDCG uses graded relevance.
      </p>

      <div className="mt-8 grid gap-6 md:grid-cols-2">
        {(["nDCG@10", "P@5"] as const).map((metric) => (
          <div key={metric} className="rounded-2xl border border-line bg-card p-5">
            <h2 className="mb-4 text-sm font-medium">{metric} — mean over 5 queries</h2>
            <div className="space-y-3">
              {["baseline", "fashion", "hybrid"].map((s) => (
                <div key={s}>
                  <div className="mb-1 text-xs text-muted">{NAMES[s]}</div>
                  <Bar v={d.mean[s][metric]} strong={s === "hybrid"} />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 rounded-2xl border border-line bg-card p-5">
        <h2 className="mb-1 text-sm font-medium">Ablation — nDCG@5 (remove one component)</h2>
        <p className="mb-4 text-xs text-muted">Every removal hurts: each part contributes.</p>
        <div className="space-y-3">
          {["hybrid", "-binding", "-text", "-scene", "rules"].map((s) => (
            <div key={s} className="grid grid-cols-[10rem_1fr] items-center gap-3">
              <span className="text-xs text-muted">{NAMES[s]}</span>
              <Bar v={d.mean[s]["nDCG@5"]} strong={s === "hybrid"} />
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 overflow-x-auto rounded-2xl border border-line bg-card">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr className="border-b border-line">
              <th className="p-3">Query</th><th className="p-3">Vanilla CLIP</th><th className="p-3">FashionSigLIP</th><th className="p-3">Ours</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(d.per_query).map(([qid, r]) => (
              <tr key={qid} className="border-b border-line last:border-0">
                <td className="p-3"><span className="text-xs text-muted">{qid} · {r.type}</span><br />{r.query}</td>
                {["baseline", "fashion", "hybrid"].map((s) => (
                  <td key={s} className={`p-3 tabular-nums ${s === "hybrid" ? "font-semibold text-accent" : ""}`}>
                    {r[s]["nDCG@10"].toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
