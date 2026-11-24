"use client";

import { FormEvent, useState } from "react";
import ResultCard from "@/components/ResultCard";
import { SearchOut, search } from "@/lib/api";

const EXAMPLES = [
  "A person in a bright yellow raincoat.",
  "Professional business attire inside a modern office.",
  "Someone wearing a blue shirt sitting on a park bench.",
  "Casual weekend outfit for a city walk.",
  "A red tie and a white shirt in a formal setting.",
  "A white tie and a red shirt.",
];

export default function Home() {
  const [q, setQ] = useState("");
  const [compare, setCompare] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [data, setData] = useState<{ primary: SearchOut; compare: SearchOut | null; compare_note?: string | null } | null>(null);

  async function run(text: string) {
    if (text.trim().length < 2) return;
    setQ(text);
    setLoading(true);
    setErr("");
    try {
      setData(await search(text, compare));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  const spec = data?.primary.spec;

  return (
    <div>
      <section className="py-12 text-center sm:py-16">
        <h1 className="font-display text-4xl leading-tight sm:text-6xl">
          Find the outfit, <em className="text-accent">the place</em>, and the vibe.
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-muted">
          Natural-language fashion search that binds colors to the right garment — so “red tie, white shirt” never
          returns a white tie on a red shirt.
        </p>

        <form
          onSubmit={(e: FormEvent) => { e.preventDefault(); run(q); }}
          className="mx-auto mt-8 flex max-w-3xl flex-col gap-2 sm:flex-row"
        >
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Describe what you're looking for…"
            className="flex-1 rounded-full border border-line bg-card px-5 py-3 outline-none focus:border-accent"
          />
          <button
            disabled={loading}
            className="rounded-full bg-ink px-6 py-3 font-medium text-paper transition hover:opacity-90 disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </form>

        <label className="mt-4 inline-flex cursor-pointer items-center gap-2 text-sm text-muted">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} className="accent-[var(--color-accent)]" />
          Compare side-by-side with vanilla CLIP
        </label>

        <div className="mx-auto mt-5 flex max-w-4xl flex-wrap justify-center gap-2">
          {EXAMPLES.map((x) => (
            <button key={x} onClick={() => run(x)}
              className="rounded-full border border-line bg-card px-3 py-1.5 text-xs text-muted transition hover:border-accent hover:text-ink">
              {x}
            </button>
          ))}
        </div>
      </section>

      {err && (
        <p className="mb-6 rounded-xl border border-accent/40 bg-accent-soft p-4 text-sm">
          {err}. The free server sleeps when idle — the first search can take ~1 minute to wake it. Please retry.
        </p>
      )}
      {loading && <p className="mb-6 text-center text-sm text-muted">Searching… (first search after idle may take up to a minute)</p>}
      {data?.compare_note && <p className="mb-6 rounded-xl border border-line bg-card p-3 text-xs text-muted">{data.compare_note}</p>}

      {spec && (
        <section className="mb-8 rounded-2xl border border-line bg-card p-4 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase tracking-wider text-muted">Understood as</span>
            {spec.garments.map((g, i) => (
              <span key={i} className={`rounded-full px-2.5 py-1 text-xs ${g.inferred ? "border border-dashed border-line text-muted" : "bg-accent-soft"}`}>
                {[...g.modifiers, ...g.colors, g.surface].join(" ")}{g.inferred ? " (inferred)" : ""}
              </span>
            ))}
            {spec.scene && <span className="rounded-full bg-ink px-2.5 py-1 text-xs text-paper">📍 {spec.scene}</span>}
            {spec.style && <span className="rounded-full border border-line px-2.5 py-1 text-xs">style: {spec.style}</span>}
            <span className="ml-auto text-xs text-muted">
              {spec.parser.startsWith("llm") ? "LLM parser" : "rule parser"} · {data?.primary.timing.total_s}s
            </span>
          </div>
          <p className="mt-2 text-xs italic text-muted">“{spec.expanded}”</p>
        </section>
      )}

      {data && !data.compare && (
        <div className="columns-1 gap-4 sm:columns-2 lg:columns-3 xl:columns-4 [&>*]:mb-4 [&>*]:break-inside-avoid">
          {data.primary.results.map((r, i) => <ResultCard key={r.id} r={r} rank={i + 1} />)}
        </div>
      )}

      {data?.compare && (
        <div className="grid gap-6 md:grid-cols-2">
          {[
            { t: "Ours — hybrid + compositional re-rank", d: data.primary, full: true },
            { t: "Vanilla CLIP ViT-B/32", d: data.compare, full: false },
          ].map(({ t, d, full }) => (
            <div key={t}>
              <h2 className={`mb-3 font-display text-2xl ${full ? "text-accent" : "text-muted"}`}>{t}</h2>
              <div className="grid grid-cols-2 gap-3">
                {d.results.slice(0, 8).map((r, i) => <ResultCard key={r.id} r={r} rank={i + 1} compact={!full} />)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
