export const API = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Evidence = {
  garment: string;
  colors: string[];
  score: number;
  source: "region" | "caption" | "none";
  detail: { box?: number[] | null; class?: string; colors?: Record<string, number>; mention?: { surface: string } };
};

export type Result = {
  id: string;
  path: string;
  score: number;
  caption: string;
  scene_top: string;
  source: string;
  width: number;
  height: number;
  features?: Record<string, number>;
  evidence?: Evidence[];
};

export type Spec = {
  text: string;
  garments: { garment: string; surface: string; colors: string[]; modifiers: string[]; inferred: boolean }[];
  scene: string | null;
  style: string | null;
  expanded: string;
  parser: string;
};

export type SearchOut = {
  query: string;
  mode: string;
  spec?: Spec;
  results: Result[];
  timing: { total_s: number; parse_s?: number };
};

export const imageUrl = (path: string) => `${API}/images/${path}`;

export async function search(query: string, compare: boolean, k = 12) {
  const r = await fetch(`${API}/api/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, k, mode: "hybrid", compare }),
  });
  if (!r.ok) throw new Error(`Search failed (${r.status})`);
  const d = await r.json();
  return compare ? (d as { primary: SearchOut; compare: SearchOut }) : { primary: d as SearchOut, compare: null };
}

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} failed (${r.status})`);
  return r.json();
}
