const STEPS = [
  { t: "1 · Understand the query", d: "A free LLM (OpenRouter) or an offline rule parser turns the sentence into structure: garments with their own colors, the scene, and the style. Vague requests like “casual weekend outfit” are expanded into likely garments." },
  { t: "2 · Recall with three signals", d: "FashionSigLIP text→image vectors (the look), bge caption vectors (meaning), and BM25 over Florence-2 captions (exact words) each return 200 candidates, merged with reciprocal rank fusion." },
  { t: "3 · Re-rank with evidence", d: "Each candidate is scored on overall similarity, scene probability, and binding: is there a detected garment region — or a caption phrase — where this garment has this color? Swapped colors score low." },
  { t: "4 · Explain", d: "Every result shows which garments were matched, from which evidence, with their boxes drawn on the image." },
];

export default function How() {
  return (
    <div className="py-10">
      <h1 className="font-display text-4xl">How it works</h1>
      <p className="mt-2 max-w-3xl text-sm text-muted">
        CLIP-style models embed an image as one vector — a “bag of concepts”. They know a photo contains red, white, a
        tie and a shirt, but not which is which. We add region-level structure on top of a fashion-tuned encoder.
      </p>
      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {STEPS.map((s) => (
          <div key={s.t} className="rounded-2xl border border-line bg-card p-5">
            <h2 className="font-display text-2xl">{s.t}</h2>
            <p className="mt-2 text-sm text-muted">{s.d}</p>
          </div>
        ))}
      </div>
      <div className="mt-6 rounded-2xl border border-line bg-card p-5 text-sm">
        <h2 className="font-display text-2xl">Indexing (offline)</h2>
        <p className="mt-2 text-muted">
          600 images (Fashionpedia + Flickr30k) → FashionSigLIP global vector · YOLOS-Fashionpedia garment boxes →
          crop vectors + colors (LAB k-means fused with zero-shot) · CLIP prompt-ensemble scene scores · Florence-2
          captions → dense + BM25 · stored in Qdrant (images + regions collections).
        </p>
      </div>
    </div>
  );
}
