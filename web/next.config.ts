import type { NextConfig } from "next";

// Static export for GitHub Pages. The site lives under /<repo-name>, set via NEXT_PUBLIC_BASE_PATH
// at build time (empty for local dev). Search calls go to the API at NEXT_PUBLIC_API_URL.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
