import type { Metadata } from "next";
import { Inter, Instrument_Serif } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const display = Instrument_Serif({ subsets: ["latin"], weight: "400", variable: "--font-serif" });

export const metadata: Metadata = {
  title: "Multimodal Fashion & Context Retrieval",
  description: "Search fashion images by what people wear, where they are, and the vibe of the outfit.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${display.variable}`}>
      <body className="min-h-screen font-sans antialiased">
        <header className="border-b border-line">
          <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
            <Link href="/" className="font-display text-2xl tracking-tight">
              Fashion<span className="text-accent">·</span>Context
            </Link>
            <div className="flex gap-5 text-sm text-muted">
              <Link href="/" className="hover:text-ink">Search</Link>
              <Link href="/eval" className="hover:text-ink">Evaluation</Link>
              <Link href="/how" className="hover:text-ink">How it works</Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-7xl px-4 pb-20 sm:px-6">{children}</main>
      </body>
    </html>
  );
}
