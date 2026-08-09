import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="min-h-screen">
      <header className="mx-auto max-w-6xl px-5 py-5 flex items-center justify-between">
        <p className="font-[family-name:var(--font-display)] text-3xl text-[#0b3d2e]">DeepGuard AI</p>
        <nav className="flex items-center gap-3 text-sm">
          <Link to="/scan" className="hidden sm:inline text-[#3d5a4c] hover:text-[#0b3d2e]">
            Try free
          </Link>
          <Link to="/auth" className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
            Sign in
          </Link>
          <Link to="/scan" className="bg-[#0b3d2e] text-white px-4 py-1.5 font-semibold hover:bg-[#1f6b4f] transition">
            Scan now
          </Link>
        </nav>
      </header>

      <section className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-50" aria-hidden>
          <div className="absolute -left-24 top-10 h-80 w-80 rounded-full bg-[#3d9b74]/20 blur-3xl" />
          <div className="absolute right-0 top-40 h-96 w-96 rounded-full bg-[#c45c26]/12 blur-3xl" />
        </div>

        <div className="relative mx-auto max-w-6xl px-5 pt-10 pb-16 grid lg:grid-cols-[1.05fr_0.95fr] gap-10 items-center">
          <div className="animate-rise">
            <p className="text-xs tracking-[0.28em] uppercase text-[#1f6b4f] font-semibold">
              Worldwide deepfake defense
            </p>
            <h1 className="mt-4 font-[family-name:var(--font-display)] text-5xl md:text-6xl leading-[1.05] text-[#0c1f17]">
              DeepGuard AI
            </h1>
            <p className="mt-5 max-w-xl text-lg text-[#3d5a4c] leading-relaxed">
              Professional forensic detection for face-swaps, AI portraits, video forgeries,
              and voice clones — available on any browser, phone, or desktop.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                to="/scan"
                className="bg-[#0b3d2e] text-white px-6 py-3 font-semibold hover:bg-[#1f6b4f] transition"
              >
                Scan media free
              </Link>
              <Link
                to="/live"
                className="border border-[#0b3d2e] text-[#0b3d2e] px-6 py-3 font-semibold hover:bg-[#0b3d2e] hover:text-white transition"
              >
                Live camera
              </Link>
            </div>
            <p className="mt-4 text-xs text-[#3d5a4c]">
              No install required · Works on Windows, macOS, Linux, iOS, Android · Installable PWA
            </p>
          </div>

          <div className="relative animate-rise-delay">
            <div className="relative aspect-[4/5] overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?auto=format&fit=crop&w=1200&q=80"
                alt="Human face used as atmospheric visual for forensic analysis"
                className="h-full w-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#0b3d2e]/70 via-transparent to-transparent" />
              <div className="scan-line" />
              <div className="absolute bottom-5 left-5 right-5 text-white">
                <p className="font-[family-name:var(--font-display)] text-3xl">Multi-modal forensics</p>
                <p className="text-sm text-white/85 mt-1">Image · Video · Audio · Live webcam</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-14 animate-rise-delay-2">
        <h2 className="font-[family-name:var(--font-display)] text-3xl text-[#0b3d2e]">What it detects</h2>
        <p className="mt-2 max-w-2xl text-[#3d5a4c]">
          DeepGuard combines a PyTorch visual model, Grad-CAM explainability, and audio spectral
          forensics. Results are decision-support evidence — always review critical cases manually.
        </p>
        <div className="mt-8 grid md:grid-cols-3 gap-6">
          {[
            ["Face swaps & reenactment", "Crops faces, scores manipulation probability, highlights suspicious regions."],
            ["AI-generated portraits", "Catches blending seams, compression anomalies, and network attention cues."],
            ["Voice clones & TTS", "Analyzes spectral flatness and high-frequency artifacts in audio uploads."],
          ].map(([title, body]) => (
            <div key={title} className="border border-[#0b3d2e]/10 bg-white/60 p-5">
              <h3 className="font-[family-name:var(--font-display)] text-2xl text-[#0c1f17]">{title}</h3>
              <p className="mt-2 text-sm text-[#3d5a4c] leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-[#0b3d2e] text-white">
        <div className="mx-auto max-w-6xl px-5 py-14 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <p className="font-[family-name:var(--font-display)] text-4xl">Use it anywhere</p>
            <p className="mt-2 text-white/80 max-w-xl">
              Open in Chrome, Safari, Edge, or Firefox. Add to your home screen for an app-like
              experience on phone and desktop.
            </p>
          </div>
          <Link to="/scan" className="bg-white text-[#0b3d2e] px-6 py-3 font-semibold hover:bg-[#e8f2ec] transition self-start">
            Start scanning
          </Link>
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-5 py-8 text-xs text-[#3d5a4c] flex flex-wrap gap-4 justify-between">
        <p>© {new Date().getFullYear()} DeepGuard AI</p>
        <p>Forensic decision-support · Not absolute legal proof</p>
      </footer>
    </div>
  );
}
