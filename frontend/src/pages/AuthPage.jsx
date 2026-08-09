import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../lib/auth.jsx";

export default function AuthPage() {
  const { isAuthenticated, login, register } = useAuth();
  const [mode, setMode] = useState("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (isAuthenticated) return <Navigate to="/app" replace />;

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "login") {
        await login({ email, password });
      } else {
        await register({ email, password, full_name: fullName });
      }
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen grid lg:grid-cols-2">
      <section className="relative overflow-hidden px-8 py-12 lg:px-14 lg:py-16 flex flex-col justify-between">
        <div className="absolute inset-0 opacity-40" aria-hidden>
          <div className="absolute -left-20 top-24 h-72 w-72 rounded-full bg-[#3d9b74]/25 blur-3xl" />
          <div className="absolute right-0 bottom-10 h-80 w-80 rounded-full bg-[#c45c26]/15 blur-3xl" />
        </div>

        <div className="relative animate-rise">
          <p className="text-sm tracking-[0.28em] uppercase text-[#1f6b4f] font-semibold">
            DeepGuard AI
          </p>
          <h1 className="mt-5 max-w-xl font-[family-name:var(--font-display)] text-5xl md:text-6xl leading-[1.05] text-[#0c1f17]">
            See through the synthetic.
          </h1>
          <p className="mt-5 max-w-md text-[#3d5a4c] text-lg leading-relaxed">
            Forensic deepfake detection with face-aware analysis, Grad-CAM
            heatmaps, and downloadable reports.
          </p>
        </div>

        <div className="relative mt-12 animate-rise-delay">
          <div className="relative aspect-[4/3] max-w-lg overflow-hidden rounded-sm">
            <img
              src="https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?auto=format&fit=crop&w=1200&q=80"
              alt="Portrait used as atmospheric reference for face forensics"
              className="h-full w-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0b3d2e]/55 via-transparent to-transparent" />
            <div className="scan-line" />
            <div className="absolute bottom-4 left-4 right-4 text-white">
              <p className="font-[family-name:var(--font-display)] text-2xl">Face forensics layer</p>
              <p className="text-sm text-white/80">Frame sampling · Face crop · Manipulation heatmap</p>
            </div>
          </div>
        </div>
      </section>

      <section className="flex items-center justify-center px-6 py-12 lg:px-12">
        <form
          onSubmit={onSubmit}
          className="w-full max-w-md animate-rise-delay-2 bg-white/70 backdrop-blur-sm border border-[#0b3d2e]/10 p-8 shadow-[0_20px_60px_rgba(12,31,23,0.08)]"
        >
          <div className="flex gap-6 mb-8 border-b border-[#0b3d2e]/10">
            {["login", "register"].map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`pb-3 text-sm font-semibold tracking-wide uppercase transition ${
                  mode === m
                    ? "text-[#0b3d2e] border-b-2 border-[#0b3d2e]"
                    : "text-[#3d5a4c]/70"
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          {mode === "register" && (
            <label className="block mb-4">
              <span className="text-sm text-[#3d5a4c]">Full name</span>
              <input
                className="mt-1 w-full border border-[#0b3d2e]/15 bg-white px-3 py-2.5 outline-none focus:border-[#1f6b4f]"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </label>
          )}

          <label className="block mb-4">
            <span className="text-sm text-[#3d5a4c]">Email</span>
            <input
              type="email"
              className="mt-1 w-full border border-[#0b3d2e]/15 bg-white px-3 py-2.5 outline-none focus:border-[#1f6b4f]"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>

          <label className="block mb-6">
            <span className="text-sm text-[#3d5a4c]">Password</span>
            <input
              type="password"
              minLength={6}
              className="mt-1 w-full border border-[#0b3d2e]/15 bg-white px-3 py-2.5 outline-none focus:border-[#1f6b4f]"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>

          {error && <p className="mb-4 text-sm text-[#b42318]">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#0b3d2e] text-white py-3 font-semibold tracking-wide hover:bg-[#1f6b4f] transition disabled:opacity-60"
          >
            {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>
      </section>
    </div>
  );
}
