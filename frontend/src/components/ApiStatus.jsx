import { useEffect, useState } from "react";
import { checkApiHealth, getApiBase } from "../lib/api";

export default function ApiStatus() {
  const [health, setHealth] = useState({ ok: null, status: "checking", detail: "" });

  useEffect(() => {
    let alive = true;
    async function ping() {
      const h = await checkApiHealth();
      if (alive) setHealth(h);
    }
    ping();
    const id = setInterval(ping, 30000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (health.ok === null) {
    return (
      <div className="text-xs px-3 py-2 bg-[#f4faf7] border border-[#0b3d2e]/10 text-[#3d5a4c]">
        Checking API…
      </div>
    );
  }

  if (!health.ok) {
    return (
      <div className="text-sm px-3 py-2 bg-[#b42318]/10 border border-[#b42318]/25 text-[#b42318]">
        <strong>API offline.</strong> {health.detail || "Backend unreachable."}
        {getApiBase() ? (
          <span className="block mt-1 text-xs opacity-80">Configured: {getApiBase()}</span>
        ) : (
          <span className="block mt-1 text-xs opacity-80">
            VITE_API_URL is empty — set it on Netlify to your Render URL.
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="text-xs px-3 py-2 bg-[#027a48]/10 border border-[#027a48]/20 text-[#027a48] flex flex-wrap gap-x-3 gap-y-1">
      <span>API online · {health.data?.inference_mode || health.status}</span>
      {health.data?.weights_loaded != null && (
        <span>{health.data.weights_loaded ? "weights loaded" : "baseline weights"}</span>
      )}
    </div>
  );
}
