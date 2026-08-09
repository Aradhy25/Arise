const API_BASE = import.meta.env.VITE_API_URL || "";

export function getApiBase() {
  return API_BASE;
}

export async function checkApiHealth(timeoutMs = 8000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return { ok: false, status: "down", detail: `HTTP ${res.status}` };
    const data = await res.json();
    return { ok: data.status === "ok" || data.status === "degraded", status: data.status, data };
  } catch (err) {
    clearTimeout(t);
    const msg =
      err.name === "AbortError"
        ? "API timed out (Render free tier may be waking up — wait ~60s and retry)"
        : "Cannot reach API. Resume your Render service and set VITE_API_URL on Netlify.";
    return { ok: false, status: "unreachable", detail: msg };
  }
}

export async function api(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (!(options.body instanceof FormData) && options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    throw new Error(
      "Failed to reach DeepGuard API. If you use Netlify, make sure Render is Live and VITE_API_URL is set, then redeploy."
    );
  }

  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }

  if (!res.ok) {
    const detail = data?.detail;
    let message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join(", ")
          : res.statusText;
    if (res.status === 503 || (typeof text === "string" && text.includes("Suspended"))) {
      message = "API service is suspended on Render. Open Render dashboard → Resume service.";
    }
    throw new Error(message || "Request failed");
  }
  return data;
}

export function assetUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
