import { assetUrl, downloadJson } from "../lib/api";

export default function ResultPanel({ loading, result, preview }) {
  if (loading) {
    return (
      <div className="relative overflow-hidden border border-[#0b3d2e]/10 bg-white/70 p-6">
        <div className="scan-line" />
        <p className="font-[family-name:var(--font-display)] text-2xl text-[#0b3d2e]">
          Analyzing…
        </p>
        <p className="text-sm text-[#3d5a4c] mt-1">
          Multi-signal deepfake forensics in progress (model + Grad-CAM + risk scoring).
        </p>
      </div>
    );
  }

  if (!result) return null;

  const isFake = result.prediction === "FAKE";
  const heatmap = assetUrl(result.heatmap_url);
  const report = assetUrl(result.report_url);
  const isAudio = result.media_type === "audio";
  const risk = result.details?.risk || {
    level: result.risk_level,
    label: result.risk_label,
  };
  const explanation = result.explanation || result.details?.explanation || [];
  const frames = result.details?.frame_probabilities || [];
  const ensemble = result.details?.ensemble || [];

  return (
    <div className="border border-[#0b3d2e]/10 bg-white/75 p-6 space-y-5 animate-rise">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c]">Prediction</p>
          <p
            className={`font-[family-name:var(--font-display)] text-5xl leading-none mt-1 ${
              isFake ? "text-[#b42318]" : "text-[#027a48]"
            }`}
          >
            {result.prediction}
          </p>
          {risk?.label && (
            <p className="mt-2 text-sm font-semibold" style={{ color: risk.color || undefined }}>
              Risk: {(risk.level || result.risk_level || "").toUpperCase()} — {risk.label || result.risk_label}
            </p>
          )}
          {result.guest && (
            <p className="text-xs text-[#3d5a4c] mt-2">Guest scan · Sign in to save history & PDF</p>
          )}
        </div>
        <div className="text-right">
          <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c]">Confidence</p>
          <p className="text-3xl font-semibold text-[#0c1f17]">
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        {preview && !isAudio && (
          <figure>
            <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
              Input / preview
            </figcaption>
            <img src={preview} alt="Uploaded media preview" className="w-full object-cover max-h-64" />
          </figure>
        )}
        {preview && isAudio && (
          <figure>
            <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
              Audio input
            </figcaption>
            <audio controls src={preview} className="w-full" />
          </figure>
        )}
        {heatmap && (
          <figure>
            <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
              Manipulation heatmap
            </figcaption>
            <img src={heatmap} alt="Grad-CAM heatmap" className="w-full object-cover max-h-64" />
          </figure>
        )}
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <Meta label="Model" value={result.model_name} />
        <Meta label="Type" value={result.media_type || "—"} />
        <Meta label="Frames / clips" value={result.frames_analyzed ?? "—"} />
        <Meta label="Processing" value={`${result.processing_time_sec}s`} />
      </dl>

      {explanation.length > 0 && (
        <div className="bg-[#f4faf7] px-4 py-3">
          <p className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">Forensic notes</p>
          <ul className="space-y-1.5 text-sm text-[#0c1f17]">
            {explanation.map((line) => (
              <li key={line}>• {line}</li>
            ))}
          </ul>
        </div>
      )}

      {frames.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
            Video frame timeline (fake probability)
          </p>
          <div className="flex items-end gap-0.5 h-16">
            {frames.map((p, i) => (
              <div
                key={`${i}-${p}`}
                title={`Frame ${i + 1}: ${(p * 100).toFixed(1)}%`}
                className="flex-1 min-w-[3px]"
                style={{
                  height: `${Math.max(8, p * 100)}%`,
                  background: p >= 0.5 ? "#b42318" : "#1f6b4f",
                  opacity: 0.75 + p * 0.25,
                }}
              />
            ))}
          </div>
        </div>
      )}

      {ensemble.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">Ensemble votes</p>
          <div className="grid sm:grid-cols-3 gap-2 text-sm">
            {ensemble.map((v) => (
              <div key={v.model} className="bg-[#f4faf7] px-3 py-2">
                <p className="font-semibold">{v.model}</p>
                <p className={v.prediction === "FAKE" ? "text-[#b42318]" : "text-[#027a48]"}>
                  {v.prediction} · {((v.fake_probability ?? v.confidence) * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {(result.sha256 || result.details?.sha256) && (
        <p className="text-[11px] text-[#3d5a4c] break-all">
          SHA-256: {result.sha256 || result.details?.sha256}
        </p>
      )}

      <div className="flex flex-wrap gap-3">
        {report && (
          <a
            href={report}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 border border-[#0b3d2e] px-4 py-2 text-sm font-semibold text-[#0b3d2e] hover:bg-[#0b3d2e] hover:text-white transition"
          >
            Download forensic report
          </a>
        )}
        <button
          type="button"
          onClick={() => downloadJson(`deepguard-${Date.now()}.json`, result)}
          className="inline-flex items-center gap-2 border border-[#0b3d2e]/30 px-4 py-2 text-sm font-semibold text-[#0b3d2e] hover:bg-[#e8f2ec] transition"
        >
          Export JSON
        </button>
      </div>
    </div>
  );
}

function Meta({ label, value }) {
  return (
    <div className="bg-[#f4faf7] px-3 py-2">
      <dt className="text-[11px] uppercase tracking-wider text-[#3d5a4c]">{label}</dt>
      <dd className="font-semibold text-[#0c1f17] mt-0.5">{value}</dd>
    </div>
  );
}
