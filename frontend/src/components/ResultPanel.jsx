import { assetUrl } from "../lib/api";

export default function ResultPanel({ loading, result, preview }) {
  if (loading) {
    return (
      <div className="relative overflow-hidden border border-[#0b3d2e]/10 bg-white/70 p-6">
        <div className="scan-line" />
        <p className="font-[family-name:var(--font-display)] text-2xl text-[#0b3d2e]">
          Analyzing…
        </p>
        <p className="text-sm text-[#3d5a4c] mt-1">
          Extracting faces, running inference, building heatmap.
        </p>
      </div>
    );
  }

  if (!result) return null;

  const isFake = result.prediction === "FAKE";
  const heatmap = assetUrl(result.heatmap_url);
  const report = assetUrl(result.report_url);

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
        </div>
        <div className="text-right">
          <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c]">Confidence</p>
          <p className="text-3xl font-semibold text-[#0c1f17]">
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        {preview && (
          <figure>
            <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
              Input / preview
            </figcaption>
            <img src={preview} alt="Uploaded media preview" className="w-full object-cover max-h-64" />
          </figure>
        )}
        {heatmap && (
          <figure>
            <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
              Manipulation heatmap
            </figcaption>
            <img src={heatmap} alt="Grad-CAM or forensic heatmap" className="w-full object-cover max-h-64" />
          </figure>
        )}
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <Meta label="Model" value={result.model_name} />
        <Meta label="Frames analyzed" value={result.frames_analyzed} />
        <Meta label="Suspicious frames" value={result.suspicious_frames} />
        <Meta label="Processing" value={`${result.processing_time_sec}s`} />
      </dl>

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
