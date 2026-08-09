export default function HistoryList({ items, onSelect }) {
  if (!items?.length) {
    return <p className="text-sm text-[#3d5a4c]">No detections yet.</p>;
  }

  return (
    <ul className="space-y-2 max-h-[28rem] overflow-auto pr-1">
      {items.map((item) => {
        const fake = item.prediction === "FAKE";
        return (
          <li key={item.id}>
            <button
              type="button"
              onClick={() => onSelect?.(item)}
              className="w-full text-left px-3 py-2.5 bg-[#f4faf7] hover:bg-[#e8f2ec] transition border border-transparent hover:border-[#0b3d2e]/10"
            >
              <div className="flex items-center justify-between gap-2">
                <p className="font-medium text-sm truncate text-[#0c1f17]">{item.filename}</p>
                <span className={`text-xs font-bold ${fake ? "text-[#b42318]" : "text-[#027a48]"}`}>
                  {item.prediction}
                </span>
              </div>
              <p className="text-xs text-[#3d5a4c] mt-1">
                {(item.confidence * 100).toFixed(1)}% · {item.model_name} ·{" "}
                {new Date(item.created_at).toLocaleString()}
              </p>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
