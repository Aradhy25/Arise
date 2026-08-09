import { useRef, useState } from "react";

export default function DropZone({ onFile, disabled }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) onFile(file);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={`relative border-2 border-dashed px-6 py-14 text-center transition ${
        dragging
          ? "border-[#1f6b4f] bg-[#1f6b4f]/8"
          : "border-[#0b3d2e]/25 bg-white/50 hover:border-[#1f6b4f]"
      } ${disabled ? "opacity-60 pointer-events-none" : ""}`}
    >
      <div className="relative inline-flex pulse-ring mb-4 h-14 w-14 items-center justify-center rounded-full bg-[#0b3d2e] text-white">
        <span className="text-xl">↑</span>
      </div>
      <p className="font-[family-name:var(--font-display)] text-2xl text-[#0c1f17]">
        Drag & drop media
      </p>
      <p className="mt-2 text-sm text-[#3d5a4c]">
        Images (JPG, PNG, WEBP) · Videos (MP4, MOV, AVI, MKV) · Audio (WAV, MP3, M4A, FLAC)
      </p>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-6 bg-[#0b3d2e] text-white px-6 py-2.5 font-semibold hover:bg-[#1f6b4f] transition"
      >
        Upload
      </button>
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept="image/*,video/*,audio/*"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
