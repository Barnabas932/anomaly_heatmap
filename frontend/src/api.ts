export async function uploadHeatmap(params: {
  okFiles: File[];
  nokFile: File;
  mode: "gradcam" | "attn";
  alpha: number;
}) {
  const fd = new FormData();
  params.okFiles.forEach((f) => fd.append("ok_files", f));
  fd.append("nok_file", params.nokFile);
  fd.append("mode", params.mode);
  fd.append("alpha", String(params.alpha));

  const res = await fetch("/api/heatmap", {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error || "Upload failed");
  }
  return res.json();
}
