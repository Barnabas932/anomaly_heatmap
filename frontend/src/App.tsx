import React, { useMemo, useState } from "react";
import { uploadHeatmap } from "./api";

export default function App() {
  const [okFiles, setOkFiles] = useState<File[]>([]);
  const [nokFile, setNokFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"gradcam" | "attn">("gradcam");
  const [alpha, setAlpha] = useState(0.45);

  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  const canRun = useMemo(() => okFiles.length >= 10 && !!nokFile, [okFiles, nokFile]);

  async function run() {
    if (!nokFile) return;
    setLoading(true);
    setMsg(null);
    setResultUrl(null);

    try {
      const out = await uploadHeatmap({
        okFiles,
        nokFile,
        mode,
        alpha,
      });
      setResultUrl(out.result_png);
      setMsg(`Kész! OK: ${out.ok_count}, NOK: ${out.nok_name}`);
    } catch (e: any) {
      setMsg(e.message || "Hiba");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={{ marginTop: 0 }}>Heatmap Demo (ViT)</h1>
        <p style={styles.small}>
          Tölts fel legalább <b>10 OK</b> képet és <b>1 NOK</b> képet. A rendszer a NOK képre készít heatmap-et.
        </p>

        <div style={styles.row}>
          <div style={styles.block}>
            <label style={styles.label}>OK képek (min. 10)</label>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={(e) => setOkFiles(Array.from(e.target.files || []))}
            />
            <div style={styles.small}>Kiválasztva: {okFiles.length}</div>
          </div>

          <div style={styles.block}>
            <label style={styles.label}>NOK kép (1 db)</label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setNokFile((e.target.files || [])[0] || null)}
            />
            <div style={styles.small}>Kiválasztva: {nokFile ? nokFile.name : "-"}</div>
          </div>
        </div>

        <div style={styles.row}>
          <div style={styles.block}>
            <label style={styles.label}>Mód</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
              <option value="gradcam">Grad-CAM</option>
              <option value="attn">Attention Map</option>
            </select>
          </div>

          <div style={styles.block}>
            <label style={styles.label}>Alpha (overlay erősség)</label>
            <input
              type="number"
              min={0.1}
              max={0.9}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
            />
          </div>
        </div>

        <button
          style={{
            ...styles.btn,
            opacity: canRun && !loading ? 1 : 0.5,
            cursor: canRun && !loading ? "pointer" : "not-allowed",
          }}
          disabled={!canRun || loading}
          onClick={run}
        >
          {loading ? "Feldolgozás..." : "Heatmap generálás"}
        </button>

        {msg && <div style={{ marginTop: 12 }}>{msg}</div>}

        {resultUrl && (
          <div style={{ marginTop: 16 }}>
            <div style={styles.small}>Eredmény:</div>
            <img src={resultUrl} alt="heatmap" style={styles.img} />
            <div style={{ marginTop: 8 }}>
              <a href={resultUrl} target="_blank" rel="noreferrer" style={styles.link}>
                PNG megnyitása / letöltése
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", display: "flex", justifyContent: "center", alignItems: "center", padding: 16 },
  card: { width: 900, maxWidth: "100%", padding: 20, borderRadius: 14, border: "1px solid #ddd" },
  small: { opacity: 0.8, fontSize: 13, lineHeight: 1.4 },
  row: { display: "flex", gap: 12, flexWrap: "wrap", marginTop: 12 },
  block: { flex: "1 1 320px" },
  label: { display: "block", fontWeight: 700, marginBottom: 6 },
  btn: { width: "100%", marginTop: 14, padding: 12, borderRadius: 10, border: "none", fontWeight: 800 },
  img: { width: "100%", maxWidth: 600, marginTop: 8, borderRadius: 10, border: "1px solid #ddd" },
  link: { textDecoration: "none", fontWeight: 700 },
};
