import React from "react";
import ReactDOM from "react-dom/client";

function App() {
  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h1>ViT Heatmap Demo</h1>
      <p>
        Upload <b>≥10 OK</b> images and <b>1 NOK</b> image to generate
        Grad-CAM / Attention heatmaps.
      </p>

      <p style={{ opacity: 0.7 }}>
        Backend: FastAPI · Model: Vision Transformer · Storage: in-memory
      </p>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
