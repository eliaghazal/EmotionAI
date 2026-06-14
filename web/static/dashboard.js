/* dashboard.js — real-time chart updates, SocketIO, registration flow */

"use strict";

// ── Emotion config ──────────────────────────────────────────────────────────
const EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"];
const EMOTION_COLORS = {
  Angry:    "#FF4444",
  Disgust:  "#9B59B6",
  Fear:     "#E67E22",
  Happy:    "#F1C40F",
  Sad:      "#3498DB",
  Surprise: "#1ABC9C",
  Neutral:  "#95A5A6",
};

// ── SocketIO ────────────────────────────────────────────────────────────────
const socket = io();

let _currentEmbKey = null;  // emb_key of the face pending registration

// ── Timeline Chart ──────────────────────────────────────────────────────────
const timelineCtx = document.getElementById("timeline-chart").getContext("2d");
const timelineChart = new Chart(timelineCtx, {
  type: "line",
  data: {
    datasets: EMOTIONS.map(emo => ({
      label: emo,
      data: [],
      borderColor: EMOTION_COLORS[emo],
      backgroundColor: EMOTION_COLORS[emo] + "22",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.35,
      fill: false,
    })),
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Time (s)", color: "#8b949e" },
        ticks: { color: "#8b949e", maxTicksLimit: 8 },
        grid: { color: "#21262d" },
      },
      y: {
        min: -0.5, max: 6.5,
        ticks: {
          color: "#8b949e",
          callback: v => EMOTIONS[Math.round(v)] || "",
          stepSize: 1,
        },
        grid: { color: "#21262d" },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.dataset.label}`,
        },
      },
    },
  },
});

// ── Donut Chart ─────────────────────────────────────────────────────────────
const donutCtx = document.getElementById("donut-chart").getContext("2d");
const donutChart = new Chart(donutCtx, {
  type: "doughnut",
  data: {
    labels: EMOTIONS,
    datasets: [{
      data: new Array(7).fill(0),
      backgroundColor: EMOTIONS.map(e => EMOTION_COLORS[e]),
      borderColor: "#161b22",
      borderWidth: 3,
      hoverOffset: 8,
    }],
  },
  options: {
    animation: { duration: 400 },
    responsive: true,
    maintainAspectRatio: false,
    cutout: "58%",
    plugins: {
      legend: {
        position: "right",
        labels: {
          color: "#c9d1d9",
          boxWidth: 12,
          font: { size: 11 },
        },
      },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
        },
      },
    },
  },
});

// ── Session timer ────────────────────────────────────────────────────────────
let sessionStart = Date.now();
setInterval(() => {
  const s = Math.floor((Date.now() - sessionStart) / 1000);
  const m = Math.floor(s / 60);
  const sec = String(s % 60).padStart(2, "0");
  document.getElementById("session-duration").textContent = `${m}:${sec}`;
}, 1000);

// ── SocketIO: emotion_update ─────────────────────────────────────────────────
socket.on("emotion_update", ({ stats, timeline }) => {
  // Top-bar stat chips
  const dom = stats.dominant_emotion || "—";
  const domEl = document.getElementById("dominant-emotion");
  domEl.textContent = dom;
  domEl.style.color = EMOTION_COLORS[dom] || "#fff";

  document.getElementById("faces-count").textContent = stats.total_faces_detected ?? 0;
  const conf = stats.avg_confidence ?? 0;
  document.getElementById("avg-confidence").textContent = (conf * 100).toFixed(1) + "%";

  // Donut
  const dist = stats.distribution_percent || {};
  donutChart.data.datasets[0].data = EMOTIONS.map(e => dist[e] ?? 0);
  donutChart.update();

  // Timeline: plot emotion index vs time
  const tl = (timeline || []).slice(-120);
  timelineChart.data.datasets.forEach(ds => { ds.data = []; });
  tl.forEach(([t, emo]) => {
    const idx = EMOTIONS.indexOf(emo);
    if (idx >= 0) {
      timelineChart.data.datasets[idx].data.push({ x: parseFloat(t.toFixed(1)), y: idx });
    }
  });
  timelineChart.update();
});

// ── SocketIO: registration flow ──────────────────────────────────────────────
socket.on("registration_needed", ({ face_image, queue_position, queue_total, emb_key }) => {
  _currentEmbKey = emb_key;
  document.getElementById("reg-face-img").src = `data:image/jpeg;base64,${face_image}`;
  document.getElementById("reg-counter").textContent = `${queue_position} of ${queue_total}`;
  document.getElementById("reg-name-input").value = "";
  const panel = document.getElementById("reg-panel");
  panel.classList.remove("hidden");
  requestAnimationFrame(() => panel.classList.add("slide-in"));
  document.getElementById("reg-name-input").focus();
});

socket.on("registration_complete", ({ name, remaining }) => {
  if (name) {
    showToast(`✅ "${name}" registered!`);
    loadKnownFaces();
  }
  if (remaining === 0) {
    closeRegPanel();
  }
});

socket.on("all_registered", () => closeRegPanel());

function closeRegPanel() {
  const panel = document.getElementById("reg-panel");
  panel.classList.remove("slide-in");
  setTimeout(() => panel.classList.add("hidden"), 300);
}

// ── Registration actions ─────────────────────────────────────────────────────
function registerFace() {
  const name = document.getElementById("reg-name-input").value.trim();
  if (!name) {
    document.getElementById("reg-name-input").focus();
    return;
  }
  socket.emit("register_name", { name, emb_key: _currentEmbKey });
}

function skipFace() {
  socket.emit("skip_face", { emb_key: _currentEmbKey });
}

// ── Model switch ─────────────────────────────────────────────────────────────
function switchModel(type) {
  fetch("/api/switch_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_type: type }),
  })
    .then(r => r.json())
    .then(d => {
      if (d.status === "ok") {
        document.getElementById("btn-cnn").classList.toggle("active", type === "cnn");
        document.getElementById("btn-mobilenet").classList.toggle("active", type === "mobilenet");
        showToast(`Switched to ${type === "cnn" ? "Custom CNN" : "MobileNetV2"}`);
      } else {
        showToast("⚠ " + (d.error || "Unknown error"), "warn");
      }
    })
    .catch(() => showToast("⚠ Could not switch model", "warn"));
}

// ── Grad-CAM toggle ──────────────────────────────────────────────────────────
let gradcamActive = false;
let gradcamTimer = null;

function toggleGradcam() {
  fetch("/api/toggle_gradcam", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      gradcamActive = d.gradcam;
      const btn = document.getElementById("btn-gradcam");
      btn.classList.toggle("active", gradcamActive);
      const overlay = document.getElementById("gradcam-overlay");
      if (gradcamActive) {
        overlay.classList.remove("hidden");
        pollGradcam();
      } else {
        overlay.classList.add("hidden");
        clearTimeout(gradcamTimer);
      }
    });
}

function pollGradcam() {
  if (!gradcamActive) return;
  fetch("/api/gradcam")
    .then(r => r.json())
    .then(d => {
      if (d.heatmap) {
        document.getElementById("gradcam-img").src = `data:image/png;base64,${d.heatmap}`;
      }
    })
    .finally(() => {
      if (gradcamActive) gradcamTimer = setTimeout(pollGradcam, 1000);
    });
}

// ── Exports ──────────────────────────────────────────────────────────────────
function exportCSV() {
  window.location.href = "/api/export";
}
function exportPDF() {
  window.location.href = "/api/export_pdf";
}

// ── Known faces sidebar ──────────────────────────────────────────────────────
function toggleSidebar() {
  const sb = document.getElementById("faces-sidebar");
  sb.classList.toggle("open");
  if (sb.classList.contains("open")) loadKnownFaces();
}

function loadKnownFaces() {
  fetch("/api/known_faces")
    .then(r => r.json())
    .then(faces => {
      const container = document.getElementById("known-faces-list");
      if (!faces.length) {
        container.innerHTML = '<p class="empty-state">No registered faces yet.</p>';
        return;
      }
      container.innerHTML = faces.map(f => `
        <div class="face-card" id="face-${f.person_id}">
          <div style="flex:1">
            <div class="face-card-name">${escHtml(f.name)}</div>
            <div class="face-card-meta">${f.sample_count} sample${f.sample_count !== 1 ? "s" : ""}</div>
          </div>
          <button class="face-card-delete" onclick="deletePerson(${f.person_id})">Delete</button>
        </div>
      `).join("");
    });
}

function deletePerson(id) {
  if (!confirm("Remove this person from the registry?")) return;
  fetch(`/api/delete_person/${id}`, { method: "DELETE" })
    .then(r => r.json())
    .then(d => {
      if (d.deleted) {
        document.getElementById(`face-${id}`)?.remove();
        showToast("Person deleted.");
      }
    });
}

// ── Toast notifications ──────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const el = document.createElement("div");
  el.textContent = msg;
  Object.assign(el.style, {
    position: "fixed", bottom: "80px", left: "50%",
    transform: "translateX(-50%)",
    background: type === "warn" ? "#E67E22" : "#1ABC9C",
    color: "#fff", padding: "8px 18px", borderRadius: "8px",
    fontWeight: "600", fontSize: "13px", zIndex: "9999",
    boxShadow: "0 4px 16px rgba(0,0,0,.4)",
    opacity: "1", transition: "opacity .4s",
  });
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; }, 2200);
  setTimeout(() => el.remove(), 2700);
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
