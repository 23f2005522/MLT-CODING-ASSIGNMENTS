import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="KMeans CSV Analyser",
    page_icon="🐦‍🔥",
    layout="wide",
    

)

# ─────────────────────────────────────────────
#  Custom CSS  (dark industrial/utilitarian look)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0e0e10;
    color: #e8e8e8;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
code, .mono { font-family: 'Space Mono', monospace; }

.stButton>button {
    background: #1affb2;
    color: #0e0e10;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
    font-family: 'Syne', sans-serif;
    transition: 0.2s;
}
.stButton>button:hover { background: #00d494; }

.metric-box {
    background: #1a1a1f;
    border: 1px solid #2a2a35;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-label { color: #888; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }
.metric-value { color: #1affb2; font-size: 1.6rem; font-weight: 800; font-family: 'Space Mono'; }

hr { border-color: #2a2a35; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Pure-NumPy KMeans  (mirrors the assignment)
# ─────────────────────────────────────────────

def assign_clusters(X: np.ndarray, k: int, cluster_centers: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest cluster center (vectorised Euclidean)."""
    n = X.shape[0]
    z = np.zeros(n, dtype=int)
    for i in range(n):
        distances = np.array([np.linalg.norm(X[i] - cluster_centers[j]) for j in range(k)])
        z[i] = np.argmin(distances)
    return z


def compute_cluster_centers(X: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    """Compute mean of each cluster."""
    d = X.shape[1]
    centers = np.zeros((k, d))
    for i in range(k):
        pts = X[z == i]
        if len(pts) > 0:
            centers[i] = np.mean(pts, axis=0)
    return centers


def run_kmeans(X: np.ndarray, k: int, n_iter: int, init_indices=None, seed: int = 42):
    """Full KMeans loop identical to the assignment's 'Extra' section."""
    np.random.seed(seed)

    if init_indices is not None:
        centers = X[init_indices, :]
    else:
        # k-means++ style: pick k random rows as initial centers
        idx = np.random.choice(len(X), k, replace=False)
        centers = X[idx, :]

    history = []  # (z, centers) per iteration

    for _ in range(n_iter):
        z = assign_clusters(X, k, centers)
        centers = compute_cluster_centers(X, z, k)
        history.append((z.copy(), centers.copy()))

    inertia = sum(
        np.linalg.norm(X[i] - centers[z[i]]) ** 2
        for i in range(len(X))
    )
    return z, centers, inertia, history


# ─────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────
DARK_BG = "#0e0e10"
PANEL_BG = "#141418"
GRID_CLR = "#1e1e26"

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#666", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)

PALETTE = [
    "#1affb2", "#ff6b6b", "#ffd93d", "#6bcbff",
    "#d16bff", "#ff9f43", "#00cec9", "#fd79a8",
    "#a29bfe", "#55efc4",
]


def plot_2d(X, z, centers, col_x, col_y, title="KMeans Result"):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    k = len(centers)
    for i in range(k):
        pts = X[z == i]
        c = PALETTE[i % len(PALETTE)]
        ax.scatter(pts[:, 0], pts[:, 1], color=c, s=14, alpha=0.7, label=f"Cluster {i}")
        ax.scatter(centers[i, 0], centers[i, 1], color=c, s=200,
                   edgecolors="white", linewidths=1.5, marker="*", zorder=5)
    ax.set_xlabel(col_x, color="#888", fontsize=9)
    ax.set_ylabel(col_y, color="#888", fontsize=9)
    ax.set_title(title, color="#e8e8e8", fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.15, labelcolor="#ccc", fontsize=8)
    plt.tight_layout()
    return fig


def plot_3d(X, z, centers, col_names, title="KMeans Result (3-D)"):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)
    k = len(centers)
    for i in range(k):
        pts = X[z == i]
        c = PALETTE[i % len(PALETTE)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=c, s=10, alpha=0.6, label=f"Cluster {i}")
        ax.scatter(centers[i, 0], centers[i, 1], centers[i, 2],
                   color=c, s=200, edgecolors="white", linewidths=1, marker="*", zorder=5)
    ax.set_xlabel(col_names[0], color="#888", fontsize=8)
    ax.set_ylabel(col_names[1], color="#888", fontsize=8)
    ax.set_zlabel(col_names[2], color="#888", fontsize=8)
    ax.set_title(title, color="#e8e8e8", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#555", labelsize=7)
    ax.legend(framealpha=0.1, labelcolor="#ccc", fontsize=8)
    plt.tight_layout()
    return fig


def plot_elbow(inertias, k_range):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)
    ax.plot(k_range, inertias, color="#1affb2", marker="o",
            markersize=6, linewidth=2)
    ax.set_xlabel("k (number of clusters)", color="#888", fontsize=9)
    ax.set_ylabel("Inertia (SSE)", color="#888", fontsize=9)
    ax.set_title("Elbow Curve — choose k at the 'elbow'", color="#e8e8e8",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  App layout
# ─────────────────────────────────────────────

st.markdown("## 🐦‍🔥 KMeans CSV Analyser")
st.markdown("Upload any CSV → pick 2 or 3 columns → run KMeans on Numeric Columns (pure NumPy, no sklearn)")
st.markdown("---")

# ── Upload ──────────────────────────────────
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

# ── Read CSV ─────────────────────────────────
@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

df_raw = load_csv(uploaded.read())

st.markdown(f"**Loaded:** `{uploaded.name}` — {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
with st.expander("Preview data (first 10 rows)"):
    st.dataframe(df_raw.head(10), use_container_width=True)

# Only keep numeric columns
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns in your CSV.")
    st.stop()

st.markdown("---")

# ── Sidebar controls ─────────────────────────
with st.sidebar:
    st.markdown("# Configuration ")

    dim_choice = st.radio("Dimensions", ["2D (2 columns)", "3D (3 columns)"])
    n_dims = 2 if "2D" in dim_choice else 3

    if n_dims == 2:
        col_x = st.selectbox("X axis column", numeric_cols, index=0)
        col_y = st.selectbox("Y axis column", numeric_cols, index=min(1, len(numeric_cols) - 1))
        sel_cols = [col_x, col_y]
    else:
        col_x = st.selectbox("X axis column", numeric_cols, index=0)
        col_y = st.selectbox("Y axis column", numeric_cols,
                             index=min(1, len(numeric_cols) - 1))
        col_z = st.selectbox("Z axis column", numeric_cols,
                             index=min(2, len(numeric_cols) - 1))
        sel_cols = [col_x, col_y, col_z]

    st.markdown("---")
    k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
    n_iter = st.slider("Iterations", min_value=1, max_value=50, value=10)
    drop_na = st.checkbox("Drop rows with NaN in selected columns", value=True)

    st.markdown("---")
    show_elbow = st.checkbox("Show Elbow curve (k = 2…10)", value=False)
    show_iter  = st.checkbox("Show iteration-by-iteration animation", value=False)

# ── Build data matrix ─────────────────────────
sub = df_raw[sel_cols].copy()
if drop_na:
    sub = sub.dropna()
else:
    sub = sub.fillna(sub.mean())

X = sub.values.astype(float)

if len(X) < k:
    st.error(f"Only {len(X)} usable rows but k={k}. Reduce k or clean your data.")
    st.stop()

# ── Run KMeans ────────────────────────────────
col_run, col_info = st.columns([1, 3])
with col_run:
    run_btn = st.button("▶ Run KMeans")

if run_btn or "kmeans_done" in st.session_state:
    st.session_state["kmeans_done"] = True

    z, centers, inertia, history = run_kmeans(X, k, n_iter)

    # ── Metrics ──────────────────────────────
    counts = np.bincount(z, minlength=k)
    st.markdown("### 📊 Results")
    m_cols = st.columns(k + 2)
    with m_cols[0]:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Total Points</div>
          <div class='metric-value'>{len(X)}</div>
        </div>""", unsafe_allow_html=True)
    with m_cols[1]:
        st.markdown(f"""
        <div class='metric-box'>
          <div class='metric-label'>Inertia (SSE)</div>
          <div class='metric-value'>{inertia:,.1f}</div>
        </div>""", unsafe_allow_html=True)
    for i in range(k):
        with m_cols[i + 2]:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>Cluster {i}</div>
              <div class='metric-value'>{counts[i]}</div>
            </div>""", unsafe_allow_html=True)

    # ── Cluster centers table ──────────────────
    centers_df = pd.DataFrame(centers, columns=sel_cols)
    centers_df.index.name = "Cluster"
    with st.expander("Cluster Centers"):
        st.dataframe(centers_df.round(4), use_container_width=True)

    st.markdown("---")

    # ── Main plot ─────────────────────────────
    st.markdown("### 🗺️ Cluster Visualisation")
    if n_dims == 2:
        fig = plot_2d(X, z, centers, sel_cols[0], sel_cols[1],
                      title=f"KMeans k={k} — {n_iter} iterations")
        st.pyplot(fig, use_container_width=True)
    else:
        fig = plot_3d(X, z, centers, sel_cols,
                      title=f"KMeans k={k} — {n_iter} iterations (3-D)")
        st.pyplot(fig, use_container_width=True)

    # ── Elbow ─────────────────────────────────
    if show_elbow:
        st.markdown("### 📉 Elbow Curve")
        k_range = range(2, 11)
        inertias = []
        with st.spinner("Computing elbow (k=2…10)…"):
            for ki in k_range:
                _, _, ine, _ = run_kmeans(X, ki, n_iter)
                inertias.append(ine)
        st.pyplot(plot_elbow(inertias, k_range), use_container_width=True)

    # ── Iteration viewer ─────────────────────
    if show_iter and n_dims == 2:
        st.markdown("### 🔄 Iteration-by-iteration view")
        iter_idx = st.slider("Iteration", 1, len(history), len(history))
        z_i, centers_i = history[iter_idx - 1]
        fig_i = plot_2d(X, z_i, centers_i, sel_cols[0], sel_cols[1],
                        title=f"After iteration {iter_idx}")
        st.pyplot(fig_i, use_container_width=True)

    # ── Download labelled CSV ─────────────────
    st.markdown("---")
    out_df = sub.copy()
    out_df["cluster"] = z
    csv_bytes = out_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download labelled CSV",
        data=csv_bytes,
        file_name="kmeans_output.csv",
        mime="text/csv",
    )