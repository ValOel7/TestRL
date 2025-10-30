# app.py
import json, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Soweto RL – Business Strategy Simulation", layout="wide")

# -------------------------------------------------
# 1) Load data (from repo root)
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    try:
        hist = pd.read_csv("simulation_history.csv")
        cells = pd.read_csv("cell_day_marketshare.csv")
    except Exception as e:
        st.error(f"Could not read CSVs from repo root. {e}")
        st.stop()

    geo = None
    try:
        with open("soweto_boundary.geojson", "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception:
        geo = None  # optional
    return hist, cells, geo

history, cells, geo = load_data()

# -------------------------------------------------
# 2) Sidebar controls (incl. performance)
# -------------------------------------------------
st.sidebar.header("Controls")

# Playback
auto_play = st.sidebar.checkbox("Auto-play", value=True)
fps = st.sidebar.slider("Animation speed (frames/sec)", 1, 30, 10)
loop_mode = st.sidebar.selectbox("Loop mode", ["Stop at end", "Loop"], index=0)

# Map & display
point_radius = st.sidebar.slider("Point radius (m)", 10, 250, 40)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.9)
show_legend = st.sidebar.checkbox("Show legend", value=True)

# Performance
st.sidebar.subheader("Performance")
step_days = st.sidebar.slider("Days per frame (step size)", 1, 30, 5)
render_charts_live = st.sidebar.checkbox("Render charts while playing", value=False,
                                         help="Turn OFF for smoother playback.")
sample_frac = st.sidebar.slider("Map point fraction", 0.1, 1.0, 1.0,
                                help="Down-sample points per frame for speed.")

# Labels (visual only)
st.sidebar.subheader("Labels (display only)")
opp_entry_day_lbl = st.sidebar.number_input("OPP entry day (label)", 0, 365, 90)
takeover_rate_lbl = st.sidebar.number_input("Takeover rate (/day, label)", 0.0, 0.2, 0.02, 0.005)

# Persist for autoplay
st.session_state["_step_days"] = step_days
st.session_state["_fps"] = fps

# -------------------------------------------------
# 3) Prep state & constants
# -------------------------------------------------
max_day = int(cells["day"].max())

COLOR_MAP = {
    "FTM_share": [255, 140, 0],   # orange
    "LB_share" : [0, 128, 255],   # blue
    "OPP_share": [60, 179, 113],  # green
}
if "day" not in st.session_state: st.session_state.day = 0
if "playing" not in st.session_state: st.session_state.playing = auto_play

# Helper to build tidy numeric frames
def _melt_numeric(df, cols, value_name):
    cols = [c for c in cols if c in df.columns]
    m = df.melt(id_vars="day", value_vars=cols, var_name="type", value_name=value_name)
    m[value_name] = pd.to_numeric(m[value_name], errors="coerce").fillna(0.0)
    return m

# -------------------------------------------------
# 4) Title & Layout (Map left; Day control, Metrics, Conversions & Churn right)
# -------------------------------------------------
st.title("Soweto Subsistence Retail — Strategy Simulation")
st.caption("Files are read directly from the repo root. Use the sidebar for speed & rendering options.")

left, right = st.columns([1.8, 1.1])  # Map wider

# ===================== LEFT: MAP & AGGREGATE SHARE + EXPLANATION =====================
with left:
    st.subheader("Market Map")

    # Slice for current day
    cur = cells[cells["day"] == st.session_state.day].copy()

    # Optional down-sampling for speed
    if sample_frac < 1.0 and len(cur) > 0:
        cur = cur.sample(frac=sample_frac, random_state=st.session_state.day)

    # Fallback grid if missing coordinates
    if not {"lat","lon"}.issubset(cur.columns):
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    # Dominant owner & color
    cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    # Layers
    layers = []
    if geo:
        layers.append(pdk.Layer(
            "GeoJsonLayer", geo, stroked=True, filled=False,
            get_line_color=[60,60,60], line_width_min_pixels=1
        ))
    layers.append(pdk.Layer(
        "ScatterplotLayer", cur,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=point_radius,
        pickable=True, opacity=opacity
    ))

    view = pdk.ViewState(
        latitude=float(cur["lat"].mean()),
        longitude=float(cur["lon"].mean()),
        zoom=11 if geo else 9
    )
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="mapbox://styles/mapbox/light-v9"))

    if show_legend:
        st.markdown("**Legend:** 🟠 FTM  🔵 LB  🟢 OPP")

    # Aggregate Share under the map (optional)
    st.subheader("Aggregate Share Over Time")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” to show this live.")
    else:
        shares_long = _melt_numeric(history, ["FTM_share","LB_share","OPP_share"], "share_sum")
        share_chart = (
            alt.Chart(shares_long).mark_line().encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("share_sum:Q", title="Aggregate share"),
                color=alt.Color("type:N",
                    scale=alt.Scale(domain=["FTM_share","LB_share","OPP_share"],
                                    range=["#FF8C00","#0080FF","#3CB371"]),
                    legend=alt.Legend(title="Type")
                ),
                tooltip=["day","type","share_sum"]
            ).properties(height=220)
        )
        st.altair_chart(share_chart, use_container_width=True)

    # Explanation (always visible)
    st.markdown("""
---
### Explanation
**First-to-Market (FTM):** Captures share quickly due to novelty, but churn rises as competitors enter.  
**Loyalty-Based (LB):** Slower growth via retention and relationship marketing; lower churn, steady gains.  
**Opposition (OPP):** Enters later, grows by takeovers; sticky once established.
""")

# ===================== RIGHT: DAY CONTROL, METRICS, CONVERSIONS & CHURN =====================
with right:
    st.subheader("Day Control")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("⏮ Start"): st.session_state.day = 0
    with c2:
        if st.button("⏯ Play/Pause"): st.session_state.playing = not st.session_state.playing
    with c3:
        if st.button("⏭ End"): st.session_state.day = max_day

    # Single unified progress + scrubber (RIGHT ONLY)
    st.progress(min(st.session_state.day / max_day, 1.0))
    st.session_state.day = st.slider("Scrub day", 0, max_day, st.session_state.day, key="scrubber_right")

    # Key Metrics
    st.subheader("Key Metrics")
    row = history[history["day"] == st.session_state.day]
    if row.empty:
        st.write("No data for this day.")
    else:
        r = row.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("FTM Share", f"{r['FTM_share']:.1f}")
        m2.metric("LB Share", f"{r['LB_share']:.1f}")
        m3.metric("OPP Share", f"{r['OPP_share']:.1f}")
        st.caption(f"Labels — OPP entry: **day {opp_entry_day_lbl}**, takeover rate: **{takeover_rate_lbl:.3f}/day**")

    # Conversions & Churn (stacked on the right)
    st.subheader("Conversions per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” in the sidebar to show them live.")
    else:
        conv_long = _melt_numeric(history, ["FTM_conv","LB_conv","OPP_conv"], "conversions")
        conv_chart = (
            alt.Chart(conv_long).mark_line().encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("conversions:Q", title="Conversions per Day"),
                color=alt.Color("type:N", legend=alt.Legend(title="Type"))
            ).properties(height=200)
        )
        st.altair_chart(conv_chart, use_container_width=True)

    st.subheader("Churn per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” in the sidebar to show them live.")
    else:
        churn_long = _melt_numeric(history, ["FTM_churn","LB_churn","OPP_churn"], "churn")
        churn_chart = (
            alt.Chart(churn_long).mark_line().encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("churn:Q", title="Churn per Day"),
                color=alt.Color("type:N", legend=alt.Legend(title="Type"))
            ).properties(height=200)
        )
        st.altair_chart(churn_chart, use_container_width=True)

# -------------------------------------------------
# 5) Fast autoplay (no traceback)
# -------------------------------------------------
if auto_play and st.session_state.playing:
    step = int(st.session_state.get("_step_days", 5))
    nxt = st.session_state.day + step
    if nxt > max_day:
        if loop_mode == "Loop":
            nxt = 0
        else:
            nxt = max_day
            st.session_state.playing = False
    st.session_state.day = nxt
    time.sleep(max(0.0, 1.0 / max(1, st.session_state.get("_fps", 10))))
    st.rerun()
