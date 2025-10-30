# app.py
import json, time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Soweto RL – Business Strategy Simulation", layout="wide")

# -------------------------------------------------
# 1) Load data from repo root
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

# ---- (NEW) Global stable center for Deck.gl so blank basemap doesn't recenter/flicker ----
if "global_center" not in st.session_state:
    df = cells.copy()
    if {"lat", "lon"}.issubset(df.columns):
        lat_c = float((df["lat"].min() + df["lat"].max()) / 2)
        lon_c = float((df["lon"].min() + df["lon"].max()) / 2)
    else:
        lat_c, lon_c = 0.0, 0.0  # fallback if no geos
    st.session_state["global_center"] = (lat_c, lon_c)

# -------------------------------------------------
# 2) Sidebar controls (incl. performance & map mode)
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

# Map rendering mode
st.sidebar.subheader("Map rendering")
static_map = st.sidebar.checkbox(
    "Static map (fast, no tiles)", value=True,
    help="Plain static scatter (Altair). Very stable and fast."
)
blank_basemap = st.sidebar.checkbox(
    "Blank basemap (Deck.gl)", value=False,
    help="For Deck.gl mode, render without Mapbox tiles to avoid rate-limit flicker."
)

# Performance
st.sidebar.subheader("Performance")
step_days = st.sidebar.slider("Days per frame (step size)", 1, 30, 5)
render_charts_live = st.sidebar.checkbox(
    "Render charts while playing", value=False,
    help="Turn OFF for smoother playback during animation."
)
sample_frac = st.sidebar.slider(
    "Map point fraction", 0.1, 1.0, 1.0,
    help="Down-sample points per frame for speed."
)

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
COLOR_HEX = {"FTM_share":"#FF8C00", "LB_share":"#0080FF", "OPP_share":"#3CB371"}

if "day" not in st.session_state: st.session_state.day = 0
if "playing" not in st.session_state: st.session_state.playing = auto_play

def _melt_numeric(df, cols, value_name):
    cols = [c for c in cols if c in df.columns]
    m = df.melt(id_vars="day", value_vars=cols, var_name="type", value_name=value_name)
    m[value_name] = pd.to_numeric(m[value_name], errors="coerce").fillna(0.0)
    return m

# -------------------------------------------------
# 4) Title & Layout (Map left; Controls/Metrics/Conv/Churn right)
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
    have_geo = {"lat","lon"}.issubset(cur.columns)
    if not have_geo:
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    # Dominant owner & color
    cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    if static_map:
        # ---------- STATIC ALTAR SCATTER (no tiles, very stable) ----------
        all_geo = cells if {"lat","lon"}.issubset(cells.columns) else cur
        lon_min, lon_max = float(all_geo["lon"].min()), float(all_geo["lon"].max())
        lat_min, lat_max = float(all_geo["lat"].min()), float(all_geo["lat"].max())

        dom_order = ["FTM_share","LB_share","OPP_share"]
        dom_colors = [COLOR_HEX[d] for d in dom_order]

        chart = (
            alt.Chart(cur)
            .mark_circle(opacity=opacity)
            .encode(
                x=alt.X("lon:Q", scale=alt.Scale(domain=[lon_min, lon_max]), title=None),
                y=alt.Y("lat:Q", scale=alt.Scale(domain=[lat_min, lat_max]), title=None),
                color=alt.Color("dom:N",
                                scale=alt.Scale(domain=dom_order, range=dom_colors),
                                legend=None),
                tooltip=["cell_id","dom","FTM_share","LB_share","OPP_share"]
            )
            .properties(height=520)
            .encode(size=alt.value(max(10, point_radius*0.6)))
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        # ---------- INTERACTIVE DECK.GL (stabilized blank basemap) ----------
        lat_c, lon_c = st.session_state["global_center"]

        layers = []
        if geo is not None:
            layers.append(pdk.Layer(
                "GeoJsonLayer", geo, stroked=True, filled=False,
                get_line_color=[80, 80, 80], line_width_min_pixels=1,
                pickable=False
            ))

        layers.append(pdk.Layer(
            "ScatterplotLayer", cur,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=point_radius,
            pickable=False,        # reduce GPU work per frame
            opacity=opacity
        ))

        view = pdk.ViewState(
            latitude=lat_c, longitude=lon_c,
            zoom=11, bearing=0, pitch=0
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            controller=False,      # no pan/zoom events while animating
            map_style=None if blank_basemap else "mapbox://styles/mapbox/light-v9",
            parameters={"clearColor": [0.98, 0.98, 0.98, 1.0]},  # solid bg, no flashes
            tooltip={"text": "{dom}\nFTM:{FTM_share}\nLB:{LB_share}\nOPP:{OPP_share}"},
        )
        st.pydeck_chart(deck, use_container_width=True, height=540, key="deckmap")

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

    # Single unified day slider (keep red slider, no progress bar)
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
        m2.metric("LB Share",  f"{r['LB_share']:.1f}")
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
# 5) Fast autoplay (uses new API only)
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
