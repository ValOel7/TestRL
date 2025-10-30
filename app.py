import os, json, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt
import requests

st.set_page_config(page_title="Soweto RL – Business Strategy Simulation", layout="wide")

# -------------------------------------------------
# 0) CONFIG: where to read files (GitHub raw path)
# -------------------------------------------------
RAW_BASE = (
    st.secrets.get("RAW_BASE")
    or os.environ.get("RAW_BASE")
    or "https://raw.githubusercontent.com/ValOel7/TestRL"
)
# Filenames in that folder
HIST_NAME = "simulation_history.csv"
CELLS_NAME = "cell_day_marketshare.csv"
GEO_NAME = "soweto_boundary.geojson"  # optional

def url_join(base, name):
    if not base.endswith("/"): base += "/"
    return base + name

HIST_URL = url_join(RAW_BASE, HIST_NAME)
CELLS_URL = url_join(RAW_BASE, CELLS_NAME)
GEO_URL   = url_join(RAW_BASE, GEO_NAME)

# -------------------------------------------------
# 1) Sidebar: controls (no uploads)
# -------------------------------------------------
st.sidebar.header("Controls")
st.sidebar.markdown(f"**Data source:** `{RAW_BASE}`")

auto_play = st.sidebar.checkbox("Auto-play", value=True)
fps = st.sidebar.slider("Animation speed (frames/sec)", 1, 20, 6)
loop_mode = st.sidebar.selectbox("Loop mode", ["Stop at end", "Loop"], index=0)
point_radius = st.sidebar.slider("Point radius (m)", 10, 200, 35)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.9)
show_legend = st.sidebar.checkbox("Show legend", value=True)
st.sidebar.divider()
opp_entry_day_lbl = st.sidebar.number_input("Label: OPP entry day", 0, 365, 90)
takeover_rate_lbl = st.sidebar.number_input("Label: Takeover rate (/day)", 0.0, 0.2, 0.02, 0.005)
refresh = st.sidebar.button("🔄 Refresh data")

# -------------------------------------------------
# 2) Data loading via HTTP (cached)
# -------------------------------------------------
@st.cache_data(show_spinner=True, ttl=300)
def load_csv_from_url(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(pd.compat.StringIO(r.text))

@st.cache_data(show_spinner=False, ttl=300)
def load_geojson_from_url(url):
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

# Cache bust if user pressed Refresh
if refresh:
    load_csv_from_url.clear()
    load_geojson_from_url.clear()

# Try load
error_msgs = []
history = None
cells = None

try:
    history = load_csv_from_url(HIST_URL)
except Exception as e:
    error_msgs.append(f"Failed to fetch `{HIST_NAME}` from GitHub: {e}")

try:
    cells = load_csv_from_url(CELLS_URL)
except Exception as e:
    error_msgs.append(f"Failed to fetch `{CELLS_NAME}` from GitHub: {e}")

geo = load_geojson_from_url(GEO_URL)

if error_msgs:
    st.error(" \n\n".join(error_msgs))
    st.stop()

# -------------------------------------------------
# 3) Title & intro
# -------------------------------------------------
st.title("Soweto Subsistence Retail — Strategy Simulation")
st.caption(
    "Data auto-loaded from GitHub raw URLs. Use the sidebar to control playback and map display. "
    "Optional Soweto boundary is drawn if a GeoJSON is present at the configured path."
)

# -------------------------------------------------
# 4) Prepare data & session state
# -------------------------------------------------
max_day = int(cells["day"].max())

COLOR_MAP = {
    "FTM_share": [255, 140, 0],   # orange
    "LB_share" : [0, 128, 255],   # blue
    "OPP_share": [60, 179, 113],  # green
}

legend_md = """
**Legend**  
- **Orange** = First-to-Market (FTM)  
- **Blue** = Loyalty-Based (LB)  
- **Green** = Opposition (OPP)  
"""

if "day" not in st.session_state: st.session_state.day = 0
if "playing" not in st.session_state: st.session_state.playing = auto_play

# -------------------------------------------------
# 5) Layout: Map + Metrics
# -------------------------------------------------
top_col1, top_col2 = st.columns([1.4, 1.0])

with top_col1:
    st.subheader("Map")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("⏮ Start"): st.session_state.day = 0
    with c2:
        if st.button("⏯ Play/Pause"): st.session_state.playing = not st.session_state.playing
    with c3:
        if st.button("⏭ End"): st.session_state.day = max_day
    with c4:
        st.metric("Day", st.session_state.day)

    st.session_state.day = st.slider("Scrub day", 0, max_day, st.session_state.day)

    cur = cells[cells["day"] == st.session_state.day].copy()
    # If no lat/lon, make a small display grid
    if not {"lat","lon"}.issubset(cur.columns):
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    layers = []
    if geo is not None:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                geo,
                stroked=True,
                filled=False,
                get_line_color=[80, 80, 80],
                line_width_min_pixels=1,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            cur,
            get_position='[lon, lat]',
            get_radius=point_radius,
            get_fill_color='color',
            pickable=True,
            opacity=opacity,
        )
    )

    init_view = pdk.ViewState(
        latitude=float(cur["lat"].mean()),
        longitude=float(cur["lon"].mean()),
        zoom=11 if geo is not None else 9,
        pitch=0, bearing=0,
    )
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=init_view, map_style="mapbox://styles/mapbox/light-v9"))

    if show_legend:
        st.markdown(legend_md)

with top_col2:
    st.subheader("Key Metrics (Current Day)")
    hrow = history[history["day"] == st.session_state.day]
    if hrow.empty:
        st.write("No metrics for this day.")
    else:
        r = hrow.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("FTM share (sum)", f"{r['FTM_share']:.1f}")
        m2.metric("LB share (sum)", f"{r['LB_share']:.1f}")
        m3, m4 = st.columns(2)
        m3.metric("OPP share (sum)", f"{r['OPP_share']:.1f}")
        m4.metric("Day", int(r["day"]))
        st.caption(
            f"Labels — OPP entry: **day {opp_entry_day_lbl}**, takeover rate: **{takeover_rate_lbl:.3f}/day**"
        )

# -------------------------------------------------
# 6) Charts
# -------------------------------------------------
st.subheader("Trends")

shares_long = history.melt(id_vars="day",
                           value_vars=["FTM_share","LB_share","OPP_share"],
                           var_name="type", value_name="share_sum")
share_chart = (
    alt.Chart(shares_long)
    .mark_line()
    .encode(
        x=alt.X("day:Q", title="Day"),
        y=alt.Y("share_sum:Q", title="Aggregate share (sum of cell shares)"),
        color=alt.Color("type:N", scale=alt.Scale(
            domain=["FTM_share","LB_share","OPP_share"],
            range=["#FF8C00", "#0080FF", "#3CB371"]
        ), legend=alt.Legend(title="Type")),
        tooltip=["day","type","share_sum"]
    ).properties(height=250)
)
st.altair_chart(share_chart, use_container_width=True)

conv_long = history.melt(id_vars="day",
                         value_vars=["FTM_conv","LB_conv","OPP_conv"],
                         var_name="type", value_name="conversions")
conv_chart = (
    alt.Chart(conv_long).mark_line().encode(
        x="day:Q", y=alt.Y("conversions:Q", title="Conversions"),
        color=alt.Color("type:N", legend=alt.Legend(title="Conversions"))
    ).properties(height=220)
)

churn_long = history.melt(id_vars="day",
                          value_vars=["FTM_churn","LB_churn","OPP_churn"],
                          var_name="type", value_name="churn")
churn_chart = (
    alt.Chart(churn_long).mark_line().encode(
        x="day:Q", y=alt.Y("churn:Q", title="Churn"),
        color=alt.Color("type:N", legend=alt.Legend(title="Churn"))
    ).properties(height=220)
)

cA, cB = st.columns(2)
with cA: st.altair_chart(conv_chart, use_container_width=True)
with cB: st.altair_chart(churn_chart, use_container_width=True)

# -------------------------------------------------
# 7) Autoplay loop
# -------------------------------------------------
if auto_play and st.session_state.playing:
    nxt = st.session_state.day + 1
    if nxt > max_day:
        if loop_mode == "Loop":
            nxt = 0
        else:
            nxt = max_day
            st.session_state.playing = False
    st.session_state.day = nxt
    time.sleep(1.0 / fps)
    st.experimental_rerun()
