import json, time
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Soweto RL – Business Strategy Simulation", layout="wide")

# -------------------------------------------------
# 1. Load data directly from the repo root
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    try:
        hist = pd.read_csv("simulation_history.csv")
        cells = pd.read_csv("cell_day_marketshare.csv")
    except Exception as e:
        st.error(f"Could not read CSVs. Make sure they are in the repo root. {e}")
        st.stop()

    geo = None
    try:
        with open("soweto_boundary.geojson", "r", encoding="utf-8") as f:
            geo = json.load(f)
    except Exception:
        st.warning("GeoJSON not found — map will display points only.")
    return hist, cells, geo

history, cells, geo = load_data()

# -------------------------------------------------
# 2. Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")

auto_play = st.sidebar.checkbox("Auto-play", value=True)
fps = st.sidebar.slider("Animation speed (frames/sec)", 1, 20, 6)
loop_mode = st.sidebar.selectbox("Loop mode", ["Stop at end", "Loop"], index=0)
point_radius = st.sidebar.slider("Point radius (m)", 10, 200, 35)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.9)
show_legend = st.sidebar.checkbox("Show legend", value=True)

opp_entry_day_lbl = st.sidebar.number_input("Label: OPP entry day", 0, 365, 90)
takeover_rate_lbl = st.sidebar.number_input("Label: Takeover rate (/day)", 0.0, 0.2, 0.02, 0.005)

# -------------------------------------------------
# 3. Setup color map and session vars
# -------------------------------------------------
max_day = int(cells["day"].max())
COLOR_MAP = {
    "FTM_share": [255, 140, 0],   # orange
    "LB_share": [0, 128, 255],    # blue
    "OPP_share": [60, 179, 113],  # green
}

if "day" not in st.session_state: st.session_state.day = 0
if "playing" not in st.session_state: st.session_state.playing = auto_play

legend_md = """
**Legend**  
- 🟠 **FTM (First-to-Market)**  
- 🔵 **LB (Loyalty-Based)**  
- 🟢 **OPP (Opposition-Based)**  
"""

# -------------------------------------------------
# 4. Title & layout
# -------------------------------------------------
st.title("Soweto Subsistence Retail — Strategy Simulation")
st.caption("Data auto-loaded from GitHub repo root. Use the sidebar to control playback and view dynamics.")

top_col1, top_col2 = st.columns([1.4, 1.0])

# --- MAP + PLAYBACK ---
with top_col1:
    st.subheader("Market Map")
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
    if not {"lat","lon"}.issubset(cur.columns):
        st.warning("No lat/lon found in data — using synthetic grid for display.")
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    layers = []
    if geo:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer", geo, stroked=True, filled=False,
                get_line_color=[60,60,60], line_width_min_pixels=1,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer", cur,
            get_position='[lon, lat]', get_fill_color='color',
            get_radius=point_radius, pickable=True, opacity=opacity
        )
    )

    view = pdk.ViewState(
        latitude=float(cur["lat"].mean()),
        longitude=float(cur["lon"].mean()),
        zoom=11 if geo else 9
    )

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="mapbox://styles/mapbox/light-v9"))

    if show_legend: st.markdown(legend_md)

# --- METRICS ---
with top_col2:
    st.subheader("Key Metrics")
    row = history[history["day"] == st.session_state.day]
    if row.empty:
        st.write("No data for this day.")
    else:
        r = row.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("FTM share (sum)", f"{r['FTM_share']:.1f}")
        m2.metric("LB share (sum)", f"{r['LB_share']:.1f}")
        m3, m4 = st.columns(2)
        m3.metric("OPP share (sum)", f"{r['OPP_share']:.1f}")
        m4.metric("Day", int(r["day"]))
        st.caption(f"Labels — OPP entry: **day {opp_entry_day_lbl}**, takeover rate: **{takeover_rate_lbl:.3f}/day**")

# -------------------------------------------------
# 5. Charts
# -------------------------------------------------
st.subheader("Trends")

shares_long = history.melt(id_vars="day", value_vars=["FTM_share","LB_share","OPP_share"],
                           var_name="type", value_name="share_sum")
share_chart = (
    alt.Chart(shares_long)
    .mark_line()
    .encode(
        x="day:Q", y="share_sum:Q",
        color=alt.Color("type:N",
                        scale=alt.Scale(domain=["FTM_share","LB_share","OPP_share"],
                                        range=["#FF8C00","#0080FF","#3CB371"]),
                        legend=alt.Legend(title="Type")),
        tooltip=["day","type","share_sum"]
    )
    .properties(height=250, title="Aggregate Share over Time")
)
st.altair_chart(share_chart, use_container_width=True)

conv_long = history.melt(id_vars="day", value_vars=["FTM_conv","LB_conv","OPP_conv"],
                         var_name="type", value_name="conversions")
conv_chart = (
    alt.Chart(conv_long)
    .mark_line()
    .encode(x="day:Q", y="conversions:Q", color="type:N")
    .properties(height=220, title="Conversions per Day")
)

churn_long = history.melt(id_vars="day", value_vars=["FTM_churn","LB_churn","OPP_churn"],
                          var_name="type", value_name="churn")
churn_chart = (
    alt.Chart(churn_long)
    .mark_line()
    .encode(x="day:Q", y="churn:Q", color="type:N")
    .properties(height=220, title="Churn per Day")
)

c1, c2 = st.columns(2)
c1.altair_chart(conv_chart, use_container_width=True)
c2.altair_chart(churn_chart, use_container_width=True)

# -------------------------------------------------
# 6. Autoplay (animated day progression)
# -------------------------------------------------
if auto_play and st.session_state.playing:
    next_day = st.session_state.day + 1
    if next_day > max_day:
        if loop_mode == "Loop":
            next_day = 0
        else:
            next_day = max_day
            st.session_state.playing = False
    st.session_state.day = next_day
    time.sleep(1.0 / fps)
    st.experimental_rerun()
