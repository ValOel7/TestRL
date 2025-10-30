import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="Soweto Business Strategy Simulation", layout="wide")

# ---------------------------------------
# Sidebar: Data inputs & controls
# ---------------------------------------
st.sidebar.header("Data & Controls")

# Uploads
hist_file = st.sidebar.file_uploader("Upload simulation_history.csv", type=["csv"])
cells_file = st.sidebar.file_uploader("Upload cell_day_marketshare.csv", type=["csv"])
geojson_file = st.sidebar.file_uploader("Optional: Soweto boundary GeoJSON", type=["geojson","json"])

# Playback controls
st.sidebar.subheader("Playback")
auto_play = st.sidebar.checkbox("Auto-play", value=True)
fps = st.sidebar.slider("Animation speed (frames/sec)", 1, 20, 6, help="Higher = faster animation")
loop_mode = st.sidebar.selectbox("Loop mode", ["Stop at end", "Loop"], index=0)

# Display controls
st.sidebar.subheader("Map/Display")
point_radius = st.sidebar.slider("Point radius (m)", 10, 200, 35)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.9)
show_legend = st.sidebar.checkbox("Show legend", value=True)

# “What-if” knobs for labelling/annotations (visual only – data already baked into CSVs)
st.sidebar.subheader("Annotate Assumptions (labels only)")
opp_entry_day_lbl = st.sidebar.number_input("Opposition entry day (label)", min_value=0, max_value=365, value=90)
takeover_rate_lbl = st.sidebar.number_input("Takeover rate (label)", min_value=0.0, max_value=0.2, value=0.02, step=0.005, help="Label next to charts")

# ---------------------------------------
# Load data
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(f, expected_cols=None):
    df = pd.read_csv(f)
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.warning(f"Missing columns in uploaded file: {missing}")
    return df

history = None
cells = None

if hist_file is not None:
    history = load_csv(hist_file)
else:
    st.info("Upload **simulation_history.csv** (from Colab) in the sidebar.")
if cells_file is not None:
    cells = load_csv(cells_file)
else:
    st.info("Upload **cell_day_marketshare.csv** (from Colab) in the sidebar.")

boundary_layer = None
if geojson_file is not None:
    try:
        geo = json.load(geojson_file)
        boundary_layer = pdk.Layer(
            "GeoJsonLayer",
            geo,
            stroked=True,
            filled=False,
            get_line_color=[80, 80, 80],
            line_width_min_pixels=1,
        )
    except Exception as e:
        st.warning(f"Could not read boundary file: {e}")

# ---------------------------------------
# UI: Title & description
# ---------------------------------------
st.title("Soweto Subsistence Retail — Strategy Simulation")
st.caption(
    "Autoplay shows daily dynamics for 365 days. Use the sidebar to control playback and display. "
    "Upload your CSVs generated in Colab. Optional: add a Soweto boundary GeoJSON for the outline."
)

# Guard clause
if (history is None) or (cells is None):
    st.stop()

# ---------------------------------------
# Prepare derived data
# ---------------------------------------
max_day = int(cells["day"].max())
# Dominant owner color mapping
COLOR_MAP = {
    "FTM_share": [255, 140, 0],   # orange
    "LB_share" : [0, 128, 255],   # blue
    "OPP_share": [60, 179, 113],  # green
}

# Legend
legend_md = """
**Legend**  
- **Orange** = First-to-Market (FTM)  
- **Blue** = Loyalty-Based (LB)  
- **Green** = Opposition (OPP)  
"""

# ---------------------------------------
# Layout
# ---------------------------------------
top_col1, top_col2 = st.columns([1.4, 1.0])

with top_col1:
    st.subheader("Map")
    # Playback state
    if "day" not in st.session_state:
        st.session_state.day = 0
    if "playing" not in st.session_state:
        st.session_state.playing = auto_play

    # Controls row
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("⏮ Start"):
            st.session_state.day = 0
    with c2:
        if st.button("⏯ Play/Pause"):
            st.session_state.playing = not st.session_state.playing
    with c3:
        if st.button("⏭ End"):
            st.session_state.day = max_day
    with c4:
        st.metric("Day", st.session_state.day)

    # Slider (also acts as scrub bar)
    st.session_state.day = st.slider("Scrub day", 0, max_day, st.session_state.day)

    # Slice cells for current day
    cur = cells[cells["day"] == st.session_state.day].copy()
    # If no lat/lon, generate a simple grid just to show something
    if not {"lat","lon"}.issubset(cur.columns):
        st.warning("No lat/lon in cell CSV. Generating a temporary grid for display.")
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    # Map layer
    scatter = pdk.Layer(
        "ScatterplotLayer",
        cur,
        get_position='[lon, lat]',
        get_radius=point_radius,
        get_fill_color='color',
        pickable=True,
        opacity=opacity,
    )

    layers = [scatter]
    if boundary_layer is not None:
        layers.insert(0, boundary_layer)

    init_view = pdk.ViewState(
        latitude=float(cur["lat"].mean()),
        longitude=float(cur["lon"].mean()),
        zoom=11 if geojson_file is not None else 9,
        pitch=0,
        bearing=0,
    )

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=init_view, map_style="mapbox://styles/mapbox/light-v9"))

    if show_legend:
        st.markdown(legend_md)

with top_col2:
    st.subheader("Key Metrics (Current Day)")
    # Pull current day totals
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
        st.caption(f"Labels → OPP entry: **day {opp_entry_day_lbl}**, takeover rate: **{takeover_rate_lbl:.3f}/day**")

# ---------------------------------------
# Charts
# ---------------------------------------
st.subheader("Trends")

# Shares over time
shares_long = history.melt(id_vars="day", value_vars=["FTM_share","LB_share","OPP_share"],
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
    )
    .properties(height=250)
)
st.altair_chart(share_chart, use_container_width=True)

# Conversions / Churn over time
conv_long = history.melt(id_vars="day", value_vars=["FTM_conv","LB_conv","OPP_conv"],
                         var_name="type", value_name="conversions")
conv_chart = (
    alt.Chart(conv_long)
    .mark_line()
    .encode(
        x="day:Q", y=alt.Y("conversions:Q", title="Conversions"),
        color=alt.Color("type:N", legend=alt.Legend(title="Conversions"))
    )
    .properties(height=220)
)

churn_long = history.melt(id_vars="day", value_vars=["FTM_churn","LB_churn","OPP_churn"],
                          var_name="type", value_name="churn")
churn_chart = (
    alt.Chart(churn_long)
    .mark_line()
    .encode(
        x="day:Q", y=alt.Y("churn:Q", title="Churn"),
        color=alt.Color("type:N", legend=alt.Legend(title="Churn"))
    )
    .properties(height=220)
)

cA, cB = st.columns(2)
with cA:
    st.altair_chart(conv_chart, use_container_width=True)
with cB:
    st.altair_chart(churn_chart, use_container_width=True)

# ---------------------------------------
# Autoplay loop (keeps the app responsive)
# ---------------------------------------
if auto_play and st.session_state.playing:
    # Increment day by 1 frame; throttle via fps
    next_day = st.session_state.day + 1
    if next_day > max_day:
        if loop_mode == "Loop":
            next_day = 0
        else:
            next_day = max_day
            st.session_state.playing = False
    st.session_state.day = next_day
    # Sleep a bit to control frame rate
    time.sleep(1.0 / fps)
    st.experimental_rerun()

