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
        hist = pd.read_csv("simulation_history (2).csv")
        cells = pd.read_csv("cell_day_marketshare (1).csv")
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

# ---- Stable global center for Deck.gl ----
if "global_center" not in st.session_state:
    df = cells.copy()
    if {"lat", "lon"}.issubset(df.columns):
        lat_c = float((df["lat"].min() + df["lat"].max()) / 2)
        lon_c = float((df["lon"].min() + df["lon"].max()) / 2)
    else:
        lat_c, lon_c = 0.0, 0.0
    st.session_state["global_center"] = (lat_c, lon_c)

# -------------------------------------------------
# 2) Sidebar controls
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

# Life-cycle overlay toggle
show_lifecycle = st.sidebar.checkbox(
    "Show business life-cycle overlay", value=True,
    help="Overlay a conceptual market life-cycle curve on the aggregate share chart."
)

# Labels (visual only)
st.sidebar.subheader("Labels (display only)")
opp_entry_day_lbl = st.sidebar.number_input("OPP entry day (label)", 0, 365, 90)
takeover_rate_lbl = st.sidebar.number_input("Takeover rate (/day, label)", 0.0, 0.2, 0.02, 0.005)

# Persist for autoplay
st.session_state["_step_days"] = step_days
st.session_state["_fps"] = fps

# -------------------------------------------------
# 3) Prep state, constants & helper functions
# -------------------------------------------------
max_day = int(cells["day"].max())
COLOR_MAP = {
    "FTM_share": [255, 140, 0],   # orange
    "LB_share" : [0, 128, 255],   # blue
    "OPP_share": [60, 179, 113],  # green
}
COLOR_HEX = {"FTM_share":"#FF8C00", "LB_share":"#0080FF", "OPP_share":"#3CB371"}

if "day" not in st.session_state:
    st.session_state.day = 0
if "playing" not in st.session_state:
    st.session_state.playing = auto_play


def _melt_numeric(df, cols, value_name):
    cols = [c for c in cols if c in df.columns]
    m = df.melt(id_vars="day", value_vars=cols, var_name="type", value_name=value_name)
    m[value_name] = pd.to_numeric(m[value_name], errors="coerce").fillna(0.0)
    return m


# ---- Business life-cycle helpers (for overlay & insight) ----
STAGE_BOUNDS = [60, 140, 220, 300]
STAGE_LABELS = ["Launch", "Growth", "Shake-out", "Maturity", "Decline"]


def get_stage_idx(day: int) -> int:
    """0 = Launch, 1 = Growth, 2 = Shake-out, 3 = Maturity, 4 = Decline"""
    if day < 60:
        return 0
    elif day < 140:
        return 1
    elif day < 220:
        return 2
    elif day < 300:
        return 3
    else:
        return 4


# -------------------------------------------------
# 4) Title & Layout
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
    have_geo = {"lat", "lon"}.issubset(cur.columns)
    if not have_geo:
        n = cur["cell_id"].nunique()
        side = int(np.ceil(np.sqrt(n)))
        grid = [(i, j) for i in range(side) for j in range(side)][:n]
        cur = cur.sort_values("cell_id").copy()
        cur["lon"] = [g[0] for g in grid]
        cur["lat"] = [g[1] for g in grid]

    # Dominant owner & color
    cur["dom"] = cur[["FTM_share", "LB_share", "OPP_share"]].idxmax(axis=1)
    cur["color"] = cur["dom"].map(COLOR_MAP)

    if static_map:
        # ---------- STATIC ALTAIR SCATTER ----------
        all_geo = cells if {"lat", "lon"}.issubset(cells.columns) else cur
        lon_min, lon_max = float(all_geo["lon"].min()), float(all_geo["lon"].max())
        lat_min, lat_max = float(all_geo["lat"].min()), float(all_geo["lat"].max())

        dom_order = ["FTM_share", "LB_share", "OPP_share"]
        dom_colors = [COLOR_HEX[d] for d in dom_order]

        chart = (
            alt.Chart(cur)
            .mark_circle(opacity=opacity)
            .encode(
                x=alt.X("lon:Q", scale=alt.Scale(domain=[lon_min, lon_max]), title=None),
                y=alt.Y("lat:Q", scale=alt.Scale(domain=[lat_min, lat_max]), title=None),
                color=alt.Color(
                    "dom:N",
                    scale=alt.Scale(domain=dom_order, range=dom_colors),
                    legend=None,
                ),
                tooltip=["cell_id", "dom", "FTM_share", "LB_share", "OPP_share"],
            )
            .properties(height=520)
            .encode(size=alt.value(max(10, point_radius * 0.6)))
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        # ---------- INTERACTIVE DECK.GL ----------
        lat_c, lon_c = st.session_state["global_center"]

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
                    pickable=False,
                )
            )

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                cur,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=point_radius,
                pickable=False,
                opacity=opacity,
            )
        )

        view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=11, bearing=0, pitch=0)

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            controller=False,
            map_style=None if blank_basemap else "mapbox://styles/mapbox/light-v9",
            parameters={"clearColor": [0.98, 0.98, 0.98, 1.0]},
        )
        st.pydeck_chart(deck, use_container_width=True, height=540, key="deckmap")

    if show_legend:
        st.markdown("**Legend:** 🟠 FTM  🔵 LB  🟢 OPP")

    # ---------- Aggregate Share with optional life-cycle overlay ----------
    st.subheader("Aggregate Share Over Time")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” to show this live.")
    else:
        shares_long = _melt_numeric(history, ["FTM_share", "LB_share", "OPP_share"], "share_sum")

        base_chart = (
            alt.Chart(shares_long)
            .mark_line()
            .encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("share_sum:Q", title="Aggregate share"),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(
                        domain=["FTM_share", "LB_share", "OPP_share"],
                        range=["#FF8C00", "#0080FF", "#3CB371"],
                    ),
                    legend=alt.Legend(title="Type"),
                ),
                tooltip=["day", "type", "share_sum"],
            )
            .properties(height=220)
        )

        if show_lifecycle:
            # conceptual life-cycle curve (normalised, then scaled to share range)
            n_days = int(history["day"].max()) + 1
            t = np.arange(n_days)
            x_points = np.array([0, 40, 100, 180, 260, 330, n_days - 1])
            y_points = np.array([0.0, 0.05, 0.5, 1.0, 0.9, 0.6, 0.3])
            lc = np.interp(t, x_points, y_points)

            max_share = shares_long["share_sum"].max()
            lc_df = pd.DataFrame({"day": t, "life_cycle_scaled": lc * max_share})

            life_chart = (
                alt.Chart(lc_df)
                .mark_line(strokeDash=[4, 4], color="black")
                .encode(
                    x="day:Q",
                    y=alt.Y("life_cycle_scaled:Q", title="Aggregate share"),
                    tooltip=["day", "life_cycle_scaled"],
                )
            )

            # stage boundaries as vertical rules
            bounds_df = pd.DataFrame({"day": STAGE_BOUNDS})
            rules = (
                alt.Chart(bounds_df)
                .mark_rule(color="gray", strokeDash=[2, 2])
                .encode(x="day:Q")
            )

            combined = base_chart + life_chart + rules
            st.altair_chart(combined, use_container_width=True)
            st.caption(
                "Dashed black line shows a conceptual business life cycle. "
                "Dotted vertical rules mark stage boundaries: Launch, Growth, Shake-out, Maturity, Decline."
            )
        else:
            st.altair_chart(base_chart, use_container_width=True)

    # Explanation (always visible)
    st.markdown(
        """
---
### Explanation
**First-to-Market (FTM):** Captures share quickly due to novelty, but churn rises as competitors enter.  
**Loyalty-Based (LB):** Slower growth via retention and relationship marketing; lower churn, steady gains.  
**Opposition (OPP):** Enters later, grows by takeovers; sticky once established.
"""
    )

# ===================== RIGHT: DAY CONTROL, METRICS, CONVERSIONS & CHURN =====================
with right:
    st.subheader("Day Control")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("⏮ Start"):
            st.session_state.day = 0
    with c2:
        if st.button("⏯ Play/Pause"):
            st.session_state.playing = not st.session_state.playing
    with c3:
        if st.button("⏭ End"):
            st.session_state.day = max_day

    # Single unified day slider
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
        st.caption(
            f"Labels — OPP entry: **day {opp_entry_day_lbl}**, "
            f"takeover rate: **{takeover_rate_lbl:.3f}/day**"
        )

    # Conversions
    st.subheader("Conversions per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” in the sidebar to show them live.")
    else:
        conv_long = _melt_numeric(history, ["FTM_conv", "LB_conv", "OPP_conv"], "conversions")
        conv_chart = (
            alt.Chart(conv_long)
            .mark_line()
            .encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("conversions:Q", title="Conversions per Day"),
                color=alt.Color("type:N", legend=alt.Legend(title="Type")),
            )
            .properties(height=200)
        )
        st.altair_chart(conv_chart, use_container_width=True)

    # Churn
    st.subheader("Churn per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Uncheck “Render charts while playing” in the sidebar to show them live.")
    else:
        churn_long = _melt_numeric(history, ["FTM_churn", "LB_churn", "OPP_churn"], "churn")
        churn_chart = (
            alt.Chart(churn_long)
            .mark_line()
            .encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("churn:Q", title="Churn per Day"),
                color=alt.Color("type:N", legend=alt.Legend(title="Type")),
            )
            .properties(height=200)
        )
        st.altair_chart(churn_chart, use_container_width=True)

    # ---------- Strategy Insight Panel ----------
    st.subheader("Strategy Insight")

    if row.empty:
        st.write("No data to interpret for this day.")
    else:
        r = row.iloc[0]
        ftm_share = float(r["FTM_share"])
        lb_share = float(r["LB_share"])
        opp_share = float(r["OPP_share"])

        shares_dict = {
            "First-to-Market (FTM)": ftm_share,
            "Loyalty-Based (LB)": lb_share,
            "Opposition (OPP)": opp_share,
        }
        ordered = sorted(shares_dict.items(), key=lambda x: x[1], reverse=True)

        stage_idx = get_stage_idx(st.session_state.day)
        stage_name = STAGE_LABELS[stage_idx]

        st.markdown(f"**Conceptual life-cycle stage:** {stage_name}")

        st.markdown("**Ranking by aggregate share:**")
        for name, val in ordered:
            st.markdown(f"- {name}: {val:.1f}")

        leader = ordered[0][0]
        explanation = ""

        if stage_name in ["Launch", "Growth"]:
            if "FTM" in leader:
                explanation = (
                    "In the early launch/growth phase, the first-to-market strategy "
                    "benefits most from novelty and limited competition. This aligns "
                    "with classic pioneer advantage."
                )
            elif "LB" in leader:
                explanation = (
                    "Even in early stages, the loyalty-based strategy is performing well. "
                    "This suggests that relationship drivers (trust, empathy) are already "
                    "strong differentiators for subsistence customers."
                )
            else:  # OPP leading very early – unusual
                explanation = (
                    "Opposition is already strong very early in the cycle, indicating "
                    "aggressive takeovers or strong price undercutting in a still-developing market."
                )

        elif stage_name in ["Shake-out", "Maturity"]:
            if "Loyalty-Based" in leader or "LB" in leader:
                explanation = (
                    "In the shake-out/maturity phase, loyalty-based strategies tend to win. "
                    "The simulation shows that retention, customer trust and service quality "
                    "now matter more than being first into the market."
                )
            elif "Opposition" in leader:
                explanation = (
                    "Opposition dominates during shake-out/maturity, leveraging takeovers of "
                    "existing customer bases. This reflects intense competition and pressure "
                    "on incumbents with weaker loyalty."
                )
            else:  # FTM still leading
                explanation = (
                    "First-to-market is still leading in a mature market, but the gap is narrowing. "
                    "Without strengthening loyalty and customer relationships, its position is "
                    "likely to erode further."
                )

        else:  # Decline
            if "Opposition" in leader:
                explanation = (
                    "In the decline phase, opposition remains the dominant strategy, maintaining "
                    "share through entrenched positions and residual takeovers even as overall "
                    "market potential falls."
                )
            elif "Loyalty-Based" in leader or "LB" in leader:
                explanation = (
                    "Loyalty-based strategies retain a resilient core customer base in decline, "
                    "suggesting that strong relationships cushion the impact of shrinking demand."
                )
            else:  # FTM leading in decline
                explanation = (
                    "First-to-market is still visible in the decline phase, but largely as a "
                    "residual presence. Its long-term sustainability is weaker compared with "
                    "loyalty-oriented or takeover-oriented models."
                )

        st.markdown(f"**Interpretation:** {explanation}")

# -------------------------------------------------
# 5) Fast autoplay
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
