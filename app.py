# app.py
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------
# CONFIG – change these names to your actual files
# -------------------------------------------------
RAW_DATA_FILE = "Purchase_int.csv"         # <-- your raw cleaned dataset
BOUNDARY_FILE = "soweto_boundary.geojson"  # geojson you already have

st.set_page_config(page_title="Soweto RL – Business Strategy Simulation",
                   layout="wide")

# -------------------------------------------------
# 1) Helper: sample shop locations inside Soweto polygon
# -------------------------------------------------
def sample_points_in_polygon(polygon, n):
    minx, miny, maxx, maxy = polygon.bounds
    pts, tries, limit = [], 0, max(5000, n * 50)
    while len(pts) < n and tries < limit:
        x, y = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            pts.append(p)
        tries += 1
    if len(pts) < n:
        raise RuntimeError(f"Placed {len(pts)}/{n} points. Check boundary.")
    return gpd.GeoDataFrame(geometry=pts, crs=4326)


# -------------------------------------------------
# 2) Business life-cycle helpers
# -------------------------------------------------
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

# Conversion multipliers per stage & strategy [FTM, LB, OPP]
stage_conv_mult = np.array([
    [1.40, 0.50, 0.00],   # Launch
    [1.20, 0.90, 0.20],   # Growth
    [0.80, 1.15, 1.10],   # Shake-out
    [0.50, 1.20, 1.25],   # Maturity
    [0.30, 1.05, 1.00],   # Decline
])

# Churn multipliers per stage & strategy [FTM, LB, OPP]
stage_churn_mult = np.array([
    [0.60, 1.00, 1.00],   # Launch
    [1.00, 0.90, 1.00],   # Growth
    [1.60, 0.80, 1.00],   # Shake-out
    [2.20, 0.90, 1.10],   # Maturity
    [2.80, 1.00, 1.20],   # Decline
])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------------------------
# 3) Main: load raw data, preprocess, run simulation
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def run_simulation(seed: int = 42):
    np.random.seed(seed)

    # ---------- Load raw Soweto dataset ----------
    df = pd.read_csv(RAW_DATA_FILE)

    expected = [
        'Gender', 'Age', 'Marital_Status', 'Employment_Status',
        'Level_of_Education', 'Regular_Customer', 'Shopping_frequency',
        'Empathy', 'Convenience', 'Price_Sensitivity',
        'Physical_Environment', 'Perceived_Product_Quality',
        'Customer_Trust', 'Perceived_Value', 'Purchase_Intention'
    ]
    if not all(c in df.columns for c in expected):
        raise ValueError("Raw dataset does not have the expected columns.")

    df = df.dropna(subset=['Purchase_Intention']).copy()

    # ---------- Normalise key variables 0–1 ----------
    to_norm = [
        'Perceived_Value', 'Perceived_Product_Quality',
        'Physical_Environment', 'Convenience', 'Price_Sensitivity',
        'Empathy', 'Customer_Trust', 'Purchase_Intention'
    ]
    scaler = MinMaxScaler()
    df[[c + '_n' for c in to_norm]] = scaler.fit_transform(df[to_norm])

    # ---------- Correlation-based Demand Index ----------
    drivers = [
        'Perceived_Value_n', 'Price_Sensitivity_n', 'Perceived_Product_Quality_n',
        'Physical_Environment_n', 'Convenience_n', 'Empathy_n', 'Customer_Trust_n'
    ]
    corr = df[['Purchase_Intention_n'] + drivers].corr()['Purchase_Intention_n'].drop('Purchase_Intention_n')
    corr = corr.clip(lower=0)
    w = (corr / corr.sum()).to_dict()
    df['Demand_Index'] = sum(df[k] * w[k] for k in drivers)

    # ---------- Load Soweto geometry & sample shop points ----------
    sow = gpd.read_file(BOUNDARY_FILE)
    if sow.crs is None or sow.crs.to_epsg() != 4326:
        sow = sow.to_crs(4326)
    sow["geometry"] = sow.geometry.buffer(0)
    poly = unary_union(sow.geometry)

    shops_geo = sample_points_in_polygon(poly, len(df))
    shops = gpd.GeoDataFrame(df.reset_index(drop=True),
                             geometry=shops_geo.geometry, crs=4326)
    shops["lat"] = shops.geometry.y
    shops["lon"] = shops.geometry.x

    # ---------- Footfall (daily visitors) ----------
    lam0 = 60.0
    foot_scale = (
        0.35 * shops['Perceived_Value_n'] +
        0.25 * shops['Perceived_Product_Quality_n'] +
        0.15 * shops['Physical_Environment_n'] +
        0.15 * shops['Price_Sensitivity_n'] +
        0.10 * shops['Convenience_n']
    )
    lam = (lam0 * (0.5 + foot_scale)).clip(20, 150).values

    # ---------- Population / market capacity ----------
    pop_scale = MinMaxScaler().fit_transform(shops[['Shopping_frequency']]).flatten()
    pop = (50 * (0.5 + pop_scale)).astype(int)

    # ---------- Features for conversion logic ----------
    cell_attr = np.c_[
        shops['Purchase_Intention_n'].values,
        shops['Perceived_Value_n'].values,
        shops['Perceived_Product_Quality_n'].values,
        shops['Physical_Environment_n'].values,
        shops['Convenience_n'].values,
        shops['Price_Sensitivity_n'].values,
        shops['Empathy_n'].values,
        shops['Customer_Trust_n'].values,
        shops['Demand_Index'].values
    ]
    cells = len(shops)
    lat = shops['lat'].to_numpy()
    lon = shops['lon'].to_numpy()

    # ---------- Simulation parameters ----------
    days = 365
    share = np.zeros((cells, 3))      # columns: [FTM, LB, OPP]
    history, cell_history = [], []

    cap  = np.array([0.40, 0.35, 0.30])               # max conversion rate
    bias = np.array([0.05, 0.02, 0.06])

    monthly_churn_base = np.array([0.03, 0.02, 0.015])   # FTM, LB, OPP
    daily_churn_base = 1 - (1 - monthly_churn_base)**(1/30)

    opposition_entry_day = 90
    k_per_agent = {0: 7, 1: 3, 2: 5}                  # targeting intensity
    monthly_ftm_intent_decay = 0.0

    # LB loyalty mechanics
    tenure = np.zeros((cells, 3), dtype=int)
    loyalty_idx = np.zeros(cells)
    LOYALTY_GROW_RATE = 0.02
    LOYALTY_FROM = 0.5 * shops['Customer_Trust_n'].values + \
                   0.5 * shops['Empathy_n'].values

    # Seed FTM in top-demand cells
    seed_idx = np.argsort(cell_attr[:, -1])[-10:]
    share[seed_idx, 0] = 0.25

    # ---------- Main loop ----------
    for day in range(days):
        stage_idx = get_stage_idx(day)

        # Monthly rule: FTM loses 10% effective intent per month in owned cells
        if day > 0 and day % 30 == 0:
            monthly_ftm_intent_decay = min(0.9, monthly_ftm_intent_decay + 0.10)

        demand = cell_attr[:, -1]
        actions = []
        for agent in [0, 1, 2]:
            if agent == 2 and day < opposition_entry_day:
                continue
            k_now = k_per_agent[agent]
            target = np.argsort(demand + (1 - share[:, agent]))[-k_now:]
            actions.append((agent, target))

        conv = np.zeros((cells, 3))
        for agent, target in actions:
            for c in target:
                PI, PV, PQ, PE, Cn, PT, ET, CT, DI = cell_attr[c]
                months = day / 30.0

                base_p = sigmoid(
                    2.0*PV + 1.8*PQ + 1.2*PE + 1.0*Cn +
                    0.9*PT + 0.6*ET + 0.6*CT + bias[agent]
                )

                # FTM: novelty + decaying purchase intention
                if agent == 0:
                    novelty = np.exp(-months / 4.0)
                    base_p += 0.15 * novelty
                    if share[c, 0] > 0:
                        PI_eff = max(0.0, PI * (1 - monthly_ftm_intent_decay))
                        base_p += 0.5 * PI_eff

                # LB: slower, steadier
                if agent == 1:
                    base_p += 0.04
                    action_boost = np.random.uniform(0.01, 0.03)
                else:
                    action_boost = np.random.uniform(0.02, 0.06)

                # OPP: takeover bonus
                if agent == 2 and share[c, :2].sum() > 0:
                    base_p += 0.08

                # life-cycle conversion multiplier
                base_p *= stage_conv_mult[stage_idx, agent]

                p = min(cap[agent], max(0.0, base_p + action_boost))
                visitors = np.random.poisson(lam[c])
                new_sales = np.random.binomial(visitors, p)
                conv[c, agent] += new_sales

        # Update shares from conversions
        added_share = conv / np.maximum(1, pop[:, None])
        share = np.clip(share + added_share, 0, 1)

        # --- OPP explicit takeovers ---
        TAKEOVER_RATE = 0.02
        opp_cells = share[:, 2] > 0
        if opp_cells.any():
            ftm_loss = np.zeros(cells)
            lb_loss  = np.zeros(cells)
            ftm_loss[opp_cells] = TAKEOVER_RATE * share[opp_cells, 0]
            # LB loses 10% faster to OPP, moderated by loyalty
            lb_loss[opp_cells] = 1.10 * TAKEOVER_RATE * share[opp_cells, 1] * \
                                 (1 - loyalty_idx[opp_cells])

            transfer = ftm_loss + lb_loss
            capacity_left = 1.0 - share[:, 2]
            add_to_opp = np.minimum(transfer, capacity_left)
            scale = np.divide(add_to_opp, transfer,
                              out=np.zeros_like(transfer), where=transfer > 0)
            ftm_loss *= scale
            lb_loss  *= scale

            share[:, 0] = np.clip(share[:, 0] - ftm_loss, 0, 1)
            share[:, 1] = np.clip(share[:, 1] - lb_loss,  0, 1)
            share[:, 2] = np.clip(share[:, 2] + ftm_loss + lb_loss, 0, 1)

        # --- Update tenure & loyalty (LB only) ---
        present_lb = share[:, 1] > 0
        tenure[present_lb, 1] += 1
        tenure[~present_lb, 1] = 0
        loyalty_idx[present_lb] = np.clip(
            loyalty_idx[present_lb] + LOYALTY_GROW_RATE * LOYALTY_FROM[present_lb],
            0, 1
        )
        loyalty_idx[~present_lb] *= 0.98

        # --- Churn ---
        ftm_dc = daily_churn_base[0] * (1.0 + 1.0 * (day / 365.0))
        lb_dc  = daily_churn_base[1] * (1.0 - 0.5 * loyalty_idx)
        opp_dc = daily_churn_base[2]

        ftm_dc *= stage_churn_mult[stage_idx, 0]
        lb_dc  *= stage_churn_mult[stage_idx, 1]
        opp_dc *= stage_churn_mult[stage_idx, 2]

        churn = np.zeros_like(share)
        churn[:, 0] = share[:, 0] * ftm_dc
        churn[:, 1] = share[:, 1] * lb_dc
        churn[:, 2] = share[:, 2] * opp_dc

        share = np.clip(share - churn, 0, 1)

        history.append({
            'day': day,
            'FTM_share': float(share[:, 0].sum()),
            'LB_share':  float(share[:, 1].sum()),
            'OPP_share': float(share[:, 2].sum()),
            'FTM_conv':  float(conv[:, 0].sum()),
            'LB_conv':   float(conv[:, 1].sum()),
            'OPP_conv':  float(conv[:, 2].sum()),
            'FTM_churn': float(churn[:, 0].sum()),
            'LB_churn':  float(churn[:, 1].sum()),
            'OPP_churn': float(churn[:, 2].sum()),
        })

        cell_history.append(pd.DataFrame({
            'day': day,
            'cell_id': np.arange(cells),
            'lat': lat,
            'lon': lon,
            'FTM_share': share[:, 0],
            'LB_share':  share[:, 1],
            'OPP_share': share[:, 2],
            'loyalty_idx': loyalty_idx
        }))

    hist = pd.DataFrame(history)
    cell_hist = pd.concat(cell_history, ignore_index=True)
    return hist, cell_hist


# -------------------------------------------------
# 4) Run simulation once & build Streamlit UI
# -------------------------------------------------
with st.spinner("Running 365-day simulation..."):
    history, cells = run_simulation()

max_day = int(history["day"].max())
COLOR_HEX = {"FTM_share": "#FF8C00", "LB_share": "#0080FF", "OPP_share": "#3CB371"}

if "day" not in st.session_state:
    st.session_state.day = 0
if "playing" not in st.session_state:
    st.session_state.playing = True

def _melt_numeric(df, cols, value_name):
    cols = [c for c in cols if c in df.columns]
    m = df.melt(id_vars="day", value_vars=cols, var_name="type", value_name=value_name)
    m[value_name] = pd.to_numeric(m[value_name], errors="coerce").fillna(0.0)
    return m

# ---- Sidebar controls (no map-mode toggle anymore) ----
st.sidebar.header("Controls")
auto_play = st.sidebar.checkbox("Auto-play", value=True)
fps = st.sidebar.slider("Animation speed (frames/sec)", 1, 30, 10)
step_days = st.sidebar.slider("Days per frame (step size)", 1, 30, 5)
st.sidebar.subheader("Map display")
point_radius = st.sidebar.slider("Point radius (px)", 5, 40, 10)
opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.9)
render_charts_live = st.sidebar.checkbox("Render charts while playing", value=False)
show_lifecycle = st.sidebar.checkbox("Show business life-cycle overlay", value=True)

st.session_state["_step_days"] = step_days
st.session_state["_fps"] = fps

# ---- Layout ----
st.title("Soweto Subsistence Retail — Strategy Simulation")
st.caption("Simulation is run inside this app from the raw Soweto dataset.")

left, right = st.columns([1.8, 1.1])

# ================= LEFT: Map + aggregate share =================
with left:
    st.subheader("Market Map")

    cur = cells[cells["day"] == st.session_state.day].copy()
    if cur.empty:
        st.write("No cells for this day.")
    else:
        if {"lat", "lon"}.issubset(cur.columns):
            lon_min, lon_max = float(cells["lon"].min()), float(cells["lon"].max())
            lat_min, lat_max = float(cells["lat"].min()), float(cells["lat"].max())
        else:
            # fallback numeric grid
            n = cur["cell_id"].nunique()
            side = int(np.ceil(np.sqrt(n)))
            grid = [(i, j) for i in range(side) for j in range(side)][:n]
            cur = cur.sort_values("cell_id").copy()
            cur["lon"] = [g[0] for g in grid]
            cur["lat"] = [g[1] for g in grid]
            lon_min, lon_max = cur["lon"].min(), cur["lon"].max()
            lat_min, lat_max = cur["lat"].min(), cur["lat"].max()

        cur["dom"] = cur[["FTM_share", "LB_share", "OPP_share"]].idxmax(axis=1)

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
                size=alt.value(point_radius),
            )
            .properties(height=520)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**Legend:** 🟠 FTM  🔵 LB  🟢 OPP")

    # Aggregate share with optional life-cycle
    st.subheader("Aggregate Share Over Time")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed. Turn on 'Render charts while playing' to see them live.")
    else:
        shares_long = _melt_numeric(history,
                                    ["FTM_share", "LB_share", "OPP_share"],
                                    "share_sum")
        base_chart = (
            alt.Chart(shares_long)
            .mark_line()
            .encode(
                x=alt.X("day:Q", title="Day"),
                y=alt.Y("share_sum:Q", title="Aggregate share (sum across shops)"),
                color=alt.Color(
                    "type:N",
                    scale=alt.Scale(
                        domain=["FTM_share", "LB_share", "OPP_share"],
                        range=list(COLOR_HEX.values()),
                    ),
                    legend=alt.Legend(title="Strategy"),
                ),
                tooltip=["day", "type", "share_sum"],
            )
            .properties(height=220)
        )

        if show_lifecycle:
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
                .encode(x="day:Q", y="life_cycle_scaled:Q")
            )

            bounds_df = pd.DataFrame({"day": STAGE_BOUNDS})
            rules = (
                alt.Chart(bounds_df)
                .mark_rule(color="gray", strokeDash=[2, 2])
                .encode(x="day:Q")
            )

            st.altair_chart(base_chart + life_chart + rules, use_container_width=True)
        else:
            st.altair_chart(base_chart, use_container_width=True)

    st.markdown("""
---
### Explanation
**First-to-Market (FTM):** Enters first, grows fast, but suffers rising churn and 10% monthly decay in purchase intention.  
**Loyalty-Based (LB):** Enters later, grows slower, but builds loyalty that reduces churn and defends against opposition.  
**Opposition (OPP):** Enters last, only where others exist, and mainly grows by taking over existing share.
""")


# ================= RIGHT: controls + metrics =================
with right:
    st.subheader("Day Control")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⏮ Start"):
            st.session_state.day = 0
    with c2:
        if st.button("⏯ Play/Pause"):
            st.session_state.playing = not st.session_state.playing
    with c3:
        if st.button("⏭ End"):
            st.session_state.day = max_day

    st.session_state.day = st.slider("Scrub day",
                                    0, max_day,
                                    st.session_state.day)

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

    st.subheader("Conversions per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed.")
    else:
        conv_long = _melt_numeric(history,
                                  ["FTM_conv", "LB_conv", "OPP_conv"],
                                  "conversions")
        conv_chart = (
            alt.Chart(conv_long)
            .mark_line()
            .encode(
                x="day:Q", y="conversions:Q",
                color=alt.Color("type:N", legend=alt.Legend(title="Strategy"))
            )
            .properties(height=200)
        )
        st.altair_chart(conv_chart, use_container_width=True)

    st.subheader("Churn per Day")
    if (auto_play and st.session_state.playing) and (not render_charts_live):
        st.info("Charts paused for speed.")
    else:
        churn_long = _melt_numeric(history,
                                   ["FTM_churn", "LB_churn", "OPP_churn"],
                                   "churn")
        churn_chart = (
            alt.Chart(churn_long)
            .mark_line()
            .encode(
                x="day:Q", y="churn:Q",
                color=alt.Color("type:N", legend=alt.Legend(title="Strategy"))
            )
            .properties(height=200)
        )
        st.altair_chart(churn_chart, use_container_width=True)

# autoplay loop
if auto_play and st.session_state.playing:
    nxt = st.session_state.day + st.session_state.get("_step_days", 5)
    if nxt > max_day:
        nxt = max_day
        st.session_state.playing = False
    st.session_state.day = nxt
    time.sleep(max(0.0, 1.0 / max(1, st.session_state.get("_fps", 10))))
    st.rerun()
