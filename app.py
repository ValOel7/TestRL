import streamlit as st, pandas as pd, pydeck as pdk
st.set_page_config(page_title="Soweto RL Simulation", layout="wide")
cells = pd.read_csv("cell_day_marketshare.csv")
day = st.slider("Day", 0, int(cells["day"].max()), 0)
cur = cells[cells["day"]==day].copy()
cur["dom"] = cur[["FTM_share","LB_share","OPP_share"]].idxmax(axis=1)
color_map = {"FTM_share":[255,140,0], "LB_share":[0,128,255], "OPP_share":[60,179,113]}
cur["color"] = cur["dom"].map(color_map)
layer = pdk.Layer("ScatterplotLayer", cur, get_position='[lon, lat]', get_fill_color='color',
                  get_radius=25, pickable=True)
view = pdk.ViewState(latitude=cur["lat"].mean(), longitude=cur["lon"].mean(), zoom=11)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="mapbox://styles/mapbox/light-v9"))
st.caption("Orange=FTM, Blue=LB, Green=OPP. OPP enters day 90. Colors show dominant owner per point.")
