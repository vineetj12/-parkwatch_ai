import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="ParkWatch AI — Traffic Congestion Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vehicle weight → congestion proxy (bigger vehicle = more blockage)
VEHICLE_WEIGHTS = {
    "BUS": 3.0,
    "TANKER": 3.0,
    "TRUCK": 3.0,
    "LORRY": 3.0,
    "TIPPER": 2.5,
    "MAXI-CAB": 2.0,
    "VAN": 1.8,
    "LMV": 1.5,
    "LMV-NT": 1.5,
    "PASSENGER AUTO": 1.2,
    "CAR": 1.2,
    "JEEP": 1.2,
    "TAXI": 1.2,
    "MOTOR CYCLE": 0.6,
    "SCOOTER": 0.6,
    "MOPED": 0.6,
    "BICYCLE": 0.3,
}

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Card Styles */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-desc {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 5px;
    }
    /* Section Title Styles */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #818cf8;
        padding-left: 10px;
        color: #f3f4f6;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Caching ---
@st.cache_data(show_spinner="Loading and preprocessing Bengaluru traffic dataset (~110MB)...")
def load_data():
    csv_path = "dataset.csv"
    if not os.path.exists(csv_path):
        return pd.DataFrame()
        
    cols_to_use = [
        "id", "latitude", "longitude", "location", "vehicle_number", "vehicle_type", 
        "violation_type", "created_datetime", "police_station", "junction_name"
    ]
    df = pd.read_csv(csv_path, usecols=cols_to_use)
    
    # Clean coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    # Bengaluru bounding box filter
    df = df[(df["latitude"] >= 12.80) & (df["latitude"] <= 13.35) & 
            (df["longitude"] >= 77.40) & (df["longitude"] <= 77.80)]
            
    # Filter to parking-related violations
    PARKING_KEYWORDS = ["PARKING", "NO PARKING", "WRONG PARKING"]
    def is_parking_string(val):
        val_str = str(val).upper()
        return any(kw in val_str for kw in PARKING_KEYWORDS)
    df = df[df["violation_type"].apply(is_parking_string)]
    
    # Parse Datetime
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"])
    df["created_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour"] = df["created_ist"].dt.hour
    df["day_of_week"] = df["created_ist"].dt.day_name()
    df["date_str"] = df["created_ist"].dt.date.astype(str)
    
    # PCU weights mapping
    df["pcu"] = df["vehicle_type"].str.upper().str.strip().map(VEHICLE_WEIGHTS).fillna(1.0)
    
    # Junction flag
    df["is_junction"] = df["junction_name"].notna() & (~df["junction_name"].str.upper().str.strip().isin(["", "NO JUNCTION", "NULL"]))
    
    # Peak hour flag
    PEAK_HOURS = {7, 8, 9, 10, 16, 17, 18, 19, 20}
    df["is_peak"] = df["hour"].isin(PEAK_HOURS)
    
    # Congestion Impact Score formula per violation:
    # base = PCU weight (occupancy footprint)
    # multiplier = 1.5 if near junction (blocks turns), 2.0 if during peak commute hours (exponential delays)
    df["traffic_score"] = df["pcu"] * (1.0 + 0.5 * df["is_junction"].astype(float)) * (1.0 + 1.0 * df["is_peak"].astype(float))
    
    return df

df_raw = load_data()

if df_raw.empty:
    st.error("Error: Could not locate or load the `dataset.csv` file. Please ensure it is present in the workspace.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=64)
st.sidebar.title("ParkWatch AI Settings")
st.sidebar.markdown("Filter traffic congestion metrics dynamically below.")

# 1. Police Station Filter
all_stations = sorted(df_raw["police_station"].dropna().unique())
selected_stations = st.sidebar.multiselect(
    "Police Station Zone(s)",
    options=all_stations,
    default=[]
)

# 2. Time Filters
hour_range = st.sidebar.slider(
    "Hour of Day Range (IST)",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_days = st.sidebar.multiselect(
    "Day of the Week",
    options=all_days,
    default=all_days
)

# 3. Vehicle Type Filter
all_vehicles = sorted(df_raw["vehicle_type"].dropna().unique())
selected_vehicles = st.sidebar.multiselect(
    "Vehicle Type(s)",
    options=all_vehicles,
    default=[]
)

# 4. Map Type Select
map_layer_type = st.sidebar.selectbox(
    "Map Visualization Type",
    options=["Scatterplot (Individual Violations)", "3D Hexagon (Congestion Height)"],
    index=0
)

# --- Apply Filters ---
df_filtered = df_raw.copy()

if selected_stations:
    df_filtered = df_filtered[df_filtered["police_station"].isin(selected_stations)]
    
df_filtered = df_filtered[(df_filtered["hour"] >= hour_range[0]) & (df_filtered["hour"] <= hour_range[1])]

if selected_days:
    df_filtered = df_filtered[df_filtered["day_of_week"].isin(selected_days)]

if selected_vehicles:
    df_filtered = df_filtered[df_filtered["vehicle_type"].isin(selected_vehicles)]

# Check if data is empty after filtering
if df_filtered.empty:
    st.warning("No records match the current sidebar filter settings. Please adjust your selections.")
    st.stop()

# --- Dashboard Header ---
st.title("🚗 ParkWatch AI: Congestion Intelligence")
st.subheader("Bengaluru Parking-Induced Congestion Analysis Portal")
st.markdown("Quantifying real-world lane capacity reductions, delay multipliers, and physical blockage loads across time, place, and violations.")

st.markdown("---")

# --- Row 1: KPI Cards ---
kpi_cols = st.columns(4)

total_violations = len(df_filtered)
avg_score = df_filtered["traffic_score"].mean()
total_pcu_h = df_filtered["pcu"].sum()
peak_pct = df_filtered["is_peak"].mean() * 100

# Compute physical capacity loss estimates
# Standard urban lane capacity is ~1800 PCU/h. We estimate effective average lane capacity drop
avg_capacity_loss = min(15.0 + 35.0 * (df_filtered["pcu"].mean() / 3.0) + 30.0 * df_filtered["is_junction"].mean(), 85.0)
avg_delay_multiplier = 1.0 + 0.15 * (1.0 / (1.0 - (avg_capacity_loss / 100.0)))**2

with kpi_cols[0]:
    st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Active Violations</span>
            <div class="kpi-value" style="color: #818cf8;">{total_violations:,}</div>
            <span class="kpi-desc">Total parking blockages detected</span>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Average Traffic Score</span>
            <div class="kpi-value" style="color: #fb923c;">{avg_score:.2f}</div>
            <span class="kpi-desc">Congestion severity index (PCU × peak × junction)</span>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Total Blockage Load</span>
            <div class="kpi-value" style="color: #38bdf8;">{total_pcu_h:,.1f}</div>
            <span class="kpi-desc">Passenger Car Units (PCU) equivalent</span>
        </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
        <div class="kpi-card">
            <span class="kpi-label">Est. Delay Multiplier</span>
            <div class="kpi-value" style="color: #f87171;">{avg_delay_multiplier:.2f}x</div>
            <span class="kpi-desc">Travel time delay multiplier (BPR index)</span>
        </div>
    """, unsafe_allow_html=True)

# --- Row 2: Map & Spatial Distribution ---
st.markdown('<div class="section-header">Spatial Congestion Mapping (Place Analysis)</div>', unsafe_allow_html=True)
map_cols = st.columns([7, 3])

with map_cols[0]:
    st.markdown("### Interactive Congestion Map")
    
    # Calculate map viewport center
    center_lat = float(df_filtered["latitude"].mean())
    center_lon = float(df_filtered["longitude"].mean())
    
    # PyDeck Map setup
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11.5,
        pitch=45 if map_layer_type.startswith("3D") else 0,
        bearing=15 if map_layer_type.startswith("3D") else 0
    )
    
    if map_layer_type.startswith("Scatter"):
        # Map traffic scores to color lists
        df_filtered["color_r"] = np.select([df_filtered["traffic_score"] >= 6.0, df_filtered["traffic_score"] >= 4.0, df_filtered["traffic_score"] >= 2.0], [239, 249, 234], default=6)
        df_filtered["color_g"] = np.select([df_filtered["traffic_score"] >= 6.0, df_filtered["traffic_score"] >= 4.0, df_filtered["traffic_score"] >= 2.0], [68, 115, 179], default=182)
        df_filtered["color_b"] = np.select([df_filtered["traffic_score"] >= 6.0, df_filtered["traffic_score"] >= 4.0, df_filtered["traffic_score"] >= 2.0], [68, 22, 8], default=212)
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position="[longitude, latitude]",
            get_fill_color="[color_r, color_g, color_b, 160]",
            get_radius="pcu * 15",
            pickable=True,
            auto_highlight=True,
        )
        
        tooltip = {
            "html": "<b>Location:</b> {location}<br/>"
                    "<b>Junction:</b> {junction_name}<br/>"
                    "<b>Vehicle:</b> {vehicle_type}<br/>"
                    "<b>PCU Weight:</b> {pcu}<br/>"
                    "<b>Traffic Score:</b> {traffic_score}",
            "style": {"backgroundColor": "#111827", "color": "#f3f4f6", "borderColor": "#1f2937", "borderWidth": "1px", "fontSize": "11px"}
        }
    else:
        # 3D Hexagon aggregation layer
        layer = pdk.Layer(
            "HexagonLayer",
            data=df_filtered,
            get_position="[longitude, latitude]",
            radius=200,
            elevation_scale=40,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            auto_highlight=True,
            color_range=[
                [6, 182, 212],
                [234, 179, 8],
                [249, 115, 22],
                [239, 68, 68]
            ]
        )
        
        tooltip = {
            "html": "<b>Congestion Node</b><br/>"
                    "Count: {count}<br/>"
                    "Density Level: High",
            "style": {"backgroundColor": "#111827", "color": "#f3f4f6", "borderColor": "#1f2937", "borderWidth": "1px", "fontSize": "11px"}
        }
        
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip
    )
    
    st.pydeck_chart(r)

with map_cols[1]:
    st.markdown("### Top Congested Locations")
    
    # 1. Top Junctions
    df_junc = df_filtered[df_filtered["junction_name"].notna() & (~df_filtered["junction_name"].str.upper().str.strip().isin(["", "NO JUNCTION", "NULL"]))]
    if not df_junc.empty:
        top_juncs = df_junc.groupby("junction_name").agg(
            violations=("id", "count"),
            avg_traffic_score=("traffic_score", "mean")
        ).sort_values("avg_traffic_score", ascending=False).head(5)
        
        st.markdown("**Top Junction Bottlenecks (Mean Score)**")
        for j_name, row in top_juncs.iterrows():
            st.caption(f"📍 **{j_name}**")
            st.progress(min(float(row["avg_traffic_score"] / 9.0), 1.0))
            st.markdown(f"<span style='font-size: 0.8rem; color:#9ca3af;'>Traffic Score: <b>{row['avg_traffic_score']:.2f}</b> | Violations: <b>{int(row['violations'])}</b></span>", unsafe_allow_html=True)
    else:
        st.info("No junction-specific congestion data found.")
        
    # 2. Top Police Stations
    st.markdown("---")
    st.markdown("**Most Congested Police Stations**")
    top_stations = df_filtered.groupby("police_station").agg(
        violations=("id", "count"),
        avg_traffic_score=("traffic_score", "mean")
    ).sort_values("violations", ascending=False).head(5)
    
    for s_name, row in top_stations.iterrows():
        st.markdown(f"👮 **{s_name}**")
        st.markdown(f"<span style='font-size: 0.8rem; color:#9ca3af;'>Average Traffic Score: <b style='color:#fb923c;'>{row['avg_traffic_score']:.2f}</b> | Blockages: <b>{int(row['violations']):,}</b></span>", unsafe_allow_html=True)

# --- Row 3: Temporal Trends ---
st.markdown('<div class="section-header">Temporal Patterns (Time Analysis)</div>', unsafe_allow_html=True)
trend_cols = st.columns(2)

with trend_cols[0]:
    st.markdown("### Diurnal Congestion Cycle (Hour of Day)")
    # Group by hour
    df_hour = df_filtered.groupby("hour").agg(
        violations=("id", "count"),
        avg_traffic_score=("traffic_score", "mean")
    ).reset_index()
    
    # Plotly Line Chart for Hour of Day
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=df_hour["hour"], 
        y=df_hour["violations"],
        name="Violation Count",
        line=dict(color="#818cf8", width=3),
        yaxis="y1"
    ))
    fig_hour.add_trace(go.Scatter(
        x=df_hour["hour"], 
        y=df_hour["avg_traffic_score"],
        name="Average Traffic Score",
        line=dict(color="#fb923c", width=3, dash="dot"),
        yaxis="y2"
    ))
    
    fig_hour.update_layout(
        xaxis=dict(title="Hour of Day (IST)", tickmode="linear", tick0=0, dtick=2, gridcolor="#1f2937"),
        yaxis=dict(title="Number of Violations", titlefont=dict(color="#818cf8"), tickfont=dict(color="#818cf8")),
        yaxis2=dict(title="Traffic Score", titlefont=dict(color="#fb923c"), tickfont=dict(color="#fb923c"), anchor="x", overlaying="y", side="right"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with trend_cols[1]:
    st.markdown("### Congestion by Day of the Week")
    # Group by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_day = df_filtered.groupby("day_of_week").agg(
        violations=("id", "count"),
        avg_traffic_score=("traffic_score", "mean")
    ).reindex(day_order).reset_index()
    
    # Plotly Bar Chart for Day of Week
    fig_day = px.bar(
        df_day,
        x="day_of_week",
        y="avg_traffic_score",
        color="avg_traffic_score",
        color_continuous_scale=["#6366f1", "#a855f7", "#ec4899"],
        labels={"day_of_week": "Day of Week", "avg_traffic_score": "Average Traffic Score"}
    )
    fig_day.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=320
    )
    st.plotly_chart(fig_day, use_container_width=True)

# --- Row 4: Violation & Vehicle Type Analysis ---
st.markdown('<div class="section-header">Vehicle & Offender Profiles (Violation Analysis)</div>', unsafe_allow_html=True)
viol_cols = st.columns(2)

with viol_cols[0]:
    st.markdown("### Congestion Share by Vehicle Type")
    # Group by vehicle type
    df_veh = df_filtered.groupby("vehicle_type").agg(
        violations=("id", "count"),
        avg_traffic_score=("traffic_score", "mean")
    ).sort_values("violations", ascending=False).head(10).reset_index()
    
    fig_veh = px.bar(
        df_veh,
        x="violations",
        y="vehicle_type",
        color="avg_traffic_score",
        orientation="h",
        color_continuous_scale=["#38bdf8", "#818cf8", "#f43f5e"],
        labels={"vehicle_type": "Vehicle Type", "violations": "Violations Count", "avg_traffic_score": "Traffic Score"}
    )
    fig_veh.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=True,
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(autorange="reversed", gridcolor="#1f2937"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320
    )
    st.plotly_chart(fig_veh, use_container_width=True)

with viol_cols[1]:
    st.markdown("### Repeat Offenders Leaderboard")
    
    # Identify vehicles with most violations
    repeat_offenders = df_filtered.groupby("vehicle_number").agg(
        violations=("id", "count"),
        cumulative_traffic_score=("traffic_score", "sum"),
        police_stations=("police_station", lambda x: ", ".join(x.dropna().unique()[:2]))
    ).sort_values("violations", ascending=False).head(10).reset_index()
    
    # Display table of chronic offenders
    st.dataframe(
        repeat_offenders.rename(columns={
            "vehicle_number": "Vehicle License Plate",
            "violations": "Blockage Count",
            "cumulative_traffic_score": "Total Traffic Impact",
            "police_stations": "Primary Zones"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("Vehicles with high violation counts heavily reduce carriageway throughput. Targeted enforcement in these primary zones is recommended.")
