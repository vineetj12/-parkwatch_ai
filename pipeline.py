"""
ParkWatch AI — Parking Intelligence Pipeline
============================================
Processes Bengaluru traffic violation data to:
1. Detect illegal parking hotspots using DBSCAN spatial clustering
2. Compute a Congestion Impact Score (CIS) per hotspot
3. Export enriched analytics as JSON for the web dashboard
"""

import ast
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = Path("dataset.csv")
OUT_DIR = Path(".")

EARTH_RADIUS_KM = 6371.0
CLUSTER_EPS_M = 150          # metres — DBSCAN neighbourhood radius
CLUSTER_MIN_SAMPLES = 8      # minimum points to form a hotspot
PEAK_HOURS = list(range(7, 11)) + list(range(16, 21))  # 7-10 AM, 4-8 PM

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

# Parking-related violation keywords
PARKING_KEYWORDS = [
    "PARKING", "NO PARKING", "WRONG PARKING",
    "PARKING IN A MAIN ROAD", "PARKING NEAR ROAD CROSSING",
    "PARKING ON FOOTPATH", "PARKING IN NO PARKING ZONE",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_list_cell(value):
    """Parse a JSON-like list string like '["NO PARKING","WRONG PARKING"]'."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return [str(value)]


def to_radians(degrees):
    return degrees * math.pi / 180


def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in km between two lat/lng points."""
    rlat1, rlon1, rlat2, rlon2 = map(to_radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_km_vectorized(lat_c, lon_c, lats, lons):
    """Vectorized haversine: distance in km from centroid to each point in arrays lats/lons."""
    rlat_c = np.radians(lat_c)
    rlon_c = np.radians(lon_c)
    rlats = np.radians(lats)
    rlons = np.radians(lons)
    dlat = rlats - rlat_c
    dlon = rlons - rlon_c
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat_c) * np.cos(rlats) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def safe_json(obj):
    """Convert numpy/pandas types to native Python for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ── Step 1: Load & Clean ─────────────────────────────────────────────────────

print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw rows: {len(df):,} | Columns: {df.shape[1]}")

# Normalise NULL strings
df.replace({"NULL": np.nan, "null": np.nan, "": np.nan}, inplace=True)

# Parse datetime
df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)

# Parse violation_type lists
df["violation_type_list"] = df["violation_type"].apply(parse_list_cell)

# Temporal features (IST = UTC+5:30)
df["created_ist"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
df["hour"] = df["created_ist"].dt.hour
df["day_of_week"] = df["created_ist"].dt.day_name()
df["month_str"] = df["created_ist"].dt.to_period("M").astype(str)
df["date_str"] = df["created_ist"].dt.date.astype(str)

# Filter to parking-related violations only
def is_parking(vlist):
    for v in vlist:
        if any(kw in str(v).upper() for kw in PARKING_KEYWORDS):
            return True
    return False

df["is_parking"] = df["violation_type_list"].apply(is_parking)
df_park = df[df["is_parking"]].copy()
df_park.dropna(subset=["latitude", "longitude"], inplace=True)

# Clamp to Bengaluru bbox (sanity check)
df_park = df_park[
    (df_park["latitude"].between(12.8, 13.35)) &
    (df_park["longitude"].between(77.4, 77.8))
]

print(f"  Parking violations (Bengaluru bbox): {len(df_park):,}")

# ── Step 2: MiniBatchKMeans Spatial Clustering ────────────────────────────────────────

print("Running MiniBatchKMeans spatial clustering ...")
coords = df_park[["latitude", "longitude"]].values
kmeans = MiniBatchKMeans(n_clusters=200, random_state=42, batch_size=2048)
df_park["cluster"] = kmeans.fit_predict(coords)

n_clusters = df_park["cluster"].nunique()
n_noise = 0
print(f"  Clusters found: {n_clusters}")

df_clustered = df_park.copy()

# ── Step 3: Compute Cluster-Level Features ────────────────────────────────────

print("Computing cluster-level features ...")

def vehicle_score(vtype_series):
    """Average congestion weight for vehicle types in cluster."""
    weights = vtype_series.map(lambda v: VEHICLE_WEIGHTS.get(str(v).upper().strip(), 1.0))
    return weights.mean()

def dominant_junction(jname_series):
    """Return the most common non-trivial junction name in the cluster."""
    counts = jname_series[~jname_series.isin(["No Junction", np.nan, None])].value_counts()
    return counts.index[0] if len(counts) > 0 else "No Junction"

clusters_data = []

for cid, grp in df_clustered.groupby("cluster"):
    lat_c = grp["latitude"].mean()
    lon_c = grp["longitude"].mean()
    n = len(grp)

    # --- CIS sub-scores ---
    # A) Violation Density: violations per unit area (π·r² km²)
    distances_km = haversine_km_vectorized(
        lat_c, lon_c,
        grp["latitude"].values,
        grp["longitude"].values
    )
    # Use 90th percentile distance to represent the core hotspot area and prevent outlier stretching
    radius_km = max(float(np.percentile(distances_km, 90)), 0.05)
    area_km2 = math.pi * radius_km ** 2
    density = n / area_km2

    # B) Junction Proximity: fraction near a real junction
    near_junction = (
        grp["junction_name"].notna() &
        ~grp["junction_name"].isin(["No Junction"])
    ).mean()

    # C) Vehicle Size Factor: average congestion weight (normalised to 0-1 within 0.3-3.0 range)
    veh_w = vehicle_score(grp["vehicle_type"])
    veh_factor = (veh_w - 0.3) / (3.0 - 0.3)

    # D) Temporal Persistence: distinct active days / total date range
    n_days = grp["date_str"].nunique()
    date_range = max(
        (grp["created_ist"].max() - grp["created_ist"].min()).days + 1, 1
    )
    persistence = min(n_days / date_range, 1.0)

    # E) Peak Hour Overlap: fraction of violations during peak hours
    peak_frac = grp["hour"].isin(PEAK_HOURS).mean()

    # --- Hourly + Day distribution ---
    hourly = grp["hour"].value_counts().sort_index().to_dict()
    daily = grp["day_of_week"].value_counts().to_dict()
    monthly = grp["month_str"].value_counts().sort_index().to_dict()

    # Violation type mix (flattened)
    all_viol = [v for lst in grp["violation_type_list"] for v in lst]
    viol_counts = {}
    for v in all_viol:
        viol_counts[str(v)] = viol_counts.get(str(v), 0) + 1
    top_violations = sorted(viol_counts.items(), key=lambda x: -x[1])[:5]

    # Vehicle type mix
    veh_counts = grp["vehicle_type"].value_counts().head(5).to_dict()

    # Police station
    top_station = grp["police_station"].value_counts().idxmax() if grp["police_station"].notna().any() else "Unknown"
    junc = dominant_junction(grp["junction_name"])
    near_junc_pct = round(near_junction * 100, 1)

    # Location text (most common location prefix)
    loc_sample = grp["location"].dropna().head(1).values
    loc_text = loc_sample[0][:80] if len(loc_sample) > 0 else "Unknown"

    # Top active hours
    top_hours = sorted(hourly.items(), key=lambda x: -x[1])[:3]
    top_hours_str = ", ".join(f"{int(h):02d}:00" for h, _ in top_hours)

    # Date range strings
    earliest = grp["created_ist"].min().strftime("%Y-%m-%d") if pd.notna(grp["created_ist"].min()) else "N/A"
    latest = grp["created_ist"].max().strftime("%Y-%m-%d") if pd.notna(grp["created_ist"].max()) else "N/A"

    # Traffic Quantification Models
    total_pcu = float(n * veh_w)
    lcr = min(15.0 + 35.0 * (veh_w / 3.0) + 30.0 * near_junction, 85.0)
    delay_multiplier = 1.0 + 0.15 * (1.0 / (1.0 - (lcr / 100.0)))**2
    daily_pcu_load = (total_pcu / n_days) * (0.5 + 1.5 * peak_frac)

    clusters_data.append({
        "cluster_id": int(cid),
        "centroid_lat": round(lat_c, 6),
        "centroid_lon": round(lon_c, 6),
        "violation_count": int(n),
        "radius_km": round(radius_km, 4),
        "area_km2": round(area_km2, 6),
        "density": round(density, 2),
        "near_junction_pct": near_junc_pct,
        "vehicle_congestion_weight": round(veh_w, 3),
        "temporal_persistence": round(persistence, 3),
        "peak_hour_fraction": round(peak_frac, 3),
        "junction_name": junc,
        "police_station": str(top_station),
        "location_sample": loc_text,
        "top_violations": [{"type": t, "count": c} for t, c in top_violations],
        "vehicle_mix": {str(k): int(v) for k, v in veh_counts.items()},
        "hourly_distribution": {str(k): int(v) for k, v in hourly.items()},
        "daily_distribution": {str(k): int(v) for k, v in daily.items()},
        "monthly_distribution": {str(k): int(v) for k, v in monthly.items()},
        "peak_hours": top_hours_str,
        "active_days": int(n_days),
        "earliest_violation": earliest,
        "latest_violation": latest,
        "total_pcu_blockage": round(total_pcu, 1),
        "lane_capacity_reduction_pct": round(lcr, 1),
        "traffic_delay_multiplier": round(delay_multiplier, 2),
        "daily_pcu_blockage_load": round(daily_pcu_load, 1),
        # raw sub-scores stored for transparency
        "_sub_density": round(density, 4),
        "_sub_junction": round(near_junction, 4),
        "_sub_vehicle": round(veh_factor, 4),
        "_sub_persistence": round(persistence, 4),
        "_sub_peak": round(peak_frac, 4),
    })

df_clusters = pd.DataFrame(clusters_data)
print(f"  Cluster features computed for {len(df_clusters)} hotspots")

# ── Step 4: Normalise & Compute CIS ──────────────────────────────────────────

print("Computing Congestion Impact Score (CIS) ...")

scaler = MinMaxScaler()

# Normalise density (log scale first to reduce skew)
df_clusters["density_norm"] = scaler.fit_transform(
    np.log1p(df_clusters["density"].values).reshape(-1, 1)
)
df_clusters["junction_norm"] = df_clusters["_sub_junction"]
df_clusters["vehicle_norm"] = df_clusters["_sub_vehicle"]
df_clusters["persistence_norm"] = df_clusters["_sub_persistence"]
df_clusters["peak_norm"] = df_clusters["_sub_peak"]

# Weighted composite CIS (weights sum to 1)
WEIGHTS = {
    "density": 0.30,
    "junction": 0.25,
    "vehicle": 0.15,
    "persistence": 0.20,
    "peak": 0.10,
}

df_clusters["cis_raw"] = (
    WEIGHTS["density"] * df_clusters["density_norm"] +
    WEIGHTS["junction"] * df_clusters["junction_norm"] +
    WEIGHTS["vehicle"] * df_clusters["vehicle_norm"] +
    WEIGHTS["persistence"] * df_clusters["persistence_norm"] +
    WEIGHTS["peak"] * df_clusters["peak_norm"]
)

# Scale CIS to 0-100
df_clusters["cis"] = (df_clusters["cis_raw"] / df_clusters["cis_raw"].max() * 100).round(1)

# Severity tier
def severity(score):
    if score >= 70: return "CRITICAL"
    if score >= 45: return "HIGH"
    if score >= 25: return "MEDIUM"
    return "LOW"

df_clusters["severity"] = df_clusters["cis"].apply(severity)
df_clusters.sort_values("cis", ascending=False, inplace=True)
df_clusters["rank"] = range(1, len(df_clusters) + 1)

print(f"  CIS range: {df_clusters['cis'].min():.1f} - {df_clusters['cis'].max():.1f}")
print(f"  Severity breakdown:\n{df_clusters['severity'].value_counts().to_string()}")

# ── Step 5: Global Temporal Analytics ────────────────────────────────────────

print("Building global temporal analytics ...")

# Hourly global
hourly_global = df_park["hour"].value_counts().sort_index()
hourly_global_dict = {str(int(h)): int(c) for h, c in hourly_global.items()}

# Day-of-week global
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_global = df_park["day_of_week"].value_counts().reindex(dow_order, fill_value=0)
dow_global_dict = {str(k): int(v) for k, v in dow_global.items()}

# Monthly global
monthly_global = df_park["month_str"].value_counts().sort_index()
monthly_global_dict = {str(k): int(v) for k, v in monthly_global.items()}

# Violation type global (top 20)
all_viol_global = [v for lst in df_park["violation_type_list"] for v in lst]
viol_global = {}
for v in all_viol_global:
    viol_global[str(v)] = viol_global.get(str(v), 0) + 1
top_viol_global = sorted(viol_global.items(), key=lambda x: -x[1])[:20]

# Vehicle type global
veh_global = df_park["vehicle_type"].value_counts().head(15).to_dict()

# Per-police-station monthly trend
station_monthly = (
    df_park.groupby(["police_station", "month_str"])
    .size()
    .unstack(fill_value=0)
    .to_dict(orient="index")
)

# Calculate congestion (PCU blockage load) by day-of-week and hour using the entire dataset
df_park["pcu_weight"] = df_park["vehicle_type"].map(lambda v: VEHICLE_WEIGHTS.get(str(v).upper().strip(), 1.0) if pd.notna(v) else 1.0)
heat_matrix = df_park.groupby(["day_of_week", "hour"])["pcu_weight"].sum().unstack(fill_value=0.0)
heat_matrix = heat_matrix.reindex(dow_order, fill_value=0.0)
congestion_heatmap = []
for day in dow_order:
    row_vals = []
    for h in range(24):
        val = float(heat_matrix.loc[day, h]) if h in heat_matrix.columns else 0.0
        row_vals.append(round(val, 1))
    congestion_heatmap.append(row_vals)

timeseries = {
    "hourly_global": hourly_global_dict,
    "dow_global": dow_global_dict,
    "monthly_global": monthly_global_dict,
    "congestion_heatmap": congestion_heatmap,
    "top_violation_types": [{"type": t, "count": c} for t, c in top_viol_global],
    "vehicle_type_distribution": {str(k): int(v) for k, v in veh_global.items()},
    "station_monthly_trend": {
        str(k): {str(m): int(v) for m, v in vd.items()}
        for k, vd in station_monthly.items()
    },
}

# ── Step 6: Repeat Offenders ──────────────────────────────────────────────────

print("Identifying repeat offenders ...")

# Tag each violation with its cluster
df_park_c = df_clustered.copy()
repeat = (
    df_park_c.groupby(["vehicle_number", "cluster"])
    .agg(
        count=("id", "count"),
        violation_types=("violation_type_list", lambda x: list({v for lst in x for v in lst})),
        police_station=("police_station", lambda x: x.mode()[0] if x.notna().any() else "Unknown"),
    )
    .reset_index()
)
repeat = repeat[repeat["count"] >= 3].sort_values("count", ascending=False)
repeat_offenders = repeat.head(100).to_dict(orient="records")
for r in repeat_offenders:
    r["violation_types"] = [str(v) for v in r["violation_types"]]
    r["cluster"] = int(r["cluster"])
    r["count"] = int(r["count"])

# ── Step 7: Export JSON ──────────────────────────────────────────────────────

print("Exporting JSON files ...")

# Build hotspot export list (drop internal _sub_ cols)
hotspot_cols = [
    "rank", "cluster_id", "centroid_lat", "centroid_lon",
    "cis", "severity", "violation_count", "radius_km",
    "near_junction_pct", "vehicle_congestion_weight",
    "temporal_persistence", "peak_hour_fraction",
    "junction_name", "police_station", "location_sample",
    "top_violations", "vehicle_mix",
    "hourly_distribution", "daily_distribution", "monthly_distribution",
    "peak_hours", "active_days", "earliest_violation", "latest_violation",
    "total_pcu_blockage", "lane_capacity_reduction_pct", "traffic_delay_multiplier", "daily_pcu_blockage_load",
    "density_norm", "junction_norm", "vehicle_norm", "persistence_norm", "peak_norm",
]
hotspots_records = df_clusters[hotspot_cols].to_dict(orient="records")

# Summary KPIs
total_violations = int(len(df_park))
total_hotspots = int(len(df_clusters))
avg_cis = round(df_clusters["cis"].mean(), 1)
peak_hour_global = int(hourly_global.idxmax())
critical_count = int((df_clusters["severity"] == "CRITICAL").sum())
high_count = int((df_clusters["severity"] == "HIGH").sum())

summary = {
    "total_parking_violations": total_violations,
    "total_hotspots": total_hotspots,
    "average_cis": avg_cis,
    "peak_hour": peak_hour_global,
    "critical_zones": critical_count,
    "high_zones": high_count,
    "cluster_eps_m": CLUSTER_EPS_M,
    "cluster_min_samples": CLUSTER_MIN_SAMPLES,
}

output = {
    "summary": summary,
    "hotspots": hotspots_records,
    "timeseries": timeseries,
    "repeat_offenders": repeat_offenders,
}

out_path = OUT_DIR / "parkwatch_data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, default=safe_json, ensure_ascii=False, separators=(",", ":"))

size_mb = out_path.stat().st_size / 1024 / 1024
print(f"\n[OK] Exported: {out_path} ({size_mb:.2f} MB)")
print(f"\n{'='*55}")
print(f"  ParkWatch AI - Pipeline Complete")
print(f"  Total parking violations : {total_violations:>10,}")
print(f"  Hotspots detected        : {total_hotspots:>10,}")
print(f"  Critical zones           : {critical_count:>10,}")
print(f"  High-risk zones          : {high_count:>10,}")
print(f"  Average CIS              : {avg_cis:>10.1f}")
print(f"  Peak violation hour      : {peak_hour_global:>9}:00")
print(f"{'='*55}")
