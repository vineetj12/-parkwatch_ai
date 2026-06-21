# ParkWatch AI: Bengaluru Parking Congestion Analytics

ParkWatch AI is an intelligent spatial analytics system designed to identify illegal on-street parking hotspots in Bengaluru, quantify their direct impact on traffic flow, and prioritize them for targeted police enforcement.

It processes a raw database of traffic violations (~298k entries), runs spatial clustering, models capacity reduction and queue delays, and renders a high-fidelity visual dashboard.

---

## 📂 Project Structure

* **[pipeline.py](file:///c:/Users/HP/Downloads/problem1(incmplete)/problem1/pipeline.py)**: The Python data pipeline that cleans telemetry, runs spatial DBSCAN clustering, calculates the Congestion Impact Score (CIS), models traffic disruption, and exports `parkwatch_data.json`.
* **[index.html](file:///c:/Users/HP/Downloads/problem1(incmplete)/problem1/index.html)**: A premium, dark-themed single-page web dashboard using Leaflet.js and Chart.js to map hotspots, list leaderboard metrics, and visualize diurnal congestion heat grids.
* **[app.py](file:///c:/Users/HP/Downloads/problem1(incmplete)/problem1/app.py)**: An alternative Streamlit dashboard utilizing pydeck and Plotly.
* **[solution_design.txt](file:///c:/Users/HP/Downloads/problem1(incmplete)/problem1/solution_design.txt)**: Comprehensive mathematical design of the clustering parameters, sub-scores, and traffic capacity models.

---

## ⚙️ Methodology

The system uses advanced spatial clustering and traffic engineering models to analyze parking violations:

### 1. Spatial Hotspot Clustering
* **Algorithm**: **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).
* **Metric**: **Haversine Distance** (radians) to compute exact distances on the Earth's curved surface.
* **Epsilon ($\epsilon$) = 150m**: Represents the walking buffer and physical span of parking-induced road choking.
* **MinPts = 8**: Minimum violations to form a hotspot, filtering out random or transient parking events.

### 2. Congestion Impact Score (CIS) Formulation
Calculates a composite severity score ($0–100$) per hotspot using 5 weighted sub-scores:
$$CIS = 0.30 \times S_{\text{density}} + 0.25 \times S_{\text{junction}} + 0.15 \times S_{\text{vehicle}} + 0.20 \times S_{\text{persistence}} + 0.10 \times S_{\text{peak}}$$
* **Density ($S_{\text{density}}$, 30%)**: Log-scaled violations per $\text{km}^2$ to handle highly skewed ranges.
* **Junction Proximity ($S_{\text{junction}}$, 25%)**: Percentage of violations located at major road intersections.
* **Vehicle Weight ($S_{\text{vehicle}}$, 15%)**: Mapped based on physical footprint using standard Passenger Car Units (PCUs) (e.g., Bus/Truck = 3.0, Car = 1.2, Two-wheeler = 0.6).
* **Persistence ($S_{\text{persistence}}$, 20%)**: Ratio of active days to total historical span.
* **Peak Overlap ($S_{\text{peak}}$, 10%)**: Percentage of violations occurring during morning/evening rush hours.

### 3. Traffic Disruption Modelling
We convert the relative CIS risk score into physical traffic engineering metrics:
* **Lane Capacity Reduction (LCR%)**: Estimates the effective capacity loss of the road lane:
  $$LCR = \min(15.0 + 35.0 \times \frac{\text{Avg. PCU}}{3.0} + 30.0 \times \text{Junction Proximity}, 85.0)$$
* **Travel Time Delay Multiplier**: Models traffic queue delays using the Bureau of Public Roads (BPR) equation:
  $$\text{Delay Multiplier} = 1.0 + 0.15 \times \left(\frac{1.0}{1.0 - \text{LCR \%}}\right)^2$$
* **Daily PCU Blockage Load**: Combines vehicle sizes, daily persistence, and peak hour overlap to estimate aggregate daily traffic load.

---

## 🔄 Operational Workflow

The system operates via a 4-phase data and operational workflow:

```
[ Raw CSV Dataset ]
         │
         ▼  (Phase 1: Preprocessing & Cleaning)
[ Normalise NULLs, Convert to IST, Filter Parking Keywords, Geofence BBox ]
         │
         ▼  (Phase 2: Spatial & Statistical Modelling)
[ Run DBSCAN ──► Compute CIS ──► Quantify Traffic Blockage & Delay Multipliers ]
         │
         ▼  (Phase 3: Serialization & Export)
[ Generate Repeat Offenders (Count >= 3) & Weekday/Hourly Congestion Heatmap ]
         │
         ▼  (Phase 4: Visual Dashboard & Patrol Dispatch)
[ Serve Local Web Page ──► Filter by Station/Severity ──► Deploy Targeted Patrols ]
```

1. **Preprocessing & Filtering**: Cleans strings, localizes timestamps to IST (UTC+5:30), extracts only parking-specific violations, and clamps coordinates to the Bengaluru geofence.
2. **Clustering & Analytical Modeling**: Feeds spatial points to DBSCAN, computes the normalized CIS scores, and runs traffic blockage quantification.
3. **Temporal Heat Grid & Offender Synthesis**: Compiles a joint Day-of-Week x Hour-of-Day congestion matrix using the entire dataset, and maps repeat offenders ($\ge 3$ violations in the same hotspot).
4. **Interactive Dashboard & Enforcement Dispatch**: Serves the web interface locally. Patrol commanders search/filter by police station and severity to dispatch towing vehicles to the most critical chokepoints during peak hours.

---

## 🚀 Setup & Execution

### 1. Requirements
Install the necessary Python packages:
```bash
pip install pandas numpy scikit-learn
```

### 2. Run the Pipeline
Compile the raw dataset (`dataset.csv`) and output `parkwatch_data.json`:
```bash
python pipeline.py
```
*Note: Optimization changes (vectorization) have reduced this run from ~10 minutes to under 25 seconds.*

### 3. Serve the Web Dashboard
Since the dashboard uses Javascript `fetch()` to load the processed JSON, browsers block local access via CORS when opening `file://` directly. You must serve it via a local HTTP server:
```bash
python -m http.server 8000
```
Then navigate to: **[http://localhost:8000/](http://localhost:8000/)** in your browser.

### 4. Run the Streamlit Dashboard (Alternative)
```bash
streamlit run app.py
```

# -parkwatch_ai
