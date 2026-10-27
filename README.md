# Cloud-Native Spatial Intelligence Platform

An enterprise-grade, cloud-native spatial analytics platform focused on the South Iraqi Marshes, streaming Sentinel-2 L2A Earth Observation data in-memory.

## 📸 Platform Highlights

### Interactive Spatial Mapping
![Map View](map-view.png)

### Temporal Vegetation Dynamics
![Time Series Profile](time-series.png)

## 🚀 Architecture & Features
- **Cloud-Native In-Memory Streaming:** Direct STAC API queries against Microsoft Planetary Computer (COG format).
- **Spectral Masking & Index Computation:** SCL-based atmospheric cloud masking and NDVI index calculation.
- **Equal-Area Spatial Analytics:** Resampling and reprojection to `EPSG:6933` for precise area quantification ($km^2$).
- **Multi-Temporal Change Detection:** Dual-period baseline comparisons with net vegetation change metrics.
- **Time-Series Dynamic Profiling:** Multi-date spectral indexing powered by interactive Plotly visualizations.
- **Enterprise Reporting:** One-click automated exports for GeoJSON vector features and CSV executive summaries.

## 🛠️ Local Execution
```bash
pip install -r requirements.txt
streamlit run app.py