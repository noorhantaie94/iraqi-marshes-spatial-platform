from datetime import date, timedelta
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import plotly.express as px
import plotly.graph_objects as go
import pystac_client
import rasterio
import rasterio.features
from rasterio.transform import from_bounds
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. Page Configuration & Enterprise Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cloud-Native Spatial Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: bold; color: #2e7d32; }
        div[data-testid="stMetric"] {
            background-color: #1e242b;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #30363d;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Cached Core Spatial Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_process_scene(bbox, date_start, date_end, max_cloud, ndvi_thresh):
    """Queries STAC API, streams Sentinel-2 bands, applies SCL masking, and vectorizes."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    date_range = f"{date_start}/{date_end}"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    if not items:
        return None, None, None, f"No clear scenes found between {date_start} and {date_end}."

    items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    scene = items[0]

    with (
        rasterio.open(scene.assets["B04"].href) as r_src,
        rasterio.open(scene.assets["B08"].href) as n_src,
        rasterio.open(scene.assets["SCL"].href) as s_src,
    ):
        red = r_src.read(1, out_shape=(256, 256)).astype(np.float32)
        nir = n_src.read(1, out_shape=(256, 256)).astype(np.float32)
        scl = s_src.read(1, out_shape=(256, 256))

    denom = nir + red
    denom[denom == 0] = np.nan
    ndvi = (nir - red) / denom

    cloud_mask = np.isin(scl, [3, 8, 9, 10])
    ndvi[cloud_mask] = np.nan

    veg_binary = (ndvi >= ndvi_thresh).astype(np.uint8)
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], 256, 256)
    shapes = rasterio.features.shapes(veg_binary, transform=transform)

    polygons = [
        {"properties": {"val": val}, "geometry": geom}
        for geom, val in shapes
        if val == 1
    ]

    if not polygons:
        return ndvi, scene, None, "No vegetation features detected for this period."

    gdf = gpd.GeoDataFrame.from_features(polygons, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=6933)
    gdf["area_km2"] = gdf_proj.geometry.area / 1e6

    return ndvi, scene, gdf, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_time_series(bbox, date_start, date_end, max_cloud):
    """Queries STAC API across a time range and streams mean NDVI per scene."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    if not items:
        return None

    # Sort chronologically and sample to prevent request rate-limiting
    items = sorted(items, key=lambda x: x.datetime)
    sampled_items = items[:: max(1, len(items) // 10)]

    records = []
    for item in sampled_items:
        try:
            with (
                rasterio.open(item.assets["B04"].href) as r_src,
                rasterio.open(item.assets["B08"].href) as n_src,
                rasterio.open(item.assets["SCL"].href) as s_src,
            ):
                red = r_src.read(1, out_shape=(128, 128)).astype(np.float32)
                nir = n_src.read(1, out_shape=(128, 128)).astype(np.float32)
                scl = s_src.read(1, out_shape=(128, 128))

            denom = nir + red
            denom[denom == 0] = np.nan
            ndvi = (nir - red) / denom
            cloud_mask = np.isin(scl, [3, 8, 9, 10])
            ndvi[cloud_mask] = np.nan

            records.append({
                "Date": item.datetime.strftime("%Y-%m-%d"),
                "Mean NDVI": float(np.nanmean(ndvi)),
                "Cloud Cover (%)": float(item.properties.get("eo:cloud_cover", 0)),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 3. Sidebar Control Interface
# -----------------------------------------------------------------------------
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    options=["Single Date Analysis", "Dual Date Change Detection", "Time-Series Profile"],
)

area_choice = st.sidebar.selectbox(
    "Area of Interest (AOI):",
    options=["South Iraqi Marshes", "Custom Bounding Box"],
)

if area_choice == "South Iraqi Marshes":
    bbox_input = [46.8, 30.8, 47.3, 31.3]
else:
    min_lon = st.sidebar.number_input("Min Longitude", value=46.8)
    min_lat = st.sidebar.number_input("Min Latitude", value=30.8)
    max_lon = st.sidebar.number_input("Max Longitude", value=47.3)
    max_lat = st.sidebar.number_input("Max Latitude", value=31.3)
    bbox_input = [min_lon, min_lat, max_lon, max_lat]

st.sidebar.markdown("### Temporal Parameters")
if analysis_mode == "Single Date Analysis":
    col_d1, col_d2 = st.sidebar.columns(2)
    start_date = col_d1.date_input("Start Date", value=date.today() - timedelta(days=90))
    end_date = col_d2.date_input("End Date", value=date.today())
elif analysis_mode == "Dual Date Change Detection":
    st.sidebar.caption("Period 1 (Baseline)")
    p1_start = st.sidebar.date_input("P1 Start", value=date(2023, 6, 1))
    p1_end = st.sidebar.date_input("P1 End", value=date(2023, 8, 31))

    st.sidebar.caption("Period 2 (Comparison)")
    p2_start = st.sidebar.date_input("P2 Start", value=date(2024, 6, 1))
    p2_end = st.sidebar.date_input("P2 End", value=date(2024, 8, 31))
else:
    col_ts1, col_ts2 = st.sidebar.columns(2)
    ts_start = col_ts1.date_input("TS Start Date", value=date(2023, 1, 1))
    ts_end = col_ts2.date_input("TS End Date", value=date.today())

st.sidebar.markdown("### Sensor Thresholds")
cloud_thresh = st.sidebar.slider("Max Cloud Cover (%)", 0, 50, 15)
ndvi_threshold = st.sidebar.slider("NDVI Threshold", 0.1, 0.7, 0.35, 0.05)

run_button = st.sidebar.button("Run Spatial Analysis")


# -----------------------------------------------------------------------------
# 4. Dashboard Execution
# -----------------------------------------------------------------------------
st.title("Enterprise Spatial Intelligence Platform")
st.caption("Cloud-native spatial pipeline powered by Sentinel-2 L2A STAC API and vector analytics.")
st.markdown("---")

if run_button:
    if analysis_mode == "Single Date Analysis":
        with st.spinner("Processing satellite imagery..."):
            ndvi, scene, gdf_res, err = fetch_and_process_scene(
                bbox_input, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), cloud_thresh, ndvi_threshold
            )

        if err:
            st.warning(err)
        else:
            total_area = gdf_res["area_km2"].sum() if gdf_res is not None else 0
            mean_ndvi_val = float(np.nanmean(ndvi))
            poly_count = len(gdf_res) if gdf_res is not None else 0
            cloud_pct = float(scene.properties.get("eo:cloud_cover", 0))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Vegetation Area", f"{total_area:.2f} km²")
            m2.metric("Mean NDVI", f"{mean_ndvi_val:.3f}")
            m3.metric("Extracted Features", f"{poly_count:,}")
            m4.metric("Cloud Cover", f"{cloud_pct:.1f}%")

            tab1, tab2 = st.tabs(["Interactive Map", "Data & Reports"])

            with tab1:
                center_lat = (bbox_input[1] + bbox_input[3]) / 2
                center_lon = (bbox_input[0] + bbox_input[2]) / 2
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Esri Satellite").add_to(m)

                if gdf_res is not None:
                    folium.GeoJson(gdf_res, name="Vegetation Layer", style_function=lambda x: {"fillColor": "#2ca02c", "color": "#1b6e1b", "weight": 1.5, "fillOpacity": 0.6}).add_to(m)

                folium.LayerControl().add_to(m)
                components.html(m._repr_html_(), height=550, scrolling=False)

            with tab2:
                st.markdown("### Executive Summary Report")
                summary_data = {
                    "Metric": ["Analysis Mode", "Start Date", "End Date", "Total Area (km2)", "Mean NDVI", "Polygon Count", "Cloud Cover (%)"],
                    "Value": [analysis_mode, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), f"{total_area:.4f}", f"{mean_ndvi_val:.4f}", poly_count, f"{cloud_pct:.2f}"],
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        label="Download Executive Report (CSV)",
                        data=summary_df.to_csv(index=False),
                        file_name="executive_summary_report.csv",
                        mime="text/csv",
                    )
                with col_exp2:
                    if gdf_res is not None:
                        st.download_button(
                            label="Download Vector Layer (GeoJSON)",
                            data=gdf_res.to_json(),
                            file_name="spatial_features.geojson",
                            mime="application/json",
                        )

    elif analysis_mode == "Dual Date Change Detection":
        with st.spinner("Processing Period 1 and Period 2 imagery..."):
            ndvi1, scene1, gdf1, err1 = fetch_and_process_scene(bbox_input, p1_start.strftime("%Y-%m-%d"), p1_end.strftime("%Y-%m-%d"), cloud_thresh, ndvi_threshold)
            ndvi2, scene2, gdf2, err2 = fetch_and_process_scene(bbox_input, p2_start.strftime("%Y-%m-%d"), p2_end.strftime("%Y-%m-%d"), cloud_thresh, ndvi_threshold)

        if err1 or err2:
            st.warning(f"Error: {err1 or err2}")
        else:
            area1 = gdf1["area_km2"].sum() if gdf1 is not None else 0
            area2 = gdf2["area_km2"].sum() if gdf2 is not None else 0
            diff_area = area2 - area1

            c1, c2, c3 = st.columns(3)
            c1.metric("Period 1 Area", f"{area1:.2f} km²")
            c2.metric("Period 2 Area", f"{area2:.2f} km²")
            c3.metric("Net Change", f"{diff_area:+.2f} km²", delta=f"{diff_area:.2f} km²")

            tab1, tab2 = st.tabs(["Interactive Map", "Data & Reports"])

            with tab1:
                center_lat = (bbox_input[1] + bbox_input[3]) / 2
                center_lon = (bbox_input[0] + bbox_input[2]) / 2
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Esri Satellite").add_to(m)

                if gdf1 is not None:
                    folium.GeoJson(gdf1, name="Period 1 Vegetation (Baseline)", style_function=lambda x: {"fillColor": "#3182bd", "color": "#08519c", "weight": 1.5, "fillOpacity": 0.5}).add_to(m)
                if gdf2 is not None:
                    folium.GeoJson(gdf2, name="Period 2 Vegetation (Comparison)", style_function=lambda x: {"fillColor": "#2ca02c", "color": "#006d2c", "weight": 1.5, "fillOpacity": 0.5}).add_to(m)

                folium.LayerControl().add_to(m)
                components.html(m._repr_html_(), height=550, scrolling=False)

            with tab2:
                st.markdown("### Change Detection Summary Report")
                pct_change = ((area2 - area1) / area1 * 100) if area1 > 0 else 0
                summary_data = {
                    "Metric": ["Analysis Mode", "Period 1 Dates", "Period 2 Dates", "Period 1 Area (km2)", "Period 2 Area (km2)", "Net Area Change (km2)", "Relative Change (%)"],
                    "Value": [analysis_mode, f"{p1_start} to {p1_end}", f"{p2_start} to {p2_end}", f"{area1:.4f}", f"{area2:.4f}", f"{diff_area:+.4f}", f"{pct_change:+.2f}%"],
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

                st.download_button(
                    label="Download Change Detection Report (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name="change_detection_summary.csv",
                    mime="text/csv",
                )

    else:
        with st.spinner("Streaming time-series archive data..."):
            ts_df = fetch_time_series(bbox_input, ts_start.strftime("%Y-%m-%d"), ts_end.strftime("%Y-%m-%d"), cloud_thresh)

        if ts_df is None or ts_df.empty:
            st.warning("No cloud-free imagery available for the selected timeframe.")
        else:
            p1, p2 = st.columns(2)
            p1.metric("Sampled Scenes", f"{len(ts_df)}")
            p2.metric("Overall Mean NDVI", f"{ts_df['Mean NDVI'].mean():.3f}")

            fig = px.line(
                ts_df,
                x="Date",
                y="Mean NDVI",
                markers=True,
                title="Temporal Vegetation Dynamics (Mean NDVI Over Time)",
                template="plotly_dark",
            )
            fig.update_traces(line_color="#2ca02c", marker=dict(size=8, color="#006d2c"))
            fig.update_layout(xaxis_title="Observation Date", yaxis_title="Mean NDVI Index", height=450)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Time-Series Data Stream")
            st.dataframe(ts_df, use_container_width=True)
            st.download_button(
                label="Download Time-Series Metrics (CSV)",
                data=ts_df.to_csv(index=False),
                file_name="time_series_ndvi_metrics.csv",
                mime="text/csv",
            )
else:
    st.info("Select parameters in the sidebar and click 'Run Spatial Analysis' to start.")