# ============================================================
# STREAMLIT APP
# NASA POWER DAILY WEATHER DATA DOWNLOADER FOR SWAT INPUT
# ============================================================
# Features:
# - Upload subbasin shapefile ZIP
# - Upload DEM GeoTIFF
# - Select date range
# - Select weather parameters
# - Generate one INDEX.txt for each selected weather parameter
# - Generate one text file per subbasin/station
# - Visualize subbasin centroid stations
# - Visualize downloaded daily weather time series
# - Download final output as ZIP
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import re
import time
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import streamlit as st

from datetime import date
from rasterio.warp import transform


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NASA POWER to SWAT Weather App",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #0B3D91;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 17px;
        color: #444444;
        margin-bottom: 20px;
    }
    .section-box {
        background-color: #F7F9FC;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #E1E5EE;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E6E6E6;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. APP HEADER
# ============================================================

st.markdown('<div class="main-title">🌦️ NASA POWER Weather Data Downloader for SWAT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Generate SWAT-ready daily weather text files and INDEX files for each subbasin using NASA POWER data.</div>',
    unsafe_allow_html=True
)


# ============================================================
# 5. GLOBAL SETTINGS
# ============================================================

NASA_PARAMETER_INFO = {
    "pcp": {
        "label": "Precipitation",
        "folder": "pcp",
        "nasa": ["PRECTOTCORR"],
        "columns": ["DATE", "PCP_mm"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "mm/day"
    },
    "tmax": {
        "label": "Maximum Temperature",
        "folder": "tmax",
        "nasa": ["T2M_MAX"],
        "columns": ["DATE", "TMAX_C"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "°C"
    },
    "tmin": {
        "label": "Minimum Temperature",
        "folder": "tmin",
        "nasa": ["T2M_MIN"],
        "columns": ["DATE", "TMIN_C"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "°C"
    },
    "tmp": {
        "label": "Temperature Tmax and Tmin",
        "folder": "tmp",
        "nasa": ["T2M_MAX", "T2M_MIN"],
        "columns": ["DATE", "TMAX_C", "TMIN_C"],
        "out_columns": ["DATE", "TMAX_C", "TMIN_C"],
        "unit": "°C"
    },
    "tmean": {
        "label": "Mean Temperature",
        "folder": "tmean",
        "nasa": ["T2M"],
        "columns": ["DATE", "TMEAN_C"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "°C"
    },
    "rh": {
        "label": "Relative Humidity",
        "folder": "rh",
        "nasa": ["RH2M"],
        "columns": ["DATE", "RH_percent"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "%"
    },
    "wind": {
        "label": "Wind Speed",
        "folder": "wind",
        "nasa": ["WS2M"],
        "columns": ["DATE", "WIND_m_s"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "m/s"
    },
    "solar": {
        "label": "Solar Radiation",
        "folder": "solar",
        "nasa": ["ALLSKY_SFC_SW_DWN"],
        "columns": ["DATE", "SOLAR_MJ_m2_day"],
        "out_columns": ["DATE", "VALUE"],
        "unit": "MJ/m²/day"
    }
}

MISSING_VALUE = -99.0


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name


def save_uploaded_file(uploaded_file, output_path):
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def extract_shapefile_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    shp_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, file))

    if len(shp_files) == 0:
        raise FileNotFoundError("No .shp file found in the uploaded ZIP.")

    return shp_files[0]


def build_station_table(subbasin_shp, dem_path, name_field=None):
    sub_gdf = gpd.read_file(subbasin_shp)

    if sub_gdf.empty:
        raise ValueError("The uploaded subbasin shapefile is empty.")

    if sub_gdf.crs is None:
        raise ValueError("The subbasin shapefile has no CRS. Please define projection first.")

    original_crs = str(sub_gdf.crs)

    estimated_utm = sub_gdf.estimate_utm_crs()

    if estimated_utm is not None:
        sub_proj = sub_gdf.to_crs(estimated_utm)
    else:
        sub_proj = sub_gdf.to_crs(epsg=3857)

    sub_proj["centroid_geom"] = sub_proj.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        sub_proj.drop(columns="geometry"),
        geometry=sub_proj["centroid_geom"],
        crs=sub_proj.crs
    ).to_crs(epsg=4326)

    centroid_gdf["LAT"] = centroid_gdf.geometry.y
    centroid_gdf["LONG"] = centroid_gdf.geometry.x
    centroid_gdf["ID"] = np.arange(1, len(centroid_gdf) + 1)

    if name_field is not None and name_field != "Auto sub001, sub002...":
        centroid_gdf["NAME"] = centroid_gdf[name_field].astype(str)
    else:
        centroid_gdf["NAME"] = [f"sub{i+1:03d}" for i in range(len(centroid_gdf))]

    elevations = []

    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs

        if dem_crs is None:
            raise ValueError("The DEM has no CRS. Please define projection first.")

        for _, row in centroid_gdf.iterrows():
            lon = float(row["LONG"])
            lat = float(row["LAT"])

            try:
                x, y = transform("EPSG:4326", dem_crs, [lon], [lat])
                val = list(dem.sample([(x[0], y[0])]))[0][0]
            except Exception:
                val = np.nan

            if dem.nodata is not None and val == dem.nodata:
                val = np.nan

            if not np.isfinite(val):
                val = np.nan

            elevations.append(float(val) if np.isfinite(val) else np.nan)

    centroid_gdf["ELEVATION"] = elevations

    mean_elev = centroid_gdf["ELEVATION"].replace([np.inf, -np.inf], np.nan).mean()
    centroid_gdf["ELEVATION"] = centroid_gdf["ELEVATION"].fillna(mean_elev)

    station_df = centroid_gdf[["ID", "NAME", "LAT", "LONG", "ELEVATION"]].copy()
    station_df["LAT"] = station_df["LAT"].round(6)
    station_df["LONG"] = station_df["LONG"].round(6)
    station_df["ELEVATION"] = station_df["ELEVATION"].round(2)

    return station_df, original_crs


def get_required_nasa_parameters(selected_parameters):
    required = []

    for p in selected_parameters:
        required.extend(NASA_PARAMETER_INFO[p]["nasa"])

    required = sorted(list(set(required)))
    return required


def download_nasa_power_daily(lat, lon, start_date, end_date, nasa_parameters, max_retries=5):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": ",".join(nasa_parameters),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=180)

            if response.status_code == 200:
                js = response.json()

                if "properties" not in js or "parameter" not in js["properties"]:
                    raise RuntimeError("Unexpected NASA POWER response format.")

                data = js["properties"]["parameter"]
                df = pd.DataFrame(data)

                df.index = pd.to_datetime(df.index, format="%Y%m%d")
                df = df.sort_index()

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.replace([-999, -9999, -99], np.nan)

                return df

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"

        except Exception as e:
            last_error = str(e)

        time.sleep(2 * attempt)

    raise RuntimeError(f"NASA POWER request failed. Last error: {last_error}")


def prepare_weather_dataframe(raw_df):
    out = pd.DataFrame(index=raw_df.index)
    out["DATE"] = out.index.strftime("%Y%m%d")

    if "PRECTOTCORR" in raw_df.columns:
        out["PCP_mm"] = raw_df["PRECTOTCORR"].clip(lower=0)

    if "T2M_MAX" in raw_df.columns:
        out["TMAX_C"] = raw_df["T2M_MAX"]

    if "T2M_MIN" in raw_df.columns:
        out["TMIN_C"] = raw_df["T2M_MIN"]

    if "T2M" in raw_df.columns:
        out["TMEAN_C"] = raw_df["T2M"]

    if "RH2M" in raw_df.columns:
        out["RH_percent"] = raw_df["RH2M"].clip(lower=0, upper=100)

    if "WS2M" in raw_df.columns:
        out["WIND_m_s"] = raw_df["WS2M"].clip(lower=0)

    if "ALLSKY_SFC_SW_DWN" in raw_df.columns:
        out["SOLAR_MJ_m2_day"] = raw_df["ALLSKY_SFC_SW_DWN"].clip(lower=0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(MISSING_VALUE)

    value_cols = [c for c in out.columns if c != "DATE"]
    out[value_cols] = out[value_cols].round(3)

    return out


def write_index_file(folder_path, station_df):
    index_path = os.path.join(folder_path, "INDEX.txt")

    index_df = station_df[["ID", "NAME", "LAT", "LONG", "ELEVATION"]].copy()
    index_df.to_csv(index_path, index=False)

    return index_path


def write_parameter_station_file(
    out_dir,
    parameter_key,
    station_name,
    weather_df,
    include_header=True,
    delimiter=","
):
    info = NASA_PARAMETER_INFO[parameter_key]
    folder_path = os.path.join(out_dir, info["folder"])
    os.makedirs(folder_path, exist_ok=True)

    selected_cols = info["columns"]
    out_cols = info["out_columns"]

    available_cols = [c for c in selected_cols if c in weather_df.columns]

    if len(available_cols) != len(selected_cols):
        return None

    df_out = weather_df[selected_cols].copy()
    df_out.columns = out_cols

    output_path = os.path.join(folder_path, f"{safe_filename(station_name)}.txt")

    df_out.to_csv(
        output_path,
        index=False,
        header=include_header,
        sep=delimiter
    )

    return output_path


def create_zip_from_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                z.write(full_path, arcname=arcname)

    return zip_path


def make_parameter_label_map():
    return {v["label"]: k for k, v in NASA_PARAMETER_INFO.items()}


# ============================================================
# 7. SIDEBAR INPUTS
# ============================================================

with st.sidebar:
    st.header("⚙️ Input Settings")

    subbasin_zip = st.file_uploader(
        "Upload subbasin shapefile ZIP",
        type=["zip"],
        help="ZIP must contain .shp, .shx, .dbf and .prj files."
    )

    dem_file = st.file_uploader(
        "Upload DEM GeoTIFF",
        type=["tif", "tiff"],
        help="DEM is used to extract elevation for each subbasin centroid."
    )

    st.divider()

    start_date = st.date_input(
        "Start date",
        value=date(1990, 1, 1),
        min_value=date(1981, 1, 1),
        max_value=date(2025, 12, 31)
    )

    end_date = st.date_input(
        "End date",
        value=date(2025, 12, 31),
        min_value=date(1981, 1, 1),
        max_value=date(2025, 12, 31)
    )

    st.divider()

    parameter_label_map = make_parameter_label_map()

    default_labels = [
        "Precipitation",
        "Temperature Tmax and Tmin",
        "Relative Humidity",
        "Wind Speed",
        "Solar Radiation"
    ]

    selected_labels = st.multiselect(
        "Select weather parameters",
        options=list(parameter_label_map.keys()),
        default=default_labels
    )

    selected_parameters = [parameter_label_map[label] for label in selected_labels]

    st.divider()

    include_header = st.checkbox("Include header in station text files", value=True)

    delimiter_option = st.selectbox(
        "Output delimiter",
        options=["Comma (,)", "Tab", "Space"],
        index=0
    )

    if delimiter_option == "Comma (,)":
        delimiter = ","
    elif delimiter_option == "Tab":
        delimiter = "\t"
    else:
        delimiter = " "

    request_delay = st.number_input(
        "NASA request delay in seconds",
        min_value=0.0,
        max_value=10.0,
        value=0.75,
        step=0.25
    )

    st.info("For many subbasins and long date ranges, downloading may take time because NASA POWER is queried point-by-point.")


# ============================================================
# 8. MAIN LAYOUT
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">📁 <b>Input 1</b><br>Subbasin Shapefile ZIP</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">🗻 <b>Input 2</b><br>DEM GeoTIFF</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">🌐 <b>Source</b><br>NASA POWER Daily API</div>', unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# 9. VALIDATION
# ============================================================

if subbasin_zip is None or dem_file is None:
    st.warning("Please upload both the subbasin shapefile ZIP and DEM GeoTIFF from the sidebar.")
    st.stop()

if len(selected_parameters) == 0:
    st.warning("Please select at least one weather parameter.")
    st.stop()

if start_date > end_date:
    st.error("Start date must be earlier than or equal to end date.")
    st.stop()


# ============================================================
# 10. TEMPORARY WORKSPACE
# ============================================================

if "workspace" not in st.session_state:
    st.session_state.workspace = tempfile.mkdtemp()

workspace = st.session_state.workspace

input_dir = os.path.join(workspace, "inputs")
output_dir = os.path.join(workspace, "NASA_POWER_SWAT_Output")
extract_dir = os.path.join(workspace, "subbasin_extract")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)


# ============================================================
# 11. SAVE UPLOADED FILES
# ============================================================

sub_zip_path = os.path.join(input_dir, subbasin_zip.name)
dem_path = os.path.join(input_dir, dem_file.name)

save_uploaded_file(subbasin_zip, sub_zip_path)
save_uploaded_file(dem_file, dem_path)


# ============================================================
# 12. READ SHAPEFILE FIRST TO ALLOW NAME FIELD SELECTION
# ============================================================

try:
    subbasin_shp = extract_shapefile_zip(sub_zip_path, extract_dir)
    preview_gdf = gpd.read_file(subbasin_shp)

    available_fields = ["Auto sub001, sub002..."] + [
        c for c in preview_gdf.columns
        if c != "geometry"
    ]

    name_field = st.selectbox(
        "Select field for station/subbasin file names",
        options=available_fields,
        index=0
    )

except Exception as e:
    st.error(f"Could not read shapefile: {e}")
    st.stop()


# ============================================================
# 13. BUILD STATION TABLE
# ============================================================

try:
    station_df, original_crs = build_station_table(
        subbasin_shp=subbasin_shp,
        dem_path=dem_path,
        name_field=name_field
    )

except Exception as e:
    st.error(f"Error while generating station table: {e}")
    st.stop()


# ============================================================
# 14. DISPLAY STATION INFORMATION
# ============================================================

st.subheader("📍 Generated Subbasin Weather Stations")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Number of stations", len(station_df))

with c2:
    st.metric("Selected parameters", len(selected_parameters))

with c3:
    st.metric("Start year", start_date.year)

with c4:
    st.metric("End year", end_date.year)

st.write(f"**Subbasin CRS:** `{original_crs}`")

st.dataframe(station_df, use_container_width=True)

map_df = station_df.rename(columns={"LAT": "lat", "LONG": "lon"})
st.map(map_df[["lat", "lon"]])


# ============================================================
# 15. RUN DOWNLOAD
# ============================================================

st.subheader("🚀 Download and Generate SWAT Weather Files")

run_button = st.button("Start NASA POWER Download", type="primary")

if run_button:

    # Clear old output
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))

    os.makedirs(output_dir, exist_ok=True)

    # Create parameter folders and INDEX files
    for p in selected_parameters:
        folder = NASA_PARAMETER_INFO[p]["folder"]
        folder_path = os.path.join(output_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        write_index_file(folder_path, station_df)

    # Save station summary
    station_summary_path = os.path.join(output_dir, "station_summary.csv")
    station_df.to_csv(station_summary_path, index=False)

    # Combined daily folder
    combined_dir = os.path.join(output_dir, "combined_daily_csv")
    os.makedirs(combined_dir, exist_ok=True)

    required_nasa_parameters = get_required_nasa_parameters(selected_parameters)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_logs = []
    preview_store = {}

    for idx, row in station_df.iterrows():

        station_id = int(row["ID"])
        station_name = str(row["NAME"])
        lat = float(row["LAT"])
        lon = float(row["LONG"])
        elev = float(row["ELEVATION"])

        status_text.info(
            f"Downloading station {idx + 1}/{len(station_df)}: {station_name} "
            f"({lat:.5f}, {lon:.5f})"
        )

        try:
            raw_df = download_nasa_power_daily(
                lat=lat,
                lon=lon,
                start_date=start_str,
                end_date=end_str,
                nasa_parameters=required_nasa_parameters
            )

            weather_df = prepare_weather_dataframe(raw_df)

            combined_csv = os.path.join(
                combined_dir,
                f"{safe_filename(station_name)}_NASA_POWER_daily.csv"
            )
            weather_df.to_csv(combined_csv, index=False)

            for p in selected_parameters:
                write_parameter_station_file(
                    out_dir=output_dir,
                    parameter_key=p,
                    station_name=station_name,
                    weather_df=weather_df,
                    include_header=include_header,
                    delimiter=delimiter
                )

            all_logs.append({
                "ID": station_id,
                "NAME": station_name,
                "LAT": lat,
                "LONG": lon,
                "ELEVATION": elev,
                "START_DATE": start_str,
                "END_DATE": end_str,
                "N_DAYS": len(weather_df),
                "STATUS": "Downloaded"
            })

            if len(preview_store) < 5:
                preview_store[station_name] = weather_df.copy()

        except Exception as e:
            all_logs.append({
                "ID": station_id,
                "NAME": station_name,
                "LAT": lat,
                "LONG": lon,
                "ELEVATION": elev,
                "START_DATE": start_str,
                "END_DATE": end_str,
                "N_DAYS": 0,
                "STATUS": f"Failed: {e}"
            })

        progress_bar.progress((idx + 1) / len(station_df))
        time.sleep(request_delay)

    log_df = pd.DataFrame(all_logs)
    log_path = os.path.join(output_dir, "download_log.csv")
    log_df.to_csv(log_path, index=False)

    zip_path = os.path.join(workspace, "NASA_POWER_SWAT_Weather_Output.zip")
    create_zip_from_folder(output_dir, zip_path)

    st.session_state["output_dir"] = output_dir
    st.session_state["zip_path"] = zip_path
    st.session_state["log_df"] = log_df
    st.session_state["preview_store"] = preview_store
    st.session_state["selected_parameters"] = selected_parameters

    status_text.success("Processing completed successfully.")


# ============================================================
# 16. OUTPUT VISUALIZATION AND DOWNLOAD
# ============================================================

if "zip_path" in st.session_state and os.path.exists(st.session_state["zip_path"]):

    st.markdown("---")
    st.subheader("✅ Output Summary")

    log_df = st.session_state["log_df"]
    st.dataframe(log_df, use_container_width=True)

    success_count = (log_df["STATUS"] == "Downloaded").sum()
    failed_count = len(log_df) - success_count

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Successful stations", int(success_count))

    with c2:
        st.metric("Failed stations", int(failed_count))

    with c3:
        st.metric("Output folders", len(st.session_state["selected_parameters"]))

    st.subheader("📈 Weather Time-Series Preview")

    preview_store = st.session_state["preview_store"]

    if len(preview_store) > 0:

        station_choice = st.selectbox(
            "Select station for preview",
            options=list(preview_store.keys())
        )

        preview_df = preview_store[station_choice].copy()
        preview_df["DATE_DT"] = pd.to_datetime(preview_df["DATE"], format="%Y%m%d")

        value_columns = [c for c in preview_df.columns if c not in ["DATE", "DATE_DT"]]

        selected_preview_col = st.selectbox(
            "Select variable for chart",
            options=value_columns
        )

        chart_df = preview_df[["DATE_DT", selected_preview_col]].copy()
        chart_df = chart_df.set_index("DATE_DT")

        st.line_chart(chart_df)

        st.write("Data preview")
        st.dataframe(preview_df.head(20), use_container_width=True)

    st.subheader("📦 Download Final ZIP")

    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button(
            label="Download NASA POWER SWAT Weather ZIP",
            data=f,
            file_name="NASA_POWER_SWAT_Weather_Output.zip",
            mime="application/zip",
            type="primary"
        )

    st.info(
        "The ZIP contains one folder for each selected weather parameter. "
        "Each folder has an INDEX.txt file and separate station text files."
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Developed for SWAT weather data preparation using NASA POWER daily point data. "
    "Recommended workflow: verify station coordinates, check units, then import into SWAT/QSWAT/QSWAT+."
)
