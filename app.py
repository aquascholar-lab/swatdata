# ============================================================
# STREAMLIT APP
# NASA POWER DAILY WEATHER DATA DOWNLOADER
# FOR ArcSWAT / SWAT2012 FORMAT
# ============================================================

import os
import re
import time
import shutil
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


# ------------------------------------------------------------
# Force GeoPandas to use pyogrio instead of Fiona
# This is better for Streamlit Cloud deployment
# ------------------------------------------------------------
try:
    gpd.options.io_engine = "pyogrio"
except Exception:
    pass


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NASA POWER to ArcSWAT Weather App",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# CSS
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
    .info-box {
        background-color: #F7F9FC;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #DDE3EA;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌦️ NASA POWER Weather Data Downloader for ArcSWAT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Generate ArcSWAT-ready daily weather files from NASA POWER using subbasin centroids.</div>',
    unsafe_allow_html=True
)


# ============================================================
# PARAMETER SETTINGS
# ============================================================

# ArcSWAT weather types
# pcp   = precipitation
# tmp   = maximum and minimum temperature together
# solar = solar radiation
# rh    = relative humidity
# wind  = wind speed

PARAMETER_INFO = {
    "pcp": {
        "label": "Precipitation",
        "folder": "pcp",
        "index_file": "pcp.txt",
        "nasa": ["PRECTOTCORR"],
        "unit": "mm/day"
    },
    "tmp": {
        "label": "Temperature Tmax and Tmin",
        "folder": "tmp",
        "index_file": "tmp.txt",
        "nasa": ["T2M_MAX", "T2M_MIN"],
        "unit": "degree Celsius"
    },
    "solar": {
        "label": "Solar Radiation",
        "folder": "solar",
        "index_file": "solar.txt",
        "nasa": ["ALLSKY_SFC_SW_DWN"],
        "unit": "MJ/m²/day"
    },
    "rh": {
        "label": "Relative Humidity",
        "folder": "rh",
        "index_file": "rh.txt",
        "nasa": ["RH2M"],
        "unit": "%"
    },
    "wind": {
        "label": "Wind Speed",
        "folder": "wind",
        "index_file": "wind.txt",
        "nasa": ["WS2M"],
        "unit": "m/s"
    }
}

MISSING_VALUE = -99.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    """Make safe file/station name."""
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    if name == "":
        name = "station"
    return name


def save_uploaded_file(uploaded_file, output_path):
    """Save uploaded Streamlit file."""
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def clean_folder(folder_path):
    """Remove and recreate folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def extract_shapefile_zip(zip_path, extract_dir):
    """Extract shapefile ZIP and return .shp path."""
    clean_folder(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    shp_files = []

    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, file))

    if len(shp_files) == 0:
        raise FileNotFoundError("No .shp file found in uploaded ZIP.")

    return shp_files[0]


def build_station_table(subbasin_shp, dem_path, name_field=None):
    """Generate station table from subbasin centroids and DEM elevation."""

    sub_gdf = gpd.read_file(subbasin_shp)

    if sub_gdf.empty:
        raise ValueError("The uploaded subbasin shapefile is empty.")

    if sub_gdf.crs is None:
        raise ValueError("Subbasin shapefile has no CRS. Please define projection first.")

    original_crs = str(sub_gdf.crs)

    # Use projected CRS for accurate centroid calculation
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
        centroid_gdf["NAME"] = centroid_gdf[name_field].astype(str).apply(safe_filename)
    else:
        centroid_gdf["NAME"] = [f"sub{i + 1:03d}" for i in range(len(centroid_gdf))]

    # Extract elevation from DEM
    elevations = []

    with rasterio.open(dem_path) as dem:
        dem_crs = dem.crs

        if dem_crs is None:
            raise ValueError("DEM has no CRS. Please define projection first.")

        for _, row in centroid_gdf.iterrows():
            lon = float(row["LONG"])
            lat = float(row["LAT"])

            try:
                x, y = transform("EPSG:4326", dem_crs, [lon], [lat])
                value = list(dem.sample([(x[0], y[0])]))[0][0]
            except Exception:
                value = np.nan

            if dem.nodata is not None and value == dem.nodata:
                value = np.nan

            if not np.isfinite(value):
                value = np.nan

            elevations.append(value)

    centroid_gdf["ELEVATION"] = elevations

    mean_elev = centroid_gdf["ELEVATION"].replace([np.inf, -np.inf], np.nan).mean()

    if np.isnan(mean_elev):
        mean_elev = 0.0

    centroid_gdf["ELEVATION"] = centroid_gdf["ELEVATION"].fillna(mean_elev)

    station_df = centroid_gdf[["ID", "NAME", "LAT", "LONG", "ELEVATION"]].copy()

    station_df["LAT"] = station_df["LAT"].round(6)
    station_df["LONG"] = station_df["LONG"].round(6)
    station_df["ELEVATION"] = station_df["ELEVATION"].round(2)

    return station_df, original_crs


def get_required_nasa_parameters(selected_parameters):
    """Get unique NASA POWER parameter list."""
    required = []

    for parameter in selected_parameters:
        required.extend(PARAMETER_INFO[parameter]["nasa"])

    return sorted(list(set(required)))


def download_nasa_power_daily(lat, lon, start_date, end_date, nasa_parameters, max_retries=5):
    """Download NASA POWER daily data for one station."""

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
    """Prepare NASA POWER data into clean SWAT variables."""

    out = pd.DataFrame(index=raw_df.index)
    out["DATE"] = out.index.strftime("%Y%m%d")

    if "PRECTOTCORR" in raw_df.columns:
        out["PCP_mm"] = raw_df["PRECTOTCORR"].clip(lower=0)

    if "T2M_MAX" in raw_df.columns:
        out["TMAX_C"] = raw_df["T2M_MAX"]

    if "T2M_MIN" in raw_df.columns:
        out["TMIN_C"] = raw_df["T2M_MIN"]

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


def write_arcswat_index_file(folder_path, index_file_name, station_df):
    """
    Write ArcSWAT station index file:
    ID,NAME,LAT,LONG,ELEVATION
    """

    os.makedirs(folder_path, exist_ok=True)

    index_path = os.path.join(folder_path, index_file_name)

    index_df = station_df[["ID", "NAME", "LAT", "LONG", "ELEVATION"]].copy()

    index_df.to_csv(index_path, index=False)

    return index_path


def write_arcswat_station_file(folder_path, station_name, start_date_str, values_lines):
    """
    Write ArcSWAT station file:
    First line  = YYYYMMDD
    Other lines = daily values only
    """

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{safe_filename(station_name)}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(start_date_str) + "\n")

        for line in values_lines:
            f.write(str(line) + "\n")

    return file_path


def write_parameter_files_arcswat(output_dir, parameter_key, station_name, weather_df):
    """
    Write ArcSWAT station files for selected parameter.
    """

    info = PARAMETER_INFO[parameter_key]
    folder_path = os.path.join(output_dir, info["folder"])

    start_date_str = str(weather_df["DATE"].iloc[0])

    if parameter_key == "pcp":
        values = weather_df["PCP_mm"].map(lambda x: f"{x:.3f}").tolist()

    elif parameter_key == "tmp":
        values = weather_df.apply(
            lambda row: f"{row['TMAX_C']:.3f},{row['TMIN_C']:.3f}",
            axis=1
        ).tolist()

    elif parameter_key == "solar":
        values = weather_df["SOLAR_MJ_m2_day"].map(lambda x: f"{x:.3f}").tolist()

    elif parameter_key == "rh":
        values = weather_df["RH_percent"].map(lambda x: f"{x:.3f}").tolist()

    elif parameter_key == "wind":
        values = weather_df["WIND_m_s"].map(lambda x: f"{x:.3f}").tolist()

    else:
        raise ValueError(f"Unsupported parameter: {parameter_key}")

    file_path = write_arcswat_station_file(
        folder_path=folder_path,
        station_name=station_name,
        start_date_str=start_date_str,
        values_lines=values
    )

    return file_path


def create_zip_from_folder(folder_path, zip_path):
    """Create ZIP from output folder."""

    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                z.write(full_path, arcname=arcname)

    return zip_path


def parameter_label_map():
    """Create label-to-key map."""
    return {v["label"]: k for k, v in PARAMETER_INFO.items()}


# ============================================================
# SIDEBAR
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
        help="DEM is used to extract elevation at each subbasin centroid."
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

    labels = parameter_label_map()

    default_labels = [
        "Precipitation",
        "Temperature Tmax and Tmin",
        "Solar Radiation",
        "Relative Humidity",
        "Wind Speed"
    ]

    selected_labels = st.multiselect(
        "Select ArcSWAT weather parameters",
        options=list(labels.keys()),
        default=default_labels
    )

    selected_parameters = [labels[x] for x in selected_labels]

    st.divider()

    request_delay = st.number_input(
        "NASA request delay in seconds",
        min_value=0.0,
        max_value=10.0,
        value=0.75,
        step=0.25
    )

    save_check_csv = st.checkbox(
        "Also save combined daily CSV for checking",
        value=True
    )

    st.info(
        "ArcSWAT format will be generated automatically. "
        "Station files will not contain DATE,VALUE headers."
    )


# ============================================================
# MAIN INFO
# ============================================================

st.markdown(
    """
    <div class="info-box">
    <b>ArcSWAT Output Format</b><br>
    For each weather parameter, this app creates one station index file such as 
    <code>pcp.txt</code>, <code>tmp.txt</code>, <code>solar.txt</code>, 
    <code>rh.txt</code>, and <code>wind.txt</code>. 
    Each station file starts with <code>YYYYMMDD</code> followed by daily values only.
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Weather Source", "NASA POWER")

with c2:
    st.metric("Output Format", "ArcSWAT")

with c3:
    st.metric("Station Method", "Subbasin Centroid")

with c4:
    st.metric("Elevation Source", "DEM")


# ============================================================
# VALIDATION
# ============================================================

if subbasin_zip is None or dem_file is None:
    st.warning("Please upload both subbasin shapefile ZIP and DEM GeoTIFF.")
    st.stop()

if len(selected_parameters) == 0:
    st.warning("Please select at least one weather parameter.")
    st.stop()

if start_date > end_date:
    st.error("Start date must be earlier than or equal to end date.")
    st.stop()


# ============================================================
# WORKSPACE
# ============================================================

if "workspace" not in st.session_state:
    st.session_state.workspace = tempfile.mkdtemp()

workspace = st.session_state.workspace

input_dir = os.path.join(workspace, "inputs")
extract_dir = os.path.join(workspace, "subbasin_extract")
output_dir = os.path.join(workspace, "ArcSWAT_NASA_POWER_Output")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# SAVE UPLOADED FILES
# ============================================================

sub_zip_path = os.path.join(input_dir, subbasin_zip.name)
dem_path = os.path.join(input_dir, dem_file.name)

save_uploaded_file(subbasin_zip, sub_zip_path)
save_uploaded_file(dem_file, dem_path)


# ============================================================
# READ SHAPEFILE FOR FIELD SELECTION
# ============================================================

try:
    subbasin_shp = extract_shapefile_zip(sub_zip_path, extract_dir)
    preview_gdf = gpd.read_file(subbasin_shp)

    available_fields = ["Auto sub001, sub002..."] + [
        c for c in preview_gdf.columns
        if c != "geometry"
    ]

    name_field = st.selectbox(
        "Select field for station names / file names",
        options=available_fields,
        index=0
    )

except Exception as e:
    st.error(f"Could not read shapefile: {e}")
    st.stop()


# ============================================================
# BUILD STATION TABLE
# ============================================================

try:
    station_df, original_crs = build_station_table(
        subbasin_shp=subbasin_shp,
        dem_path=dem_path,
        name_field=name_field
    )

except Exception as e:
    st.error(f"Error while creating stations: {e}")
    st.stop()


# ============================================================
# DISPLAY STATION TABLE AND MAP
# ============================================================

st.subheader("📍 Subbasin Weather Stations")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Number of stations", len(station_df))

with m2:
    st.metric("Selected parameters", len(selected_parameters))

with m3:
    st.metric("Start year", start_date.year)

with m4:
    st.metric("End year", end_date.year)

st.write(f"**Subbasin CRS:** `{original_crs}`")

st.dataframe(station_df, use_container_width=True)

map_df = station_df.rename(columns={"LAT": "lat", "LONG": "lon"})
st.map(map_df[["lat", "lon"]])


# ============================================================
# START PROCESSING
# ============================================================

st.subheader("🚀 Generate ArcSWAT Weather Files")

run_button = st.button("Start NASA POWER Download and Create ArcSWAT Files", type="primary")

if run_button:

    clean_folder(output_dir)

    # Create parameter folders and ArcSWAT station index files
    for p in selected_parameters:
        info = PARAMETER_INFO[p]
        folder_path = os.path.join(output_dir, info["folder"])
        os.makedirs(folder_path, exist_ok=True)

        write_arcswat_index_file(
            folder_path=folder_path,
            index_file_name=info["index_file"],
            station_df=station_df
        )

    # Save station summary
    station_summary_path = os.path.join(output_dir, "station_summary.csv")
    station_df.to_csv(station_summary_path, index=False)

    # Optional CSV checking folder
    check_csv_dir = os.path.join(output_dir, "daily_csv_for_checking")

    if save_check_csv:
        os.makedirs(check_csv_dir, exist_ok=True)

    required_nasa_parameters = get_required_nasa_parameters(selected_parameters)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    progress_bar = st.progress(0)
    status_text = st.empty()

    logs = []
    preview_store = {}

    for idx, row in station_df.iterrows():

        station_id = int(row["ID"])
        station_name = str(row["NAME"])
        lat = float(row["LAT"])
        lon = float(row["LONG"])
        elev = float(row["ELEVATION"])

        status_text.info(
            f"Downloading station {idx + 1}/{len(station_df)}: "
            f"{station_name} ({lat:.5f}, {lon:.5f})"
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

            # Save CSV for checking
            if save_check_csv:
                check_csv_path = os.path.join(
                    check_csv_dir,
                    f"{safe_filename(station_name)}_NASA_POWER_daily.csv"
                )
                weather_df.to_csv(check_csv_path, index=False)

            # Write ArcSWAT files
            for p in selected_parameters:
                write_parameter_files_arcswat(
                    output_dir=output_dir,
                    parameter_key=p,
                    station_name=station_name,
                    weather_df=weather_df
                )

            logs.append({
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
            logs.append({
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

    log_df = pd.DataFrame(logs)
    log_path = os.path.join(output_dir, "download_log.csv")
    log_df.to_csv(log_path, index=False)

    # Create README
    readme_path = os.path.join(output_dir, "README_ArcSWAT_Format.txt")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("NASA POWER Weather Data Prepared for ArcSWAT / SWAT2012\n")
        f.write("=====================================================\n\n")
        f.write("Folder contents:\n")
        f.write("pcp/pcp.txt     = precipitation station index file\n")
        f.write("tmp/tmp.txt     = temperature station index file\n")
        f.write("solar/solar.txt = solar radiation station index file\n")
        f.write("rh/rh.txt       = relative humidity station index file\n")
        f.write("wind/wind.txt   = wind speed station index file\n\n")
        f.write("Station files:\n")
        f.write("First line is start date in YYYYMMDD format.\n")
        f.write("Remaining lines contain daily values only.\n\n")
        f.write("Temperature file format:\n")
        f.write("First line: YYYYMMDD\n")
        f.write("Daily lines: TMAX,TMIN\n\n")
        f.write("Units:\n")
        f.write("Precipitation = mm/day\n")
        f.write("Temperature = degree Celsius\n")
        f.write("Solar radiation = MJ/m2/day\n")
        f.write("Relative humidity = percent\n")
        f.write("Wind speed = m/s\n")

    zip_path = os.path.join(workspace, "ArcSWAT_NASA_POWER_Weather_Output.zip")
    create_zip_from_folder(output_dir, zip_path)

    st.session_state["zip_path"] = zip_path
    st.session_state["output_dir"] = output_dir
    st.session_state["log_df"] = log_df
    st.session_state["preview_store"] = preview_store
    st.session_state["selected_parameters"] = selected_parameters

    status_text.success("ArcSWAT weather files created successfully.")


# ============================================================
# OUTPUT DISPLAY
# ============================================================

if "zip_path" in st.session_state and os.path.exists(st.session_state["zip_path"]):

    st.markdown("---")
    st.subheader("✅ Output Summary")

    log_df = st.session_state["log_df"]

    st.dataframe(log_df, use_container_width=True)

    success_count = (log_df["STATUS"] == "Downloaded").sum()
    failed_count = len(log_df) - success_count

    o1, o2, o3 = st.columns(3)

    with o1:
        st.metric("Successful stations", int(success_count))

    with o2:
        st.metric("Failed stations", int(failed_count))

    with o3:
        st.metric("Output ZIP", "Ready")

    st.subheader("📈 Daily Weather Preview")

    preview_store = st.session_state["preview_store"]

    if len(preview_store) > 0:

        station_choice = st.selectbox(
            "Select station for preview",
            options=list(preview_store.keys())
        )

        preview_df = preview_store[station_choice].copy()
        preview_df["DATE_DT"] = pd.to_datetime(preview_df["DATE"], format="%Y%m%d")

        value_columns = [
            c for c in preview_df.columns
            if c not in ["DATE", "DATE_DT"]
        ]

        selected_col = st.selectbox(
            "Select variable",
            options=value_columns
        )

        chart_df = preview_df[["DATE_DT", selected_col]].set_index("DATE_DT")

        st.line_chart(chart_df)

        st.write("First 20 rows")
        st.dataframe(preview_df.head(20), use_container_width=True)

    st.subheader("📦 Download ArcSWAT ZIP")

    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button(
            label="Download ArcSWAT NASA POWER Weather ZIP",
            data=f,
            file_name="ArcSWAT_NASA_POWER_Weather_Output.zip",
            mime="application/zip",
            type="primary"
        )

    st.info(
        "Use pcp/pcp.txt, tmp/tmp.txt, solar/solar.txt, rh/rh.txt, and wind/wind.txt "
        "inside ArcSWAT Weather Data Definition."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Prepared for ArcSWAT / SWAT2012 weather input generation using NASA POWER daily point data."
)
