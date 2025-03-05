import os
import glob
import logging

import pandas as pd
import streamlit as st
import plotly.express as px

#########################################
# Logging Configuration
#########################################

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

#########################################
# Global Variables / Settings
#########################################

DATA_DIR_2023 = "VIZ_2023_Messquerschnitt"
DATA_DIR_2024 = "VIZ_2024_Messquerschnitt"

MEASUREMENT_COLS = [
    "q_kfz_mq_hr",
    "v_kfz_mq_hr",
    "q_pkw_mq_hr",
    "v_pkw_mq_hr",
    "q_lkw_mq_hr",
    "v_lkw_mq_hr"
]

# Mapping between Python weekday (0=Monday, ..., 6=Sunday) and readable labels
WEEKDAY_MAP = {
    0: "Mo",
    1: "Tu",
    2: "We",
    3: "Th",
    4: "Fr",
    5: "Sa",
    6: "Su"
}
WEEKDAY_REVERSE_MAP = {v: k for k, v in WEEKDAY_MAP.items()}


#########################################
# Helper Functions
#########################################

def load_messquerschnitt_data(year_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from the specified directory (year_dir) that match the
    Messquerschnitt format (excluding files with 'stammdaten' in the name).
    Concatenate them into a single DataFrame.

    Parameters
    ----------
    year_dir : str
        Directory path where the Messquerschnitt CSV files are stored.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame containing all relevant CSV data.
    """
    all_files = glob.glob(os.path.join(year_dir, "*.csv"))
    df_list = []

    for filename in all_files:
        # Ignore Stammdaten files
        if "stammdaten" in filename.lower():
            continue
        try:
            temp_df = pd.read_csv(filename, header=0, delimiter=";")
            df_list.append(temp_df)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

    if len(df_list) == 0:
        logger.warning(f"No Messquerschnitt CSV files found in '{year_dir}'.")
        return pd.DataFrame()

    return pd.concat(df_list, axis=0, ignore_index=True)


def load_stammdaten(year_dir: str) -> pd.DataFrame:
    """
    Load the Stammdaten CSV file from the given folder (year_dir).
    Returns an empty DataFrame if none are found.

    Parameters
    ----------
    year_dir : str
        Directory path where the Stammdaten CSV file might be stored.

    Returns
    -------
    pd.DataFrame
        The Stammdaten DataFrame (empty if not found or on error).
    """
    all_files = glob.glob(os.path.join(year_dir, "*.csv"))
    for filename in all_files:
        if "stammdaten" in filename.lower():
            try:
                return pd.read_csv(filename, header=0, delimiter=";")
            except Exception as e:
                logger.error(f"Error reading stammdaten '{filename}': {e}")
                return pd.DataFrame()

    logger.warning(f"No stammdaten CSV found in '{year_dir}'.")
    return pd.DataFrame()


@st.cache_data
def get_missing_data_info_cached(df: pd.DataFrame, min_hours_per_day: int) -> pd.DataFrame:
    """
    Cached version of get_missing_data_info() for the scenario
    where all stations are considered (i.e. no station filter).

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    min_hours_per_day : int
        Minimum number of hours per day to consider the day valid.

    Returns
    -------
    pd.DataFrame
        Missing data information.
    """
    return get_missing_data_info(df, None, min_hours_per_day)


def get_missing_data_info(
    df: pd.DataFrame,
    selected_stations: list = None,
    min_hours_per_day: int = 0
) -> pd.DataFrame:
    """
    Identify missing hours for each day within the range of min(tag_dt) to max(tag_dt).
    If selected_stations is None or empty => consider all stations;
    otherwise only consider those in `selected_stations`.

    A day is completely *ignored* (i.e., not reported as missing hours) if
    that station on that day has fewer than `min_hours_per_day` records.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame containing 'mq_name', 'tag', 'stunde', etc.
    selected_stations : list, optional
        List of station names to consider. If empty or None, all are considered.
    min_hours_per_day : int, optional
        Minimum number of hours required for a day to be included.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the missing hour details for each station/day,
        with columns ["mq_name", "date", "missing_hours"].
    """
    if df.empty:
        return pd.DataFrame(columns=["mq_name", "date", "missing_hours"])

    # Ensure selected_stations is a list if given
    if selected_stations is not None and not isinstance(selected_stations, list):
        selected_stations = list(selected_stations)

    # Filter by selected_stations if any
    if selected_stations is not None and len(selected_stations) > 0:
        df = df[df['mq_name'].isin(selected_stations)].copy()
        if df.empty:
            return pd.DataFrame(columns=["mq_name", "date", "missing_hours"])

    # Parse dates if needed
    df['tag_dt'] = pd.to_datetime(df['tag'], dayfirst=True, errors='coerce')
    if df['tag_dt'].isna().all():
        logger.warning("Could not parse any valid date in the 'tag' column.")
        return pd.DataFrame(columns=["mq_name", "date", "missing_hours"])

    min_date = df['tag_dt'].min()
    max_date = df['tag_dt'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    stations = df['mq_name'].unique()

    rows = []
    for station in stations:
        station_data = df[df['mq_name'] == station]

        for single_day in all_dates:
            day_data = station_data[station_data['tag_dt'] == single_day]

            # Skip the day entirely if it has fewer records than the min required
            if len(day_data) < min_hours_per_day:
                continue

            # If there's no data at all for that day => all 24 hours missing
            if day_data.empty:
                missing_hours = list(range(24))
            else:
                existing_hours = day_data['stunde'].unique()
                all_hours = set(range(24))
                missing_hours = sorted(list(all_hours - set(existing_hours)))

            if len(missing_hours) > 0:
                rows.append({
                    "mq_name": station,
                    "date": single_day.strftime("%Y-%m-%d"),
                    "missing_hours": missing_hours
                })

    return pd.DataFrame(rows)


@st.cache_data
def get_compact_missing_data_cached(df_missing: pd.DataFrame) -> pd.DataFrame:
    """
    Cached version of get_compact_missing_data() for the scenario
    with no station filter. Avoids recalculations.

    Parameters
    ----------
    df_missing : pd.DataFrame
        The missing data detail DataFrame.

    Returns
    -------
    pd.DataFrame
        A compact DataFrame summarizing total missing hours per station.
    """
    return get_compact_missing_data(df_missing)


def get_compact_missing_data(df_missing: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the number of missing hours per station into a concise summary.
    Returns [mq_name, total_missing_hours].

    NOTE: This function assumes `df_missing` has already excluded/ignored
    station-days with fewer than `min_hours_per_day`. The days you do not
    want to count should be filtered out in `get_missing_data_info(...)`.

    Parameters
    ----------
    df_missing : pd.DataFrame
        The missing data detail DataFrame with columns ["mq_name", "missing_hours"].

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["mq_name", "total_missing_hours"].
    """
    if df_missing.empty:
        return pd.DataFrame(columns=["mq_name", "total_missing_hours"])

    # Count how many hours are missing (i.e., length of the list in 'missing_hours')
    df_missing["missing_count"] = df_missing["missing_hours"].apply(len)

    # Group by station and sum up missing hours
    df_compact = df_missing.groupby("mq_name", as_index=False)["missing_count"].sum()

    # Rename for clarity
    df_compact.rename(columns={"missing_count": "total_missing_hours"}, inplace=True)
    return df_compact


def prepare_data_for_bar_plots(df: pd.DataFrame,
                               selected_cols: list) -> tuple:
    """
    Prepare data for bar plots by calculating the average per hour and per weekday.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame after filtering.
    selected_cols : list
        Columns to average (e.g., 'q_kfz_mq_hr').

    Returns
    -------
    tuple of pd.DataFrame
        (df_hour_mean, df_weekday_mean) for hour-based and weekday-based bar charts.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df['tag_dt'] = pd.to_datetime(df['tag'], dayfirst=True, errors='coerce')
    df = df[~df['tag_dt'].isna()].copy()
    df['weekday'] = df['tag_dt'].dt.weekday

    df_hour_mean = (
        df.groupby('stunde')[selected_cols]
          .mean(numeric_only=True)
          .reset_index()
    )

    df_weekday_mean = (
        df.groupby('weekday')[selected_cols]
          .mean(numeric_only=True)
          .reset_index()
    )

    return df_hour_mean, df_weekday_mean


def plot_bar_chart_hour(df_hour: pd.DataFrame,
                        selected_cols: list) -> px.bar:
    """
    Create a bar chart of average values by hour.

    Parameters
    ----------
    df_hour : pd.DataFrame
        DataFrame containing the mean values grouped by hour.
    selected_cols : list
        List of columns to visualize.

    Returns
    -------
    px.bar
        A Plotly Express bar chart.
    """
    if df_hour.empty:
        return px.bar(title="No data available.")

    long_df = df_hour.melt(
        id_vars="stunde",
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    fig = px.bar(
        long_df,
        x="stunde",
        y="Value",
        color="Column",
        barmode="group",
    )
    fig.update_layout(xaxis_title="Hour", yaxis_title="Average")
    return fig


def plot_bar_chart_weekday(df_weekday: pd.DataFrame,
                           selected_cols: list) -> px.bar:
    """
    Create a bar chart of average values by weekday.

    Parameters
    ----------
    df_weekday : pd.DataFrame
        DataFrame containing the mean values grouped by weekday.
    selected_cols : list
        List of columns to visualize.

    Returns
    -------
    px.bar
        A Plotly Express bar chart.
    """
    if df_weekday.empty:
        return px.bar(title="No data available.")

    df_weekday['Weekday'] = df_weekday['weekday'].map(WEEKDAY_MAP)
    long_df = df_weekday.melt(
        id_vars=["weekday", "Weekday"],
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    fig = px.bar(
        long_df,
        x="Weekday",
        y="Value",
        color="Column",
        barmode="group",
    )
    fig.update_layout(xaxis_title="Weekday", yaxis_title="Average")
    return fig


def prepare_data_for_line_plots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for line plots (absolute values).
    Ensures 'tag_dt' is valid; summation is handled in the plotting function.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame.

    Returns
    -------
    pd.DataFrame
        The valid DataFrame with parsed dates.
    """
    if df.empty:
        return pd.DataFrame()

    df['tag_dt'] = pd.to_datetime(df['tag'], dayfirst=True, errors='coerce')
    df = df[~df['tag_dt'].isna()].copy()
    return df


def determine_label_mode(stations_included: list, stations_in_df: list) -> str:
    """
    Decide whether to show 'all', 'single', or 'multi' lines/bars.

    Parameters
    ----------
    stations_included : list
        Stations explicitly included by the user.
    stations_in_df : list
        Stations present in the filtered DataFrame.

    Returns
    -------
    str
        One of ["all", "single", "multi"].
    """
    if len(stations_included) == 0:
        return "all"
    elif len(stations_in_df) == 1:
        return "single"
    else:
        return "multi"


#########################################
# Plot Functions for Line and Bar Charts
#########################################

def plot_line_chart_hour(df_line: pd.DataFrame,
                         selected_cols: list,
                         station_label_map: dict,
                         stations_included: list) -> px.line:
    """
    Create a line chart for absolute values by hour (0..23).

    Parameters
    ----------
    df_line : pd.DataFrame
        The data prepared for line plots.
    selected_cols : list
        Columns to sum for plotting.
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    title : str
        Chart title.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.line
        A Plotly Express line chart.
    """
    if df_line.empty:
        return px.line(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["stunde"]
    elif label_mode == "single":
        group_cols = ["stunde"]
    else:
        group_cols = ["mq_name", "stunde"]

    df_sums = df_line.groupby(group_cols)[selected_cols].sum(numeric_only=True).reset_index()
    long_df = df_sums.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.line(
        long_df,
        x="stunde",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(xaxis_title="Hour", yaxis_title="Absolute Value")
    return fig


def plot_line_chart_weekday(df_line: pd.DataFrame,
                            selected_cols: list,
                            station_label_map: dict,
                            stations_included: list) -> px.line:
    """
    Create a line chart for absolute values by weekday.

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line plots.
    selected_cols : list
        Columns to sum for plotting.
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.line
        A Plotly Express line chart.
    """
    if df_line.empty:
        return px.line(title="No data available.")

    df_line['weekday'] = df_line['tag_dt'].dt.weekday
    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["weekday"]
    elif label_mode == "single":
        group_cols = ["weekday"]
    else:
        group_cols = ["mq_name", "weekday"]

    df_sums = df_line.groupby(group_cols)[selected_cols].sum(numeric_only=True).reset_index()
    df_sums["Weekday"] = df_sums["weekday"].map(WEEKDAY_MAP)

    long_df = df_sums.melt(
        id_vars=group_cols + ["Weekday"],
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.line(
        long_df,
        x="Weekday",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(xaxis_title="Weekday", yaxis_title="Absolute Value")
    return fig


#########################################
# Daily (Date-based) Plot Functions
#########################################

def plot_bar_chart_date(df_line: pd.DataFrame,
                        selected_cols: list,
                        station_label_map: dict,
                        stations_included: list) -> px.bar:
    """
    Bar chart showing absolute values by day (tag_dt).

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to plot (summed).
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.bar
        A grouped bar chart.
    """
    if df_line.empty:
        return px.bar(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_sums = df_line.groupby(group_cols)[selected_cols].sum(numeric_only=True).reset_index()
    long_df = df_sums.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.bar(
        long_df,
        x="tag_dt",
        y="Value",
        color="Station_Column",
        barmode="group",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Absolute Value")
    return fig


def plot_boxplot_chart_date(df_line: pd.DataFrame,
                            selected_cols: list,
                            station_label_map: dict,
                            stations_included: list) -> px.box:
    """
    Create a boxplot of daily summed values for each station/column combination.

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to plot (summed).
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.box
        A Plotly Express boxplot.
    """
    if df_line.empty:
        return px.box(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_sums = df_line.groupby(group_cols)[selected_cols].sum(numeric_only=True).reset_index()

    long_df = df_sums.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )

    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.box(
        long_df,
        x="Station_Column",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(
        yaxis_title="Absolute Value",
    )
    # Hide repetitive labels on x-axis
    fig.update_xaxes(showticklabels=False)
    return fig


def plot_peak_bar_chart_date(df_line: pd.DataFrame,
                             selected_cols: list,
                             station_label_map: dict,
                             stations_included: list) -> px.bar:
    """
    Bar chart showing the peak (max) values by day (tag_dt).

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to take the maximum for each day.
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.bar
        A bar chart of peak daily values.
    """
    if df_line.empty:
        return px.bar(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_maxs = df_line.groupby(group_cols)[selected_cols].max(numeric_only=True).reset_index()
    long_df = df_maxs.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.bar(
        long_df,
        x="tag_dt",
        y="Value",
        color="Station_Column",
        barmode="group",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Absolute Value")
    return fig


def plot_line_chart_date(df_line: pd.DataFrame,
                         selected_cols: list,
                         station_label_map: dict,
                         stations_included: list) -> px.line:
    """
    Line chart showing absolute values by day (tag_dt).

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to plot (summed).
    station_label_map : dict
        Mapping of station ID to a descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.line
        A Plotly Express line chart.
    """
    if df_line.empty:
        return px.line(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_sums = df_line.groupby(group_cols)[selected_cols].sum(numeric_only=True).reset_index()
    long_df = df_sums.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.line(
        long_df,
        x="tag_dt",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Absolute Value")
    return fig


def plot_peak_volume_line_chart_date(df_line: pd.DataFrame,
                                     selected_cols: list,
                                     station_label_map: dict,
                                     stations_included: list) -> px.line:
    """
    Line chart showing peak (max) values by day (tag_dt).

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to take maximum for each day.
    station_label_map : dict
        Mapping of station ID to a descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.line
        A Plotly Express line chart.
    """
    if df_line.empty:
        return px.line(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_maxs = df_line.groupby(group_cols)[selected_cols].max(numeric_only=True).reset_index()
    long_df = df_maxs.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.line(
        long_df,
        x="tag_dt",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Absolute Value")
    return fig


#########################################
# New: Additional Plot Functions
#########################################

def plot_boxplot_peak_volume(df_line: pd.DataFrame,
                             selected_cols: list,
                             station_label_map: dict,
                             stations_included: list) -> px.box:
    """
    Create a boxplot of daily peak (max) values for each station/column combination.

    Parameters
    ----------
    df_line : pd.DataFrame
        Data prepared for line/bar plots (with valid 'tag_dt').
    selected_cols : list
        The numeric columns to plot (max values).
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        Stations explicitly included by the user.

    Returns
    -------
    px.box
        A Plotly Express boxplot.
    """
    if df_line.empty:
        return px.box(title="No data available.")

    stations_in_df = df_line['mq_name'].unique().tolist()
    label_mode = determine_label_mode(stations_included, stations_in_df)

    if label_mode == "all":
        group_cols = ["tag_dt"]
    elif label_mode == "single":
        group_cols = ["tag_dt"]
    else:
        group_cols = ["mq_name", "tag_dt"]

    df_maxs = df_line.groupby(group_cols)[selected_cols].max(numeric_only=True).reset_index()
    long_df = df_maxs.melt(
        id_vars=group_cols,
        value_vars=selected_cols,
        var_name="Column",
        value_name="Value"
    )
    if 'mq_name' not in long_df.columns:
        long_df['mq_name'] = None

    def make_label(row):
        if label_mode == "all":
            return f"Total - {row['Column']}"
        elif label_mode == "single":
            st_id = stations_in_df[0]
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"
        else:
            st_id = row['mq_name']
            st_label = station_label_map.get(st_id, st_id)
            return f"{st_label} - {row['Column']}"

    long_df["Station_Column"] = long_df.apply(make_label, axis=1)

    fig = px.box(
        long_df,
        x="Station_Column",
        y="Value",
        color="Station_Column",
    )
    fig.update_layout(yaxis_title="Peak Value")
    fig.update_xaxes(showticklabels=False)
    return fig


def plot_scatter_speed_vs_volume(df: pd.DataFrame,
                                 q_column: str,
                                 v_column: str,
                                 station_label_map: dict,
                                 stations_included: list) -> px.scatter:
    """
    Create a scatter plot with speed on the x-axis and volume on the y-axis.

    If no stations are selected, the data is aggregated (averaged) across all stations
    per hour. Otherwise, each station's data is plotted separately.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame containing the relevant data.
    q_column : str
        Column name for volume (e.g., 'q_kfz_mq_hr').
    v_column : str
        Column name for speed (e.g., 'v_kfz_mq_hr').
    station_label_map : dict
        Mapping of station ID to a more descriptive label.
    stations_included : list
        List of stations explicitly included by the user.

    Returns
    -------
    px.scatter
        A Plotly Express scatter plot.
    """
    if df.empty:
        return px.scatter(title="No data available.")

    if len(stations_included) == 0:
        # Aggregate data: group by hour and compute mean values for q and v
        df_agg = df.groupby('stunde')[[q_column, v_column]].mean().reset_index()
        fig = px.scatter(
            df_agg,
            x=v_column,
            y=q_column,
            hover_data=['stunde'],
            labels={v_column: "Speed", q_column: "Volume"}
        )
    else:
        # Filter data to selected stations
        df_filtered = df[df['mq_name'].isin(stations_included)].copy()
        df_filtered['Station_Label'] = df_filtered['mq_name'].apply(lambda x: station_label_map.get(x, x))
        fig = px.scatter(
            df_filtered,
            x=v_column,
            y=q_column,
            color='Station_Label',
            hover_data=['stunde'],
            labels={v_column: "Speed", q_column: "Volume"}
        )
    fig.update_layout(xaxis_title="Speed", yaxis_title="Volume")
    return fig


#########################################
# Streamlit App Main Function
#########################################

def main():
    """
    Main function to run the Streamlit application.

    1. Loads and concatenates Messquerschnitt data for the selected year.
    2. Provides sidebar widgets for:
       - Year selection
       - Weekdays filter
       - Hour range filter
       - Minimum hours per day
       - Stations to include/exclude
       - Dropdown for selecting the measurement pair for the scatter plot.
       - Multiple date ranges to ignore
    3. Displays various charts (bar, line, box, scatter) for hourly, weekday, and daily analyses.
    4. Shows missing-data summaries.
    """
    st.title("Interactive Traffic Count Analysis")

    # --------------------
    # Data Year Selection
    # --------------------
    year_selected = st.sidebar.selectbox("Select Data Year", ["2023", "2024"], index=0)
    data_dir = DATA_DIR_2023 if year_selected == "2023" else DATA_DIR_2024

    st.write(f"**Selected Year:** {year_selected}")
    st.write(f"**Data Directory:** {data_dir}")

    # --------------------
    # Load Data
    # --------------------
    with st.spinner("Loading Messquerschnitt data..."):
        df_data = load_messquerschnitt_data(data_dir)
    if df_data.empty:
        st.error("No Messquerschnitt data found!")
        return
    st.success(f"{len(df_data):,} rows of Messquerschnitt data loaded.")

    with st.spinner("Loading Stammdaten..."):
        df_stamm = load_stammdaten(data_dir)
    if df_stamm.empty:
        st.warning("No Stammdaten found.")
        df_stamm = pd.DataFrame(columns=["MQ_KURZNAME", "STRASSE"])
    else:
        st.info(f"Stammdaten loaded. {len(df_stamm):,} rows.")

    # --------------------
    # Build station label map
    # --------------------
    stamm_unique = df_stamm.drop_duplicates(subset=["MQ_KURZNAME"]).copy()
    stamm_unique["STRASSE"] = stamm_unique["STRASSE"].fillna("Unknown")

    station_label_map = {}
    for _, row in stamm_unique.iterrows():
        mq = row["MQ_KURZNAME"]
        street = row["STRASSE"]
        if pd.notna(mq):
            station_label_map[mq] = f"{street} ({mq})"

    # For stations not in Stammdaten
    all_mq_in_data = df_data["mq_name"].unique().tolist()
    for mq in all_mq_in_data:
        if mq not in station_label_map:
            station_label_map[mq] = f"{mq} (No Stammdaten)"

    all_stations_sorted = sorted(all_mq_in_data)

    # --------------------
    # Sidebar: Filters & Settings
    # --------------------
    available_weekdays = list(WEEKDAY_MAP.values())  # ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    selected_weekdays = st.sidebar.multiselect(
        "Select Weekdays",
        options=available_weekdays,
        default=["Tu", "We", "Th"]
    )

    start_hour, end_hour = st.sidebar.slider(
        "Select Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    min_hours_per_day = st.sidebar.number_input(
        "Minimum Hours per Day",
        step=1,
        min_value=0,
        max_value=24,
        value=18
    )

    min_days_per_year = st.sidebar.number_input(
        "Minimum Days per Year",
        step=1,
        min_value=0,
        max_value=365,
        value=20
    )

    stations_included = st.sidebar.multiselect(
        "Stations to Include (optional)",
        options=all_stations_sorted,
        default=[],
        format_func=lambda x: station_label_map[x] if x in station_label_map else x
    )

    stations_excluded = st.sidebar.multiselect(
        "Stations to Exclude (optional)",
        options=all_stations_sorted,
        default=[],
        format_func=lambda x: station_label_map[x] if x in station_label_map else x
    )

    cols_selected = st.sidebar.multiselect(
        "Measurement Columns",
        options=MEASUREMENT_COLS,
        default=["q_kfz_mq_hr"]
    )
    if not cols_selected:
        st.warning("Please select at least one measurement column!")
        return

    # --------------------
    # Sidebar: Dropdown for Scatter Plot Measurement Pair
    # --------------------
    scatter_option = st.sidebar.selectbox(
        "Select Measurement Pair for Scatter Plot",
        options=[
            "KFZ: q_kfz_mq_hr vs. v_kfz_mq_hr",
            "PKW: q_pkw_mq_hr vs. v_pkw_mq_hr",
            "LKW: q_lkw_mq_hr vs. v_lkw_mq_hr"
        ]
    )
    if scatter_option.startswith("KFZ"):
        q_scatter = "q_kfz_mq_hr"
        v_scatter = "v_kfz_mq_hr"
    elif scatter_option.startswith("PKW"):
        q_scatter = "q_pkw_mq_hr"
        v_scatter = "v_pkw_mq_hr"
    else:
        q_scatter = "q_lkw_mq_hr"
        v_scatter = "v_lkw_mq_hr"

    # --------------------
    # Exclude Multiple Date Ranges
    # --------------------
    st.sidebar.markdown("### Ignore Date Ranges")
    st.sidebar.write("Add multiple date ranges to remove from the dataset. Useful for ignoring invalid/outlier periods.")

    date_ranges_input = st.sidebar.text_area(
        "Enter ranges in format: YYYY-MM-DD to YYYY-MM-DD, ...",
        value="2023-05-04 to 2023-05-11, 2023-05-15 to 2023-05-15, 2023-06-21 to 2023-06-22, 2023-10-03 to 2023-10-03, 2023-09-12 to 2023-09-12, 2023-06-08 to 2023-06-08, 2023-05-18 to 2023-05-18, 2023-03-08 to 2023-03-08, 2023-02-15 to 2023-02-15",
        height=250
    )

    excluded_ranges = []
    if date_ranges_input.strip():
        for rng in date_ranges_input.split(","):
            rng = rng.strip()
            if " to " in rng:
                start_str, end_str = rng.split(" to ", 1)
                try:
                    start_date = pd.to_datetime(start_str.strip())
                    end_date = pd.to_datetime(end_str.strip())
                    if start_date <= end_date:
                        excluded_ranges.append((start_date, end_date))
                    else:
                        st.sidebar.warning(f"Invalid range: {rng} (start > end)")
                except ValueError:
                    st.sidebar.warning(f"Cannot parse date range: {rng}")

    # --------------------
    # Filter Data
    # --------------------
    df_data['tag_dt'] = pd.to_datetime(df_data['tag'], dayfirst=True, errors='coerce')
    df_data = df_data[~df_data['tag_dt'].isna()].copy()

    # 1) Filter by included stations (if any)
    if len(stations_included) > 0:
        df_data = df_data[df_data['mq_name'].isin(stations_included)]

    # 2) Filter out excluded stations (if any)
    if len(stations_excluded) > 0:
        df_data = df_data[~df_data['mq_name'].isin(stations_excluded)]

    final_stations = df_data['mq_name'].unique().tolist()

    # 3) Weekday filter
    df_data['weekday'] = df_data['tag_dt'].dt.weekday
    selected_weekday_nums = [WEEKDAY_REVERSE_MAP[w] for w in selected_weekdays]
    df_data = df_data[df_data['weekday'].isin(selected_weekday_nums)]

    # 4) Hour range filter
    df_data = df_data[(df_data['stunde'] >= start_hour) & (df_data['stunde'] <= end_hour)]

    # 5) Exclude the specified date ranges (NEW FEATURE)
    for (start_d, end_d) in excluded_ranges:
        df_data = df_data[
            ~((df_data['tag_dt'] >= start_d) & (df_data['tag_dt'] <= end_d))
        ]

    # 6) Min number of hours per day
    df_data = df_data.groupby(['mq_name', 'tag']).filter(lambda x: len(x) > min_hours_per_day)

    # 7) Min number of days per year
    df_data["year"] = df_data["tag_dt"].dt.year
    day_counts_per_mq_year = df_data.groupby(["mq_name", "year"])["tag_dt"].nunique()
    valid_mq_names = day_counts_per_mq_year[day_counts_per_mq_year > min_days_per_year].index.get_level_values(
        0).unique()
    df_data = df_data[df_data["mq_name"].isin(valid_mq_names)]

    if df_data.empty:
        st.warning("No data left after applying filters!")
        return

    # --------------------
    # Prepare Data
    # --------------------
    df_hour_bar, df_wday_bar = prepare_data_for_bar_plots(df_data, cols_selected)
    df_line = prepare_data_for_line_plots(df_data)

    if len(final_stations) == 0:
        station_names_str = "NO STATIONS"
    elif len(final_stations) == 1:
        single_station = final_stations[0]
        station_names_str = station_label_map.get(single_station, single_station)
    else:
        station_names_str = f"{len(final_stations)} stations selected"

    st.write(f"**Selected Stations:** {station_names_str}")

    # --------------------
    # Bar Charts (Average)
    # --------------------
    st.subheader("Average Volume per Hour")
    fig_hour_bar = plot_bar_chart_hour(
        df_hour_bar, cols_selected
    )
    st.plotly_chart(fig_hour_bar, use_container_width=True)

    st.subheader("Average Volume per Weekday")
    fig_wday_bar = plot_bar_chart_weekday(
        df_wday_bar, cols_selected
    )
    st.plotly_chart(fig_wday_bar, use_container_width=True)

    # --------------------
    # Line Charts (Absolute)
    # --------------------
    st.subheader("Absolute Volumes per Hour")
    fig_hour_line = plot_line_chart_hour(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_hour_line, use_container_width=True)

    st.subheader("Absolute Volumes per Weekday")
    fig_wday_line = plot_line_chart_weekday(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_wday_line, use_container_width=True)

    # --------------------
    # Daily Charts
    # --------------------
    st.subheader("Absolute Volumes per Day")
    fig_day_bar = plot_bar_chart_date(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_day_bar, use_container_width=True)

    st.subheader("Absolute Volumes per Day")
    fig_day_line = plot_line_chart_date(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_day_line, use_container_width=True)

    # --------------------
    # Boxplots
    # --------------------
    st.subheader("Daily Sums")
    fig_box = plot_boxplot_chart_date(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_box, use_container_width=True, key="boxplot")

    st.subheader("Daily Peak Volume")
    fig_box_peak = plot_boxplot_peak_volume(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_box_peak, use_container_width=True, key="boxplot_peak")

    # --------------------
    # Peak Volume Charts
    # --------------------
    st.subheader("Peak Volume per Day")
    fig_peak_bar = plot_peak_bar_chart_date(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_peak_bar, use_container_width=True, key="peak_bar")

    st.subheader("Peak Volume per Day")
    fig_peak_line = plot_peak_volume_line_chart_date(
        df_line,
        cols_selected,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_peak_line, use_container_width=True, key="peak_line")

    # --------------------
    # Scatter Plot: Speed vs Volume
    # --------------------
    st.subheader("Speed-Volume Fundamental Diagram")
    fig_scatter = plot_scatter_speed_vs_volume(
        df_data,
        q_scatter,
        v_scatter,
        station_label_map,
        stations_included
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------
    # Missing Data Summary
    # --------------------
    st.subheader("Missing Hours per Day per Station)")
    if len(final_stations) == 0:
        st.info("No stations are present after filtering, so missing data is not applicable.")
        return

    if len(stations_included) == 0 and len(stations_excluded) == 0:
        df_missing = get_missing_data_info_cached(df_data, min_hours_per_day)
    else:
        df_missing = get_missing_data_info(df_data, final_stations, min_hours_per_day)

    if df_missing.empty:
        st.success("No missing values detected after the applied filters.")
    else:
        df_missing_display = df_missing.copy()
        df_missing_display["Station_Name"] = df_missing_display["mq_name"].apply(
            lambda x: station_label_map.get(x, x)
        )
        st.dataframe(df_missing_display[["Station_Name", "date", "missing_hours"]], use_container_width=True)

    if len(stations_included) == 0 and len(stations_excluded) == 0:
        df_missing_compact = get_compact_missing_data_cached(df_missing)
    else:
        df_missing_compact = get_compact_missing_data(df_missing)

    st.subheader("Number of Missing Hours per Station)")
    if df_missing_compact.empty:
        st.success("No missing values detected in the filtered dataset.")
    else:
        df_missing_compact["Station_Name"] = df_missing_compact["mq_name"].apply(
            lambda x: station_label_map.get(x, x)
        )
        st.dataframe(df_missing_compact[["Station_Name", "total_missing_hours"]], use_container_width=True)

    # --------------------
    # Days per Station (after all filters)
    # --------------------
    st.subheader("Days per Station")
    df_days_per_station = (
        df_data.groupby("mq_name")["tag_dt"].nunique().reset_index(name="days_count")
    )
    df_days_per_station["Station_Name"] = df_days_per_station["mq_name"].apply(
        lambda x: station_label_map.get(x, x)
    )
    st.dataframe(df_days_per_station[["Station_Name", "days_count"]], use_container_width=True)


if __name__ == "__main__":
    main()
