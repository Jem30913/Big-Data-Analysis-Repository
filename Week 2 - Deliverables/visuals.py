import os
import glob
import re

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType,
    DoubleType, IntegerType,
)

# ──────────────────────────────────────────────
# 0.  CONFIG
# ──────────────────────────────────────────────
DATA_DIR    = "./csv"          
OUTPUT_DIR  = "./output_results"    
import shutil
if os.path.exists(OUTPUT_DIR) and not os.path.isdir(OUTPUT_DIR):
    os.remove(OUTPUT_DIR)  
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = [                     
    "#5b9bd5",   
    "#9b72cf",   
    "#3fbf99",   
    "#e8a838",   
    "#e06b8b",   
    "#4dbdbd",   
    "#a3c46e",   
    "#d4745a",   
]

# ──────────────────────────────────────────────
# 1.  SPARK SESSION
# ──────────────────────────────────────────────
def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("WeatherPipeline")
        .config("spark.sql.session.timeZone", "Asia/Manila")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# ──────────────────────────────────────────────
# 2.  EXPLICIT SCHEMA  
# ──────────────────────────────────────────────
WEATHER_SCHEMA = StructType([
    StructField("station_id",  StringType(),  True),
    StructField("datetime",    StringType(),  True),   
    StructField("lat",         DoubleType(),  True),
    StructField("lon",         DoubleType(),  True),
    StructField("temp",        DoubleType(),  True),
    StructField("pressure",    DoubleType(),  True),
    StructField("humidity",    DoubleType(),  True),
    StructField("wind_speed",  DoubleType(),  True),
    StructField("wind_dir",    DoubleType(),  True),
    StructField("cloud_cover", DoubleType(),  True),
    StructField("visibility",  DoubleType(),  True),
    StructField("rain_3h",     DoubleType(),  True),
])


# ──────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────
def extract_location(filepath: str) -> str:
    """
    Derive a human-readable location label from the filename.
    'path/to/DecodedCamNorte.csv'  →  'CamNorte'
    Falls back to the bare stem if the pattern doesn't match.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"Decoded(.+)", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def load_all_csvs(spark: SparkSession, data_dir: str):
    """
    Load every CSV in *data_dir*, add a `location` column derived from
    the filename, and union all DataFrames into one.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

    frames = []
    for path in csv_files:
        location = extract_location(path)
        df = (
            spark.read
            .option("header", "true")
            .option("nullValue", "")
            .schema(WEATHER_SCHEMA)
            .csv(path)
            .withColumn("location", F.lit(location))
        )
        frames.append(df)
        print(f"  Loaded  {os.path.basename(path)}  →  location='{location}'  "
              f"rows={df.count()}")

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.unionByName(df)

    return combined


# ──────────────────────────────────────────────
# 4.  CLEANING & PREPARATION
# ──────────────────────────────────────────────
NUMERIC_COLS = [
    "temp", "pressure", "humidity",
    "wind_speed", "wind_dir",
    "cloud_cover", "visibility", "rain_3h",
]

def clean(df):
    """
    • Parse datetime string → TimestampType
    • Cast numeric columns (already typed in schema; re-cast for safety)
    • Drop rows where all key measurements are null
    • Fill remaining nulls with 0 for rain_3h (missing = no rain)
    """
    df = df.withColumn(
        "ts",
        F.coalesce(
            F.try_to_timestamp(F.col("datetime"), F.lit("yyyy-MM-dd HH:mm:ss")),
            F.try_to_timestamp(F.col("datetime"), F.lit("yyyy-MM-dd HH:mm")),
            F.try_to_timestamp(F.col("datetime")),   
        ),
    )

    for c in NUMERIC_COLS:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    df = df.dropna(
        how="all",
        subset=["temp", "pressure", "humidity", "wind_speed"],
    )

    df = df.fillna({"rain_3h": 0.0})

    df = df.withColumn("date", F.to_date(F.col("ts")))

    return df.cache()  


# ──────────────────────────────────────────────
# 5.  AGGREGATIONS
# ──────────────────────────────────────────────
def agg_monthly_temp(df):
    """Monthly average temperature per location – outliers removed first."""
    filtered = df.filter((F.col("temp") >= 10) & (F.col("temp") <= 45))
    return (
        filtered
        .withColumn("month", F.date_trunc("month", F.col("ts")))
        .groupBy("location", "month")
        .agg(
            F.round(F.avg("temp"), 2).alias("avg_temp"),
            F.round(F.min("temp"), 2).alias("min_temp"),
            F.round(F.max("temp"), 2).alias("max_temp"),
        )
        .orderBy("location", "month")
    )

def agg_daily_temp(df):
    """Daily average temperature per location (kept for CSV export)."""
    filtered = df.filter((F.col("temp") >= 10) & (F.col("temp") <= 45))
    return (
        filtered.groupBy("location", "date")
        .agg(F.round(F.avg("temp"), 2).alias("avg_temp"))
        .orderBy("location", "date")
    )

def agg_avg_humidity(df):
    """Average humidity per location."""
    return (
        df.groupBy("location")
        .agg(F.round(F.avg("humidity"), 2).alias("avg_humidity"))
        .orderBy("avg_humidity", ascending=False)
    )

def agg_total_rain(df):
    """Total rainfall per location."""
    return (
        df.groupBy("location")
        .agg(F.round(F.sum("rain_3h"), 2).alias("total_rain_mm"))
        .orderBy("total_rain_mm", ascending=False)
    )

def agg_wind_trend(df):
    """Average wind speed per location per day."""
    return (
        df.groupBy("location", "date")
        .agg(F.round(F.avg("wind_speed"), 2).alias("avg_wind_speed"))
        .orderBy("location", "date")
    )

def run_aggregations(df):
    print("\n[Aggregations]")
    monthly_temp = agg_monthly_temp(df)
    daily_temp   = agg_daily_temp(df)
    avg_humidity = agg_avg_humidity(df)
    total_rain   = agg_total_rain(df)
    wind_trend   = agg_wind_trend(df)

    for name, sdf in [
        ("monthly_avg_temp", monthly_temp),
        ("daily_avg_temp",   daily_temp),
        ("avg_humidity",     avg_humidity),
        ("total_rain",       total_rain),
        ("wind_trend",       wind_trend),
    ]:
        out = os.path.join(OUTPUT_DIR, f"{name}.csv")
        sdf.toPandas().to_csv(out, index=False)
        print(f"  Saved → {out}")

    return monthly_temp, avg_humidity, total_rain, wind_trend


# ──────────────────────────────────────────────
# 6.  VISUALIZATIONS  —  dark minimalist theme
# ──────────────────────────────────────────────

BG      = "#1c1c1c"   
PANEL   = "#252525"   
GRID_C  = "#383838"  
TEXT_C  = "#e0e0e0"   
SUBTEXT = "#909090"   

def _apply_dark_style(fig, axes_list):
    """Push dark background + clean typography to every axes in the figure."""
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_C, labelsize=9, length=3)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        ax.title.set_color(TEXT_C)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=GRID_C, linestyle="--", linewidth=0.6, alpha=0.7)


def _colour_map(locations):
    return {loc: PALETTE[i % len(PALETTE)] for i, loc in enumerate(locations)}


def plot_temp_trends(monthly_temp_sdf):
    import numpy as np

    pdf = monthly_temp_sdf.toPandas()
    pdf["month"] = pd.to_datetime(pdf["month"])

    locations = sorted(pdf["location"].unique())
    cmap      = _colour_map(locations)
    n         = len(locations)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(22, 3.4 * n),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if n == 1:
        axes = [axes]

    _apply_dark_style(fig, axes)
    fig.suptitle(
        "Monthly Average Temperature by Location (°C)",
        fontsize=15, fontweight="bold", color=TEXT_C,
        x=0.5, y=1.005,
    )

    for ax, loc in zip(axes, locations):
        sub   = pdf[pdf["location"] == loc].sort_values("month").copy()
        color = cmap[loc]


        ax.fill_between(sub["month"], sub["min_temp"], sub["max_temp"],
                        alpha=0.15, color=color, linewidth=0)
        ax.plot(sub["month"], sub["avg_temp"],
                color=color, linewidth=2.0, solid_capstyle="round")
        ax.fill_between(sub["month"], sub["avg_temp"],
                        sub["avg_temp"].min() - 0.5,
                        alpha=0.12, color=color, linewidth=0)
        i_max = sub["avg_temp"].idxmax()
        i_min = sub["avg_temp"].idxmin()
        for idx, offset, va in [(i_max, 9, "bottom"), (i_min, -9, "top")]:
            ax.scatter(sub.loc[idx, "month"], sub.loc[idx, "avg_temp"],
                       s=50, color=color, zorder=5, linewidths=0)
            ax.annotate(
                f"{sub.loc[idx, 'avg_temp']:.1f}°",
                xy=(sub.loc[idx, "month"], sub.loc[idx, "avg_temp"]),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=va, fontsize=8,
                color="#ffffff", fontweight="bold",
            )

        ax.text(0.012, 0.82, loc, transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                color=color, alpha=0.95)

        lo = max(15, sub["avg_temp"].min() - 2)
        hi = sub["avg_temp"].max() + 2.5
        ax.set_ylim(lo, hi)
        ax.set_ylabel("°C", fontsize=9, labelpad=6, color=SUBTEXT)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].tick_params(axis="x", rotation=35, labelsize=9, colors=TEXT_C)
    axes[-1].set_xlabel("Month", fontsize=10, labelpad=8, color=TEXT_C)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "temp_trends.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved → {path}")


def plot_humidity_bar(avg_humidity_sdf):
    pdf    = avg_humidity_sdf.toPandas().sort_values("avg_humidity")
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(pdf))]

    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_dark_style(fig, [ax])

    bars = ax.barh(
        pdf["location"], pdf["avg_humidity"],
        color=colors, edgecolor="none", height=0.52,
    )

    x_max = pdf["avg_humidity"].max()
    for bar, val in zip(bars, pdf["avg_humidity"]):
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} %",
            va="center", ha="left",
            fontsize=10, fontweight="bold", color="#ffffff",
        )

    for rank, (bar, val) in enumerate(zip(bars, pdf["avg_humidity"]), 1):
        ax.text(
            0.6,
            bar.get_y() + bar.get_height() / 2,
            f"#{len(pdf) - rank + 1}",
            va="center", ha="left",
            fontsize=8, color=BG, fontweight="bold", alpha=0.6,
        )

    ax.set_title("Average Relative Humidity by Location",
                 fontsize=14, fontweight="bold", pad=14, color=TEXT_C)
    ax.set_xlabel("Humidity (%)", fontsize=10, labelpad=8, color=TEXT_C)
    ax.set_xlim(0, x_max + 10)
    ax.tick_params(axis="y", labelsize=10, colors=TEXT_C)
    ax.grid(axis="x", color=GRID_C, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=2)
    path = os.path.join(OUTPUT_DIR, "avg_humidity.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved → {path}")

def plot_wind_vs_temp(df_spark):
    import numpy as np
    import pyspark.sql.functions as SF

    binned = (
        df_spark
        .select("location", "wind_speed", "temp")
        .filter(
            (SF.col("wind_speed") >= 0) & (SF.col("wind_speed") <= 25) &
            (SF.col("temp") >= 10)      & (SF.col("temp") <= 45)
        )
        .dropna()
        .withColumn("ws_bin", (SF.col("wind_speed") / 0.5).cast("int") * 0.5)
        .groupBy("location", "ws_bin")
        .agg(
            SF.avg("temp").alias("avg_temp"),
            SF.count("*").alias("n"),
        )
        .toPandas()
    )

    locations = sorted(binned["location"].unique())
    cmap      = _colour_map(locations)
    ncols     = 4
    nrows     = -(-len(locations) // ncols)   

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(22, 5.5 * nrows),
        sharey=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.32},
    )
    all_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    _apply_dark_style(fig, all_axes)

    fig.suptitle(
        "Wind Speed vs Temperature — Binned Average  (dot size ∝ frequency)",
        fontsize=14, fontweight="bold", color=TEXT_C, y=1.02,
    )

    for i, loc in enumerate(locations):
        ax    = all_axes[i]
        sub   = binned[binned["location"] == loc].sort_values("ws_bin")
        color = cmap[loc]

        s_raw  = sub["n"].clip(upper=sub["n"].quantile(0.95))
        s_norm = (s_raw / s_raw.max()) * 200 + 15

        sc = ax.scatter(
            sub["ws_bin"], sub["avg_temp"],
            s=s_norm, color=color,
            alpha=0.75, edgecolors="none", zorder=3,
        )

        if len(sub) > 4:
            z = np.polyfit(sub["ws_bin"], sub["avg_temp"], 1)
            xs = np.linspace(sub["ws_bin"].min(), sub["ws_bin"].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs),
                    color="#cccccc", linewidth=1.4, linestyle="--", alpha=0.55)

        ax.set_title(loc, fontsize=11, fontweight="bold",
                     color=color, pad=7)
        ax.set_xlabel("Wind Speed (m/s)", fontsize=9,
                      labelpad=6, color=TEXT_C)
        ax.set_ylabel("Avg Temp (°C)", fontsize=9,
                      labelpad=6, color=TEXT_C)
        ax.tick_params(labelsize=8, colors=TEXT_C)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=False))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))

    for j in range(len(locations), len(all_axes)):
        all_axes[j].set_visible(False)

    path = os.path.join(OUTPUT_DIR, "wind_vs_temp.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved → {path}")

def plot_correlation_heatmap(df_spark):
    import numpy as np
    import pyspark.sql.functions as SF

    cols = NUMERIC_COLS
    n    = len(cols)

    LABELS = {
        "temp":        "Temp",
        "pressure":    "Pressure",
        "humidity":    "Humidity",
        "wind_speed":  "Wind Speed",
        "wind_dir":    "Wind Dir",
        "cloud_cover": "Cloud Cover",
        "visibility":  "Visibility",
        "rain_3h":     "Rainfall",
    }

    corr_vals = {}
    for c in cols:
        row = []
        for d in cols:
            val = df_spark.select(SF.corr(c, d)).collect()[0][0]
            row.append(round(val, 3) if val is not None else 0.0)
        corr_vals[c] = row

    corr_matrix = pd.DataFrame(corr_vals, index=cols)
    corr_matrix.columns = [LABELS[c] for c in cols]
    corr_matrix.index   = [LABELS[c] for c in cols]

    mask_arr = np.zeros((n, n), dtype=bool)
    mask_arr[np.triu_indices_from(mask_arr, k=1)] = True
    mask = pd.DataFrame(mask_arr,
                        index=corr_matrix.index,
                        columns=corr_matrix.columns)

    fig, ax = plt.subplots(figsize=(12, 10))
    _apply_dark_style(fig, [ax])

    sns.heatmap(
        corr_matrix, mask=mask,
        annot=True, fmt=".2f",
        linewidths=1.2, linecolor=BG,
        cmap="mako",
        center=0, vmin=-1, vmax=1,
        ax=ax,
        annot_kws={"size": 10, "weight": "bold", "color": "#ffffff"},
        square=True,
        cbar_kws={"shrink": 0.72, "pad": 0.02},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_C, labelsize=9)
    cbar.outline.set_edgecolor(GRID_C)

    ax.set_title("Weather Variable Correlation Matrix",
                 fontsize=14, fontweight="bold",
                 color=TEXT_C, pad=18)
    ax.tick_params(axis="x", rotation=30, labelsize=10,
                   colors=TEXT_C, length=0)
    ax.tick_params(axis="y", rotation=0,  labelsize=10,
                   colors=TEXT_C, length=0)

    fig.tight_layout(pad=2)
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart saved → {path}")


def run_visualizations(monthly_temp, avg_humidity, _total_rain, _wind_trend, raw_df):
    print("\n[Visualizations]")
    plot_temp_trends(monthly_temp)  
    plot_humidity_bar(avg_humidity)
    plot_wind_vs_temp(raw_df)
    plot_correlation_heatmap(raw_df)


# ──────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Philippine Weather Data Pipeline")
    print("=" * 60)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n[Loading CSVs from '{DATA_DIR}']")
    raw = load_all_csvs(spark, DATA_DIR)

    print("\n[Cleaning]")
    clean_df = clean(raw)
    print(f"  Total rows after cleaning: {clean_df.count():,}")

    monthly_temp, avg_humidity, total_rain, wind_trend = run_aggregations(clean_df)

    run_visualizations(monthly_temp, avg_humidity, total_rain, wind_trend, clean_df)

    print("\n[Sample Aggregations]")
    print("-- Daily Avg Temp (top 10) --")
    monthly_temp.show(10, truncate=False)

    print("-- Avg Humidity per Location --")
    avg_humidity.show(truncate=False)

    print("-- Total Rainfall per Location --")
    total_rain.show(truncate=False)

    clean_df.unpersist()
    spark.stop()
    print("\nPipeline complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
