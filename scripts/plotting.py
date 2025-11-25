import plotly.graph_objects as go 
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def plot_tiepoints(day, day_ds, tiepoints, scan_positions=13):
    # Compute GR for coloring
    GR = (day_ds.TBCH2 - day_ds.TBCH1) / (day_ds.TBCH2 + day_ds.TBCH1)

    fig = go.Figure([
        # ALL observations (colored by GR)
        go.Scatter(
            x=day_ds.TBCH2.values.flatten(),
            y=day_ds.TBCH1.values.flatten(),
            mode='markers', 
            name='OBS',
            marker=dict(
                color=GR.values.flatten(),
                colorscale='Viridis',
                colorbar=dict(title='Gradient Ratio'),
                showscale=True,
                size=7,
            )
        ),

        # FYI tie points (all scans)
        go.Scatter(
            x=[tiepoints["FYI"]["ch2"][s][day-1] for s in range(scan_positions)],
            y=[tiepoints["FYI"]["ch1"][s][day-1] for s in range(scan_positions)],
            mode='markers',
            name='FYI',
            marker=dict(color='cyan', size=8)
        ),

        # MYI tie points (all scans)
        go.Scatter(
            x=[tiepoints["MYI"]["ch2"][s][day-1] for s in range(scan_positions)],
            y=[tiepoints["MYI"]["ch1"][s][day-1] for s in range(scan_positions)],
            mode='markers',
            name='MYI',
            marker=dict(color='magenta', size=8)
        ),

        # Open water tie points
        go.Scatter(
            x=[tiepoints["water"]["ch2"][s][day-1] for s in range(scan_positions)],
            y=[tiepoints["water"]["ch1"][s][day-1] for s in range(scan_positions)],
            mode='markers',
            name='OW',
            marker=dict(color='orange', size=8)
        )
    ])

    # -------------------------------------------------------------
    # 1) Add dashed line for GR = -0.015 (paper’s classification boundary)
    # -------------------------------------------------------------
    g = -0.015
    ratio = (1 - g) / (1 + g)  # TB31 = ratio * TB22 for constant GR
    x_min = np.nanmin(day_ds.TBCH2.values)
    x_max = np.nanmax(day_ds.TBCH2.values)
    x_line = np.array([x_min, x_max])
    y_line = ratio * x_line

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        name=f'GR = {g}',
        line=dict(color='black', dash='dash')
    ))

    # -------------------------------------------------------------
    # 2) Add the ICE LINE from nadir FYI <--> nadir MYI tie-points
    # -------------------------------------------------------------
    nadir = scan_positions // 2   # e.g., for 13 beams → 6

    FYI_x = tiepoints["FYI"]["ch2"][nadir][day-1]
    FYI_y = tiepoints["FYI"]["ch1"][nadir][day-1]
    MYI_x = tiepoints["MYI"]["ch2"][nadir][day-1]
    MYI_y = tiepoints["MYI"]["ch1"][nadir][day-1]

    fig.add_trace(go.Scatter(
        x=[FYI_x, MYI_x],
        y=[FYI_y, MYI_y],
        mode='lines',
        name='Ice line (nadir)',
        line=dict(color='gray', width=3, dash='dot')
    ))

    # -------------------------------------------------------------
    # Labels and layout
    # -------------------------------------------------------------
    fig.update_layout(
        title=f'Tie points (Day {day})',
        xaxis_title='TB 31.65 GHz',
        yaxis_title='TB 22.35 GHz',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig.show()

def plot_geo(ds):
    # Extract variables from the dataset
    lat = ds.LAT
    lon = ds.LON
    tbch1 = ds.TBCH1

    # Create the map with geographic projection
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines and continents
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.gridlines(draw_labels=True)

    # Plot the TBCH1 data
    scatter = ax.scatter(lon, lat, c=tbch1, cmap='viridis', s=10, transform=ccrs.PlateCarree())
    plt.colorbar(scatter, label='TBCH1 (K)')
    plt.title('TBCH1 Temperature Map')
    plt.tight_layout()
    plt.show()

def plot_raw_vs_smoothed(
    tie_raw_dict,
    tie_smoothed_dict,
    category,
    channel,
    hemisphere="north"
):
    """
    Creates a Plotly figure with a slider to pick the scan position (0–12).
    Shows RAW vs SMOOTHED tie points for that scan.
    """

    raw_data    = tie_raw_dict[category][channel]       # 13 lists
    smooth_data = tie_smoothed_dict[category][channel]  # 13 lists

    N_scans = len(raw_data)
    N_days  = len(raw_data[0])
    days    = np.arange(1, N_days + 1)

    # Figure with empty initial trace (will be updated by slider)
    fig = go.Figure()

    # Add all raw & smooth traces, but make only scan 0 visible initially
    for scan in range(N_scans):
        # Raw trace
        fig.add_trace(
            go.Scatter(
                x=days,
                y=raw_data[scan],
                mode="lines",
                name=f"Raw Scan {scan}",
                line=dict(width=2, dash="dot"),
                visible=(scan == 0)
            )
        )
        # Smoothed trace
        fig.add_trace(
            go.Scatter(
                x=days,
                y=smooth_data[scan],
                mode="lines",
                name=f"Smoothed Scan {scan}",
                line=dict(width=3),
                visible=(scan == 0)
            )
        )

    # Slider steps
    steps = []
    for scan in range(N_scans):
        step = {
            "method": "update",
            "label": f"{scan}",
            "args": [
                {"visible": [False] * (2 * N_scans)},  # turn all off
                {"title": f"Scan {scan} — Raw vs Smoothed Tie Points"}
            ],
        }
        # Turn on raw+smoothed for this scan
        step["args"][0]["visible"][2*scan]   = True   # raw trace
        step["args"][0]["visible"][2*scan+1] = True   # smoothed trace
        steps.append(step)

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Scan: "},
        "pad": {"t": 50},
        "steps": steps
    }]

    fig.update_layout(
        sliders=sliders,
        title=(
            f"Raw vs Smoothed Tie-Points ({hemisphere.capitalize()})<br>"
            f"Category: {category}, Channel: {channel}"
        ),
        xaxis_title="Day",
        yaxis_title="Brightness Temperature (K)",
        template="plotly_white",
        height=600
    )

    fig.show()

def plot_sic_side_by_side(day_ds, SIC, day, 
                          projection=ccrs.NorthPolarStereo(),
                          title_prefix="Computed SIC (1-channel)"):

    # =======================================================================
    # 1. Extract raw arrays (2D: time × n13_obs)
    # =======================================================================
    lon = np.array(day_ds.LON)
    lat = np.array(day_ds.LAT)
    sic_1ch = np.array(SIC)

    if lon.shape != lat.shape:
        raise ValueError("LAT and LON must have the same shape.")

    # Match SIC shape to swath if needed
    if sic_1ch.shape != lon.shape:
        try:
            sic_1ch = np.broadcast_to(sic_1ch, lon.shape)
        except Exception:
            sic_1ch = np.full(lon.shape, np.nan)

    # =======================================================================
    # 2. Interpolate ERA5 SIC onto the swath grid (CRITICAL FIX)
    # =======================================================================
    #
    # ERA5 SIC is on a regular lat/lon grid → must be sampled at swath points.
    #
    # This produces a 2D array aligned with (lon, lat).
    #
    # =======================================================================

    if "siconc" in day_ds:
        try:
            siconc_swath = day_ds.siconc.interp(
                LAT=day_ds.LAT,
                LON=day_ds.LON,
                method="nearest"
            )
        except Exception:
            # fallback: ERA5 may have wrong dim names
            siconc_swath = np.full(lon.shape, np.nan)
    else:
        siconc_swath = np.full(lon.shape, np.nan)

    siconc_swath = np.array(siconc_swath)

    # =======================================================================
    # 3. Mask invalid coordinate points
    # =======================================================================
    mask_invalid = np.isnan(lon) | np.isnan(lat)

    sic_flat = sic_1ch[~mask_invalid]
    era5_flat = siconc_swath[~mask_invalid]

    lon_flat = lon[~mask_invalid]
    lat_flat = lat[~mask_invalid]

    # =======================================================================
    # 4. Compute map extent based on valid swath region
    # =======================================================================
    if lon_flat.size > 0:
        lon_min = float(np.nanmin(lon_flat))
        lon_max = float(np.nanmax(lon_flat))
        lat_min = float(np.nanmin(lat_flat))
        lat_max = float(np.nanmax(lat_flat))
    else:
        lon_min, lon_max = -180, 180
        lat_min, lat_max = 60, 90  # fallback for Arctic

    # =======================================================================
    # 5. Create figure + two subplots
    # =======================================================================
    fig, axs = plt.subplots(
        1, 2, figsize=(14, 7),
        subplot_kw={"projection": projection}
    )
    ax1, ax2 = axs

    # =======================================================================
    # LEFT PANEL: computed SIC
    # =======================================================================
    ax1.set_title(f"{title_prefix} — Day {day}", fontsize=12)
    ax1.coastlines(resolution="110m")
    ax1.add_feature(cfeature.LAND, facecolor="lightgray")

    sc1 = ax1.scatter(
        lon_flat, lat_flat, c=sic_flat,
        cmap="viridis", vmin=0, vmax=1,
        s=7, marker="s", transform=ccrs.PlateCarree(), rasterized=True
    )
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label("SIC (0–1)")

    # =======================================================================
    # RIGHT PANEL: ERA5 SIC (interpolated onto swath)
    # =======================================================================
    ax2.set_title(f"ERA5 SIC — Day {day}", fontsize=12)
    ax2.coastlines(resolution="110m")
    ax2.add_feature(cfeature.LAND, facecolor="lightgray")

    sc2 = ax2.scatter(
        lon_flat, lat_flat, c=era5_flat,
        cmap="viridis", vmin=0, vmax=1,
        s=7, marker="s", transform=ccrs.PlateCarree(), rasterized=True
    )
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label("SIC (0–1)")

    # =======================================================================
    # 6. Set map extents
    # =======================================================================
    extent = [lon_min, lon_max, lat_min, lat_max]
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.set_extent(extent, crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.show()