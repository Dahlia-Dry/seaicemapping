# ---------------------------------------------
# 1. PREPARE DAILY GROUPINGS
# ---------------------------------------------
# Convert orbit timestamps to dates (assuming time coordinate exists per orbit)
ds = ds.assign_coords(date=("orbit", pd.to_datetime(ds.time.values).normalize()))

dates = np.unique(ds.date.values)

# Output storage structure:
# tiepoints[category][channel][day][scan_pos]
categories = ["water", "FYI", "MYI"]
tie_raw  = {cat: {ch: [] for ch in ["ch1", "ch2"]} for cat in categories}
tie_smth = {cat: {ch: None for ch in ["ch1", "ch2"]} for cat in categories}

# ---------------------------------------------
# 2. PROCESS EACH DAY IN THE MONTH
# ---------------------------------------------
for day in dates:
    day_ds = ds.sel(orbit=ds.date == day)

    # Create gradient ratio
    GR = (day_ds.ch2_TB - day_ds.ch1_TB) / (day_ds.ch2_TB + day_ds.ch1_TB)

    # ------------------------------------------------------------------
    # SELECT PURE WATER + ICE PIXELS USING ERA5 SIC & TB THRESHOLDS
    # ------------------------------------------------------------------
    
    # Land mask
    ocean_mask = day_ds.lsm == 0

    # Water = ERA5 SIC = 0 + TB limits
    mask_water = (
        (day_ds.sic_era5 == 0) &
        (day_ds.ch1_TB > TB_MIN_WATER) & (day_ds.ch1_TB < TB_MAX_WATER) &
        (day_ds.ch2_TB > TB_MIN_WATER) & (day_ds.ch2_TB < TB_MAX_WATER) &
        ocean_mask
    )

    # Ice = ERA5 SIC > 0.8 + TB limits
    mask_ice = (
        (day_ds.sic_era5 > 0.8) &
        (day_ds.ch1_TB > TB_MIN_ICE) & (day_ds.ch1_TB < TB_MAX_ICE) &
        (day_ds.ch2_TB > TB_MIN_ICE) & (day_ds.ch2_TB < TB_MAX_ICE) &
        ocean_mask
    )

    # Split into FYI / MYI using gradient ratio
    mask_MYI = mask_ice & (GR < GR_THRESHOLD)
    mask_FYI = mask_ice & (GR >= GR_THRESHOLD)

    # ---------------------------------------------
    # 3. COMPUTE DAILY MEAN TIE-POINTS PER SCAN-POSITION (13)
    # ---------------------------------------------
    for scan in range(SCAN_POSITIONS):

        # Extract scan slice
        sub = day_ds.isel(incidence_angle=scan)

        # For each category and channel
        for cat, mask in zip(
            ["water", "FYI", "MYI"],
            [mask_water, mask_FYI, mask_MYI]
        ):
            for ch_name, ch_data in zip(
                ["ch1", "ch2"],
                [sub.ch1_TB, sub.ch2_TB]
            ):
                vals = ch_data.where(mask.isel(incidence_angle=scan), drop=True)
                if vals.size == 0:
                    tie_raw[cat][ch_name].append(np.nan)
                else:
                    tie_raw[cat][ch_name].append(float(vals.mean()))

# ---------------------------------------------
# 4. CONVERT RAW LISTS TO ARRAYS
# ---------------------------------------------
for cat in categories:
    for ch in ["ch1", "ch2"]:
        # Shape: (days * scan_positions)
        arr = np.array(tie_raw[cat][ch]).reshape(len(dates), SCAN_POSITIONS)
        tie_raw[cat][ch] = arr

# ---------------------------------------------
# 5. APPLY ±7-DAY (15-day) SMOOTHING PER SCAN-POSITION
# ---------------------------------------------
def smooth_15day(data):
    # data shape: (days, 13)
    smoothed = np.full_like(data, np.nan, dtype=float)
    for sp in range(data.shape[1]):
        series = pd.Series(data[:, sp])
        smoothed[:, sp] = series.rolling(
            window=15,
            center=True,
            min_periods=1
        ).mean().values
    return smoothed

for cat in categories:
    for ch in ["ch1", "ch2"]:
        tie_smth[cat][ch] = smooth_15day(tie_raw[cat][ch])

# ---------------------------------------------
# tie_smth now contains the smoothed tie-points:
#
# tie_smth["water"]["ch1"] → array(days, 13)
# tie_smth["FYI"]["ch2"]   → array(days, 13)
# tie_smth["MYI"]["ch1"]   → array(days, 13)
#
# Ready for use in SIC algorithm.
# ---------------------------------------------