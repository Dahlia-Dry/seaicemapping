# this script aggregates SCAMS colocated netCDF files into daily files for March 1976
# author: Dahlia Dry
# w/assistance from GitHub Copilot

import netCDF4 as netcdf
from netCDF4 import num2date, date2num
from pathlib import Path
import numpy as np
import pandas as pd

# PARAMETERS-------------------------------------------------   
input_dir = Path("SCAMS_colocated")  
out_dir = Path("march1976_daily")
out_dir.mkdir(exist_ok=True)
vars_of_interest = ['LAT','LON','TBCH1','TBCH2','lsm','siconc']
#-------------------------------------------------------------

files = sorted(input_dir.glob("*.nc"))

def times_from_var(time_var):
    vals = np.asarray(time_var[:]).ravel()
    if vals.size == 0:
        return pd.to_datetime([])
    units = getattr(time_var, "units", None)
    cal = getattr(time_var, "calendar", "standard")
    if units:
        try:
            dt = num2date(vals, units=units, calendar=cal)
            return pd.to_datetime(dt)
        except Exception:
            return pd.to_timedelta(vals, unit="D") + pd.Timestamp("1970-01-01")
    else:
        return pd.to_timedelta(vals, unit="D") + pd.Timestamp("1970-01-01")

# iterate days
days = pd.date_range("1976-03-01","1976-03-31",freq="D")
time_units_out = "seconds since 1970-01-01 00:00:00"
calendar_out = "standard"

for day in days:
    day_norm = pd.to_datetime(day).normalize()
    matches = []  # list of (path, times(pd.DatetimeIndex))
    for f in files:
        try:
            with netcdf.Dataset(str(f), "r") as src:
                if "Time" not in src.variables:
                    continue
                tvals = times_from_var(src.variables["Time"])
                if tvals.size == 0:
                    continue
                # choose file if any time sample falls on the target day
                if np.any(tvals.normalize() == day_norm):
                    matches.append((f, tvals))
        except Exception as e:
            print(f"skip {f.name}: {e}")

    if not matches:
        print(f"{day.date()}: no files")
        continue

    # Gather total number of time records to write
    time_slices = []
    total_time = 0
    for f, tvals in matches:
        n = len(tvals)
        time_slices.append((f, total_time, n, tvals))
        total_time += n

    out_file = out_dir / f"march1976_{day.strftime('%Y%m%d')}.nc"
    print(f"Writing {out_file.name}: {len(matches)} file(s), {total_time} time records")

    # inspect first match to infer spatial dims for each var when possible
    var_meta = {}
    for v in vars_of_interest:
        var_meta[v] = {"found": False, "shape": None, "dtype": None, "fill": None}

    for f, _, _, _ in time_slices:
        with netcdf.Dataset(str(f), "r") as src:
            for v in vars_of_interest:
                if v == "Time":
                    continue
                if v in src.variables and not var_meta[v]["found"]:
                    src_var = src.variables[v]
                    src_shape = src_var.shape
                    # if var has a time dim (first dim) remove it to get spatial shape
                    if src_shape and src_var.dimensions[0].lower() in ("time","record","t"):
                        spatial_shape = src_shape[1:]
                        spatial_dims = src_var.dimensions[1:]
                    else:
                        spatial_shape = src_shape
                        spatial_dims = src_var.dimensions
                    var_meta[v].update({
                        "found": True,
                        "shape": tuple(spatial_shape),
                        "dims": tuple(spatial_dims),
                        "dtype": getattr(src_var, "datatype", "f4"),
                        "fill": getattr(src_var, "_FillValue", None),
                        "attrs": {a: getattr(src_var, a) for a in getattr(src_var, "ncattrs", lambda: [])()}
                    })

    # create output file and dims
    with netcdf.Dataset(str(out_file), "w", format="NETCDF4_CLASSIC") as dst:
        dst.createDimension("time", total_time)
        time_var = dst.createVariable("time", "f8", ("time",))
        time_var.units = time_units_out
        time_var.calendar = calendar_out
        time_var.long_name = "time"

        # create spatial dims per variable
        created_dims = set()
        for v, meta in var_meta.items():
            if v == "Time":
                continue
            if meta["found"]:
                for dim_name, dim_len in zip(meta["dims"], meta["shape"]):
                    if dim_name not in created_dims:
                        dst.createDimension(dim_name, dim_len)
                        created_dims.add(dim_name)

        # create output variables
        out_vars = {}
        for v, meta in var_meta.items():
            if v == "Time":
                continue
            if meta["found"]:
                out_dims = ("time",) + meta["dims"]
                try:
                    out_vars[v] = dst.createVariable(v, meta["dtype"], out_dims,
                                                    zlib=True, complevel=4,
                                                    fill_value=meta["fill"])
                except Exception:
                    out_vars[v] = dst.createVariable(v, "f4", out_dims,
                                                    zlib=True, complevel=4, fill_value=meta["fill"])
                # copy attrs
                for k, val in meta.get("attrs", {}).items():
                    try:
                        out_vars[v].setncattr(k, val)
                    except Exception:
                        pass
            else:
                # create scalar-in-time variable
                out_vars[v] = dst.createVariable(v, "f4", ("time",), fill_value=np.nan)
        # write data per matched file
        for fpath, offset, n, tvals in time_slices:
            with netcdf.Dataset(str(fpath), "r") as src:
                # write time numbers
                tnums = date2num(tvals.to_pydatetime(), units=time_units_out, calendar=calendar_out)
                time_var[offset:offset+n] = tnums

                for v, outv in out_vars.items():
                    if v not in src.variables:
                        # fill with fill or nan
                        fillv = getattr(outv, "_FillValue", np.nan)
                        outv[offset:offset+n, ...] = np.full((n,) + outv.shape[1:], fillv)
                        continue

                    src_var = src.variables[v]
                    data = np.asarray(src_var[:])
                    # if source var has time dim as first dim
                    if src_var.dimensions and src_var.dimensions[0].lower() in ("time","record","t"):
                        try:
                            outv[offset:offset+n, ...] = data
                        except ValueError:
                            # shape mismatch: attempt per-time copy, fallback fill
                            for ti in range(min(n, data.shape[0])):
                                try:
                                    outv[offset+ti, ...] = data[ti]
                                except Exception:
                                    fillv = getattr(outv, "_FillValue", np.nan)
                                    outv[offset+ti, ...] = np.full(outv.shape[1:], fillv)
                    else:
                        # broadcast static spatial var across time frames
                        for ti in range(n):
                            try:
                                outv[offset+ti, ...] = data
                            except Exception:
                                fillv = getattr(outv, "_FillValue", np.nan)
                                outv[offset+ti, ...] = np.full(outv.shape[1:], fillv)

        # global attrs
        dst.setncattr("source_files", ", ".join([p.name for p,_,_,_ in time_slices]))
        dst.setncattr("created_by", "seaice_calculation.py")
    print(f"Wrote {out_file.name}")