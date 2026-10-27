"""Load the monthly altimetry series and tie it to the Baltic height system."""
import numpy as np
import pandas as pd

from . import config


def load_monthly() -> pd.DataFrame:
    rows = []
    for line in config.RAW_FILE.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for tok in line.split():
            ym, val, n = tok.split(":")
            rows.append((pd.Period(f"{ym[:4]}-{ym[4:]}", "M"), float(val), int(n)))
    df = pd.DataFrame(rows, columns=["month", "anom_m", "n_passes"]).set_index("month")
    # fill the few missing months by linear interpolation and flag them
    full = pd.period_range(df.index.min(), df.index.max(), freq="M")
    df = df.reindex(full)
    df["filled"] = df["anom_m"].isna()
    df["anom_m"] = df["anom_m"].interpolate()
    df["n_passes"] = df["n_passes"].fillna(0).astype(int)

    offset = config.ANCHOR_LEVEL_BS - df.loc[pd.Period(config.ANCHOR_MONTH, "M"), "anom_m"]
    df["level_bs"] = df["anom_m"] + offset
    df.attrs["offset"] = float(offset)
    yr = df["level_bs"].groupby(df.index.year).mean()
    df.attrs["check"] = {
        "year": config.CHECK_YEAR,
        "computed": round(float(yr.loc[config.CHECK_YEAR]), 3),
        "literature": config.CHECK_LEVEL_BS,
    }
    return df


if __name__ == "__main__":
    d = load_monthly()
    print(d.head(), d.tail(), d.attrs, sep="\n")
    print("filled months:", int(d["filled"].sum()))
