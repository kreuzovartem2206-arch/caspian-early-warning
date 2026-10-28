"""Forward-looking risk for LUKOIL Caspian objects that have had no incident yet.

The level model gives the sea level; each object turns it into a water depth at the
site. Risk = the chance that the depth falls below what a supply vessel needs.

Assumptions (all in config.EXPOSURE):
- Published depths are referenced to the Caspian chart datum -28.0 m BS. This is
  stated explicitly only for Grayfer (neftegaz.ru: "5,4 м (-28 БСВ)"); for the rest
  it is an assumption. If a depth was surveyed at a higher level (around -27 m in
  2000-2005), the real depth today is up to 1 m smaller: see `datum_shift_m`.
- Required depth = vessel draft + under-keel clearance.
- Wind set-down (сгон) lowers the level locally for hours to days; values from
  Ivkina et al. 2019 (M Peshnoy station, NE Caspian), applied to all sites.
"""
import numpy as np
import pandas as pd

from . import config, models


def depth_at(level_bs: float, depth_ref: float, ref_level: float = config.CHART_DATUM_BS) -> float:
    return depth_ref + (level_bs - ref_level)


def critical_level(depth_ref: float, need: float, ref_level: float = config.CHART_DATUM_BS) -> float:
    """Sea level at which the depth equals `need`."""
    return ref_level - (depth_ref - need)


def forecast_now(y: pd.Series, pool, scales: pd.DataFrame):
    """Ensemble forecast from the last observed month with error samples per horizon."""
    t_now = y.index[-1]
    fc = models.ensemble({n: f(y, config.HORIZON) for n, f in models.MODELS.items()})
    last = scales[scales.origin == scales.origin.max()].set_index("h")["scale"]
    out = []
    for h, v in enumerate(fc, start=1):
        e = pool.get("ensemble", t_now, h) * float(last.get(h, 1.0))
        out.append((t_now + h, float(v), e))
    return out


def p_below(fc_now, level: float) -> float:
    """P(level < `level` in at least one month of the horizon), as max over horizons."""
    return max(float(np.mean(v + e < level)) for _, v, e in fc_now)


def first_year_below(start_level: float, start_year: float, rate_cm: float, level: float):
    if start_level <= level:
        return start_year
    if rate_cm >= 0:
        return None
    return start_year + (start_level - level) / (-rate_cm / 100)


def assess(y: pd.Series, pool, scales: pd.DataFrame) -> pd.DataFrame:
    cfg = config.EXPOSURE
    fc_now = forecast_now(y, pool, scales)
    lvl_now = float(y.iloc[-1])
    ann_now = float(y.iloc[-12:].mean())
    yr_now = y.index[-1].year + (y.index[-1].month - 0.5) / 12
    fc_min = min(fc_now, key=lambda r: r[1])
    lo_min = min(v + np.quantile(e, 0.10) for _, v, e in fc_now)
    need = cfg["draft_m"] + cfg["ukc_m"]
    rows = []
    for o in cfg["objects"]:
        for which, d in (("min", o["depth_min"]), ("typical", o["depth_typ"])):
            r = {"id": o["id"], "object": o["object"], "operator": o["operator"], "status": o["status"],
                 "depth_basis": which, "depth_ref_m": d, "depth_source": o["source"],
                 "depth_now_m": round(depth_at(lvl_now, d), 2),
                 "depth_fc_min_m": round(depth_at(fc_min[1], d), 2),
                 "depth_fc_min_month": str(fc_min[0]),
                 "depth_fc_p10_m": round(depth_at(lo_min, d), 2)}
            for tag, sd in (("calm", 0.0), ("setdown_typ", cfg["setdown_typ_m"]), ("setdown_max", cfg["setdown_max_m"])):
                lc = critical_level(d, need + sd)
                r[f"crit_level_{tag}"] = round(lc, 2)
                r[f"p24_{tag}"] = round(p_below(fc_now, lc), 3)
                for sc in cfg["scenarios"]:
                    yr = first_year_below(ann_now, yr_now, sc["rate_cm"], lc)
                    r[f"year_{tag}_{sc['id']}"] = None if yr is None else round(yr, 1)
            r["depth_2030_hi_m"] = round(depth_at(cfg["level_2030"][0], d), 2)
            r["depth_2030_lo_m"] = round(depth_at(cfg["level_2030"][1], d), 2)
            # depth with the datum shifted (surveyed at a higher level)
            r["depth_now_if_shift_m"] = round(depth_at(lvl_now, d) - cfg["datum_shift_m"], 2)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs.update(level_now=lvl_now, annual_now=ann_now, need=need,
                     fc_min=(str(fc_min[0]), fc_min[1]), fc_p10_min=lo_min)
    return out
