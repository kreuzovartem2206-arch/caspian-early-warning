"""30-year outlook for the Caspian Sea level and depth at LUKOIL objects.

Stitching
---------
Years 1-2 (Nov 2026 - Oct 2028) come from the monthly stochastic ensemble forecast
(forecast_now.csv, 80% intervals from the hindcast errors). Everything after that starts
from the end of that forecast: the level of the last forecast year (Nov 2027 - Oct 2028
mean, assigned to 2028) is the common starting point of all long-term paths.

Long-term paths, on the ANNUAL MEAN level (autumn-winter minimum is ~0.15 m lower):
1. "ours"  AR(1) on observed annual-mean changes 1997-2025, mean drift uncertain;
           20,000 Monte Carlo paths, start level drawn from the 2028 forecast interval.
2. CMIP6   Samant & Prange 2023 (Commun. Earth Environ. 4, 357): total decline 2020-2100
           = climate-driven trend of best-fit models + observed 1996-2021 trend (~4.3 m):
           SSP1-2.6 ~7.9 m (3.65 + 4.3), SSP2-4.5 ~8 m, SSP5-8.5 ~14 m.
3. CESM    Nandini-Weiss et al. 2020 (Int. J. Climatol. 40, 2717), CESM1.2.2 with the
           Caspian water balance: ~9 m (RCP4.5) and ~18 m (RCP8.5) by the end of the century,
           spread here evenly over 2006-2100 (the RCP period).
"Live" lines: each scenario trend gets its own realisation of interannual variability with the
statistics of the observed annual changes (AR(1), same phi and sigma as our model), pinned to
zero at 2028 and at 2056 so the scenario's start and end stay exactly as published. For our
model the live line is the Monte Carlo path closest to the median (one plausible future).
Object thresholds and probabilities are still computed from the trends / all 20,000 paths.
Straight trends ignore the area feedback (a smaller sea evaporates less), so beyond
~2040 the fast scenarios are upper bounds of the decline rather than central estimates.
"""
import numpy as np
import pandas as pd

from . import config

END_YEAR = 2056
FIT_YEARS = (1997, 2025)
N_PATHS = 20000
SEED = 20261029
Z80 = 2.5631  # width of a central 80% normal interval in sigmas

SCENARIOS = [
    {"id": "ssp126", "group": "CMIP6", "label": "CMIP6 SSP1-2.6: −7,9 м за 2020–2100 гг.", "total_m": 7.9, "span": (2020, 2100),
     "source": "Samant & Prange 2023: тренд моделей 3,65 м + наблюдённый тренд ~4,3 м"},
    {"id": "ssp245", "group": "CMIP6", "label": "CMIP6 SSP2-4.5: −8 м за 2020–2100 гг.", "total_m": 8.0, "span": (2020, 2100),
     "source": "Samant & Prange 2023, межмодельный разброс 2–15 м"},
    {"id": "ssp585", "group": "CMIP6", "label": "CMIP6 SSP5-8.5: −14 м за 2020–2100 гг.", "total_m": 14.0, "span": (2020, 2100),
     "source": "Samant & Prange 2023, межмодельный разброс 11–21 м"},
    {"id": "cesm45", "group": "CESM", "label": "CESM RCP4.5: −9 м к 2100 г.", "total_m": 9.0, "span": (2006, 2100),
     "source": "Nandini-Weiss et al. 2020 (Бременский университет)"},
    {"id": "cesm85", "group": "CESM", "label": "CESM RCP8.5: −18 м к 2100 г.", "total_m": 18.0, "span": (2006, 2100),
     "source": "Nandini-Weiss et al. 2020 (Бременский университет)"},
]


def annual_means(y: pd.Series) -> pd.Series:
    ann = y.groupby(y.index.year).agg(["mean", "count"])
    return ann[ann["count"] == 12]["mean"]


def fit_ar1(changes: np.ndarray):
    mu = changes.mean()
    x0, x1 = changes[:-1] - mu, changes[1:] - mu
    phi = float(np.clip((x0 * x1).sum() / (x0 * x0).sum(), -0.9, 0.9))
    resid = x1 - phi * x0
    sigma = float(resid.std(ddof=1))
    se_mu = float(changes.std(ddof=1) / np.sqrt(len(changes)) * np.sqrt((1 + phi) / (1 - phi)))
    return mu, se_mu, phi, sigma


def stochastic(start_level, start_sd, start_change, mu, se_mu, phi, sigma, years):
    rng = np.random.default_rng(SEED)
    mus = rng.normal(mu, se_mu, N_PATHS)
    lev = rng.normal(start_level, start_sd, N_PATHS)
    x = np.full(N_PATHS, start_change)
    out = np.empty((years, N_PATHS))
    for k in range(years):
        x = mus + phi * (x - mus) + rng.normal(0, sigma, N_PATHS)
        lev = lev + x
        out[k] = lev
    return out


def live_noise(phi, sigma, n, seed):
    """Interannual wiggle with observed AR(1) statistics, zero at both ends (bridge)."""
    rng = np.random.default_rng(seed)
    x, dev = 0.0, np.empty(n)
    for k in range(n):
        x = phi * x + rng.normal(0, sigma)
        dev[k] = x
    dev = np.cumsum(dev)
    return dev - dev[-1] * np.arange(1, n + 1) / n


def forecast_years(fc: pd.DataFrame):
    """Two forecast 'years' Nov-Oct from the 24-month monthly forecast."""
    rows = []
    for k in range(2):
        g = fc.iloc[12 * k: 12 * (k + 1)]
        rows.append({"year": int(str(g.target.iloc[-1])[:4]),
                     "fc_p50": float(g.forecast.mean()),
                     "fc_p10": float(g.pi_lo.mean()), "fc_p90": float(g.pi_hi.mean()),
                     "sd": float((g.pi_hi - g.pi_lo).mean() / Z80)})
    return rows


def run(y: pd.Series, exposure_rows: list, fc_now: pd.DataFrame):
    ann = annual_means(y)
    ch = ann.diff().dropna().loc[FIT_YEARS[0]:FIT_YEARS[1]].to_numpy()
    mu, se_mu, phi, sigma = fit_ar1(ch)

    fy = forecast_years(fc_now)
    start_year = fy[-1]["year"]                      # 2028
    start_level = fy[-1]["fc_p50"]
    start_change = fy[-1]["fc_p50"] - fy[0]["fc_p50"]
    n = END_YEAR - start_year
    years = np.arange(start_year + 1, END_YEAR + 1)
    paths = stochastic(start_level, fy[-1]["sd"], start_change, mu, se_mu, phi, sigma, n)
    for s in SCENARIOS:
        s["rate"] = -s["total_m"] / (s["span"][1] - s["span"][0])

    live = {s["id"]: live_noise(phi, sigma, n, SEED + 7 * (k + 1)) for k, s in enumerate(SCENARIOS)}
    med = np.quantile(paths, 0.5, axis=1)
    rep = paths[:, np.argmin(((paths - med[:, None]) ** 2).sum(axis=0))]
    rows = [{"year": f["year"], "fc_p10": f["fc_p10"], "fc_p50": f["fc_p50"], "fc_p90": f["fc_p90"]} for f in fy]
    rows[-1].update({"stoch_p10": fy[-1]["fc_p10"], "stoch_p50": start_level, "stoch_p90": fy[-1]["fc_p90"],
                     **{s["id"]: start_level for s in SCENARIOS}, **{s["id"] + "_live": start_level for s in SCENARIOS},
                     "stoch_live": start_level})
    for i, yr in enumerate(years):
        r = {"year": int(yr),
             "stoch_p10": float(np.quantile(paths[i], 0.10)), "stoch_p50": float(np.quantile(paths[i], 0.50)),
             "stoch_p90": float(np.quantile(paths[i], 0.90))}
        for s in SCENARIOS:
            r[s["id"]] = start_level + s["rate"] * (yr - start_year)
            r[s["id"] + "_live"] = r[s["id"]] + float(live[s["id"]][i])
        r["stoch_live"] = float(rep[i])
        rows.append(r)
    levels = pd.DataFrame(rows)
    hist = pd.DataFrame({"year": ann.index.astype(int), "observed": ann.to_numpy()})

    # objects: year the annual-mean depth falls below the working depth and below zero
    need = config.EXPOSURE["draft_m"] + config.EXPOSURE["ukc_m"]
    datum = config.CHART_DATUM_BS
    obj_rows = []
    for o in [r for r in exposure_rows if r["depth_basis"] == "min"]:
        d = o["depth_ref_m"]
        lvl_need, lvl_dry = datum - (d - need), datum - d
        rec = {"id": o["id"], "object": o["object"], "depth_ref_m": d,
               "level_need_bs": round(lvl_need, 2), "level_dry_bs": round(lvl_dry, 2),
               "depth_2028_m": round(d + start_level - datum, 2)}
        for s in SCENARIOS + [{"id": "stoch"}]:
            for tag, lv in (("need", lvl_need), ("dry", lvl_dry)):
                if s["id"] == "stoch":
                    hit = np.where(levels.stoch_p50.to_numpy()[1:] <= lv)[0]
                    yr = None if not len(hit) else int(levels.year.iloc[1 + hit[0]])
                elif start_level <= lv:
                    yr = start_year
                else:
                    yr = start_year + (lv - start_level) / s["rate"]
                    yr = None if yr > END_YEAR + 0.5 else int(np.ceil(yr))
                rec[f"year_{tag}_{s['id']}"] = yr
        for yr in (2036, 2046, 2056):
            i = int(yr - start_year - 1)
            rec[f"p_below_need_{yr}"] = round(float((paths[: i + 1] < lvl_need).any(axis=0).mean()), 3)
            rec[f"depth_p50_{yr}"] = round(d + float(np.quantile(paths[i], 0.5)) - datum, 2)
            rec[f"depth_p10_{yr}"] = round(d + float(np.quantile(paths[i], 0.1)) - datum, 2)
            for s in SCENARIOS:
                rec[f"depth_{s['id']}_{yr}"] = round(d + start_level + s["rate"] * (yr - start_year) - datum, 2)
        obj_rows.append(rec)
    objects = pd.DataFrame(obj_rows)
    meta = {"mu_cm": mu * 100, "se_mu_cm": se_mu * 100, "phi": phi, "sigma_cm": sigma * 100,
            "start_year": start_year, "start_level": start_level, "start_sd": fy[-1]["sd"],
            "fit_years": FIT_YEARS, "end_year": END_YEAR,
            "scenarios": [{**s, "rate_cm": s["rate"] * 100} for s in SCENARIOS]}
    return levels, hist, objects, meta
