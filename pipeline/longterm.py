"""30-year outlook for the Caspian Sea level and depth at LUKOIL objects.

The 24-month statistical forecast does not extend to decades, so the outlook is a set
of transparent scenarios on the ANNUAL MEAN level (the autumn-winter minimum is
about 0.15 m lower than the annual mean):

1. Stochastic model (data-driven): annual change follows an AR(1) process around a
   mean drift. Drift, its uncertainty, the AR coefficient and the noise are fitted to
   the observed annual changes 1997-2025. 20,000 Monte Carlo paths give the median
   and the 10-90% band.
2. Rate scenarios (straight lines from the 2026 level):
   - "slow"      -2 m by 2100, lower bound of IO RAS estimates (Kostianoy et al.);
   - "r2005"     observed rate 2005-2019;
   - "ssp245"    CMIP6 SSP2-4.5, ~8 m total drop 2020-2100 (Samant & Prange 2023);
   - "ssp585"    CMIP6 SSP5-8.5, ~14 m total drop 2020-2100 (Samant & Prange 2023);
   - "r2019"     observed rate 2019-2025.
Lines ignore the area feedback (a smaller sea evaporates less), so beyond 2040
the fast scenarios are upper bounds of the decline rather than central estimates.
"""
import numpy as np
import pandas as pd

from . import config

HORIZON_YEARS = 30
FIT_YEARS = (1997, 2025)
N_PATHS = 20000
SEED = 20261029


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


def stochastic(start_level: float, start_change: float, mu, se_mu, phi, sigma, years: int):
    rng = np.random.default_rng(SEED)
    mus = rng.normal(mu, se_mu, N_PATHS)
    lev = np.full(N_PATHS, start_level)
    x = np.full(N_PATHS, start_change)
    out = np.empty((years, N_PATHS))
    for k in range(years):
        x = mus + phi * (x - mus) + rng.normal(0, sigma, N_PATHS)
        lev = lev + x
        out[k] = lev
    return out


def run(y: pd.Series, exposure_rows: list):
    ann = annual_means(y)
    ch = ann.diff().dropna().loc[FIT_YEARS[0]:FIT_YEARS[1]].to_numpy()
    mu, se_mu, phi, sigma = fit_ar1(ch)
    start_year = y.index[-1].year                     # 2026
    start_level = float(y.iloc[-12:].mean())          # last 12 months as the 2026 annual mean
    start_change = float(ann.iloc[-1] - ann.iloc[-2]) # last full-year change
    years = np.arange(start_year + 1, start_year + 1 + HORIZON_YEARS)
    paths = stochastic(start_level, start_change, mu, se_mu, phi, sigma, HORIZON_YEARS)

    def rate(a, b):
        s = ann.loc[a:b]
        return float(np.polyfit(s.index.to_numpy(), s.to_numpy(), 1)[0])

    scen = [
        {"id": "slow", "label": "Замедление: −2 м к 2100 г. (нижняя оценка ИО РАН)", "rate": -2.0 / (2100 - start_year)},
        {"id": "r2005", "label": "Темп 2005–2019 гг.", "rate": rate(2005, 2019)},
        {"id": "ssp245", "label": "CMIP6 SSP2-4.5: −8 м за 2020–2100 гг.", "rate": -8.0 / 80},
        {"id": "ssp585", "label": "CMIP6 SSP5-8.5: −14 м за 2020–2100 гг.", "rate": -14.0 / 80},
        {"id": "r2019", "label": "Темп 2019–2025 гг.", "rate": rate(2019, 2025)},
    ]
    rows = []
    for i, yr in enumerate(years):
        r = {"year": int(yr),
             "stoch_p10": float(np.quantile(paths[i], 0.10)), "stoch_p50": float(np.quantile(paths[i], 0.50)),
             "stoch_p90": float(np.quantile(paths[i], 0.90))}
        for s in scen:
            r[s["id"]] = start_level + s["rate"] * (yr - start_year)
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
               "depth_now_m": round(d + start_level - datum, 2)}
        for s in scen:
            for tag, lv in (("need", lvl_need), ("dry", lvl_dry)):
                yr = start_year if start_level <= lv else (start_year + (lv - start_level) / s["rate"] if s["rate"] < 0 else None)
                rec[f"year_{tag}_{s['id']}"] = None if yr is None else round(yr, 1)
        for yr in (2036, 2046, 2056):
            i = int(yr - start_year - 1)
            rec[f"p_below_need_{yr}"] = round(float((paths[: i + 1] < lvl_need).any(axis=0).mean()), 3)
            rec[f"depth_p50_{yr}"] = round(d + float(np.quantile(paths[i], 0.5)) - datum, 2)
            rec[f"depth_p10_{yr}"] = round(d + float(np.quantile(paths[i], 0.1)) - datum, 2)
        obj_rows.append(rec)
    objects = pd.DataFrame(obj_rows)
    meta = {"mu_cm": mu * 100, "se_mu_cm": se_mu * 100, "phi": phi, "sigma_cm": sigma * 100,
            "start_year": start_year, "start_level": start_level, "fit_years": FIT_YEARS,
            "scenarios": [{**s, "rate_cm": s["rate"] * 100} for s in scen]}
    return levels, hist, objects, meta
