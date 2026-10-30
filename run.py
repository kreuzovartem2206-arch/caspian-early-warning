"""Run the full hindcast: ingest -> forecasts -> intervals -> alerts -> outputs.

    python run.py

Writes CSV/JSON to ./outputs. Runtime ~5-8 min (SARIMA is refitted every month);
`python run.py --reuse` skips refitting and reuses outputs/forecasts_all_models.csv.
"""
import json
import sys

import numpy as np
import pandas as pd

from pipeline import alerts, backtest, config, exposure, ingest, longterm, models


def main():
    config.OUT_DIR.mkdir(exist_ok=True)
    df = ingest.load_monthly()
    y = df["level_bs"]
    print(f"series {y.index[0]}..{y.index[-1]}, offset {df.attrs['offset']:.3f} m, check {df.attrs['check']}")

    cache = config.OUT_DIR / "forecasts_all_models.csv"
    if "--reuse" in sys.argv and cache.exists():
        fc = pd.read_csv(cache)
        for c in ("origin", "target"):
            fc[c] = fc[c].map(lambda s: pd.Period(s, "M"))
        print("reusing cached forecasts")
    else:
        fc = backtest.run(df)
        fc.to_csv(cache, index=False)
    pool = backtest.ErrorPool(fc)
    ens = backtest.add_intervals(fc, "ensemble", pool)
    ens.to_csv(config.OUT_DIR / "forecasts_ensemble_pi.csv", index=False)

    sk = backtest.skill(fc)
    sk.to_csv(config.OUT_DIR / "skill_by_horizon.csv")
    print(sk.loc[[1, 6, 12, 24]].round(3))

    # coverage of the 80% interval
    v = ens.dropna(subset=["pi_lo", "actual"])
    coverage = float(((v.actual >= v.pi_lo) & (v.actual <= v.pi_hi)).mean())
    print(f"80% PI empirical coverage: {coverage:.2%} over {len(v)} verified forecasts")
    static = backtest.add_intervals(fc, "ensemble", pool, adaptive=False).dropna(subset=["pi_lo", "actual"])
    coverage_static = float(((static.actual >= static.pi_lo) & (static.actual <= static.pi_hi)).mean())
    print(f"   (without adaptive scale: {coverage_static:.2%})")

    base = fc[fc.model == "seasonal_naive"].groupby("origin")["forecast"].min()
    results, probs = [], {}
    for thr in config.THRESHOLDS:
        p = backtest.crossing_probability(fc, thr["level_bs"], pool=pool, scales=ens)
        probs[thr["id"]] = p
        r = alerts.evaluate(y, p, base, thr)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1))

    # sensitivity: lead time across a range of thresholds
    sens = []
    for lv in np.round(np.arange(-27.6, -29.21, -0.1), 2):
        p = backtest.crossing_probability(fc, lv, pool=pool, scales=ens)
        r = alerts.evaluate(y, p, base, {"id": f"t{lv}", "object": "", "level_bs": lv,
                                          "kind": "sweep", "basis": "", "incident": None})
        sens.append({k: r[k] for k in ("level_bs", "first_crossing", "sustained_alert_from",
                                       "lead_months_sustained", "baseline_lead_sustained",
                                       "false_alarm_months_before_crossing")})
    pd.DataFrame(sens).to_csv(config.OUT_DIR / "threshold_sensitivity.csv", index=False)

    # forward-looking risk for objects without incidents
    expo = exposure.assess(y, pool, ens)
    expo.to_csv(config.OUT_DIR / "exposure_objects.csv", index=False)
    print(expo[expo.depth_basis == "min"][["id", "depth_now_m", "depth_fc_min_m", "depth_fc_p10_m",
                                            "p24_calm", "year_calm_r2019"]].to_string(index=False))

    rise = alerts.spring_rise(y)
    rise.to_csv(config.OUT_DIR / "spring_rise.csv")
    silt = alerts.compound_siltation_flag(y, rise)
    silt.to_csv(config.OUT_DIR / "siltation_flag_experimental.csv")
    print("experimental siltation flag years:", list(silt.index[silt.flag]))

    # compact bundle for the dashboard
    def per(o):
        return str(o)

    fans = {}
    for r in results:
        for key in ("sustained_alert_from",):
            o = r[key]
            if o:
                g = ens[ens.origin == pd.Period(o, "M")]
                fans[f"{r['id']}"] = {
                    "origin": o,
                    "target": [per(t) for t in g.target],
                    "forecast": g.forecast.round(3).tolist(),
                    "pi_lo": g.pi_lo.round(3).tolist(),
                    "pi_hi": g.pi_hi.round(3).tolist(),
                }
    # live forecast issued at the last observed month
    t_now = y.index[-1]
    f_now = {n: f(y, config.HORIZON) for n, f in models.MODELS.items()}
    f_ens = models.ensemble(f_now)
    last_scale = ens[ens.origin == ens.origin.max()].set_index("h")["scale"]
    now = {"origin": per(t_now), "target": [], "forecast": [], "pi_lo": [], "pi_hi": []}
    for h, v in enumerate(f_ens, start=1):
        e = pool.get("ensemble", t_now, h)
        s = float(last_scale.get(h, 1.0))
        now["target"].append(per(t_now + h)); now["forecast"].append(round(float(v), 3))
        now["pi_lo"].append(round(float(v + s * np.quantile(e, config.PI_LEVELS[0])), 3))
        now["pi_hi"].append(round(float(v + s * np.quantile(e, config.PI_LEVELS[1])), 3))
    pd.DataFrame(now).to_csv(config.OUT_DIR / "forecast_now.csv", index=False)
    print("now:", now["target"][11], now["forecast"][11], now["pi_lo"][11], now["pi_hi"][11],
          "| min next 24m:", min(now["forecast"]))

    # 30-year outlook (scenarios + stochastic model on annual means)
    lt_levels, lt_hist, lt_objects, lt_meta = longterm.run(y, expo.to_dict("records"), pd.DataFrame(now))
    lt_hist.merge(lt_levels, on="year", how="outer").round(3).to_csv(config.OUT_DIR / "longterm_levels.csv", index=False)
    lt_objects.to_csv(config.OUT_DIR / "longterm_objects.csv", index=False)
    (config.OUT_DIR / "longterm_meta.json").write_text(json.dumps(lt_meta, ensure_ascii=False, indent=1, default=str))
    print("longterm 2056 median", round(float(lt_levels.stoch_p50.iloc[-1]), 2))

    bundle = {
        "now": now,
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "source": "NASA/USDA G-REALM, target 000270 Caspian Sea (TOPEX/Jason/Sentinel-6A)",
        "datum": {"offset_m": round(df.attrs["offset"], 3), **df.attrs["check"],
                  "anchor": f"{config.ANCHOR_LEVEL_BS} m BS in {config.ANCHOR_MONTH}"},
        "series": {"month": [per(m) for m in y.index], "level": y.round(3).tolist()},
        "prob": {k: {"origin": [per(o) for o in v.index], "p": v.p_cross.round(3).tolist()}
                 for k, v in probs.items()},
        "fans": fans,
        "results": results,
        "skill": {int(h): {c: round(float(sk.loc[h, c]), 4) for c in sk.columns} for h in sk.index},
        "coverage80": round(coverage, 3),
        "coverage80_static": round(coverage_static, 3),
        "sensitivity": sens,
        "exposure": {"rows": json.loads(expo.to_json(orient="records", force_ascii=False)),
                     "need_m": expo.attrs["need"], "level_now": round(expo.attrs["level_now"], 3),
                     "cfg": {k: v for k, v in config.EXPOSURE.items() if k != "objects"}},
        "siltation_flag_years": [int(v) for v in silt.index[silt.flag]],
        "spring_rise": {int(k): {"rise_m": round(v.rise_m, 3),
                                 "z": None if pd.isna(v.z_vs_past) else round(v.z_vs_past, 2)}
                        for k, v in rise.iterrows()},
    }
    (config.OUT_DIR / "dashboard_data.json").write_text(json.dumps(bundle, ensure_ascii=False))
    print("done ->", config.OUT_DIR)


if __name__ == "__main__":
    main()
