"""Evidence table for LUKOIL Caspian objects without incidents that are at risk now.

Everything here is observed or published, not forecast, except the clearly marked
24-month model columns:
- sea level: NASA G-REALM satellite series (monthly), tied to m BS;
- depths: published values, referenced to -28.0 m BS (see config.EXPOSURE);
- local check: Kazhydromet gauge-based forecasts for Kulaly / Tyuleniy, summer 2026.

    python run.py --reuse && python report/build_at_risk.py
Writes outputs/at_risk_objects.csv and report/at_risk.html.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import config, ingest  # noqa: E402

OUT = ROOT / "outputs"

# commissioning / design reference year (sources in the page)
REF_YEAR = {"grayfer": (2023, "добыча с 2023 г."), "khazar": (2023, "контракт 2023 г., FEED"),
            "kalamkas": (2023, "контракт 2023 г., FEED"), "filanovsky": (2016, "добыча с октября 2016 г."),
            "korchagin": (2010, "добыча с 2010 г."), "kuvykin": (2023, "проект, ввод около 2029 г."),
            "khvalynskoye": (2023, "проект")}
# Kazhydromet gauge-based forecast, Northern Caspian (MIKE 21), 16-21 Jul and 20-25 Aug 2026, m BS
KAZHYDROMET = {"Кулалы": {"jul_mean": -29.41, "jul_min": -29.46, "aug_mean": -29.43, "aug_min": -29.46},
               "Тюлений": {"jul_mean": -29.22, "jul_min": -29.27, "aug_mean": -29.22, "aug_min": -29.27},
               "Пешной": {"jul_mean": -28.79, "jul_min": -28.83, "aug_mean": -28.91, "aug_min": -28.94}}
LOCAL_GAUGE = {"khazar": "Кулалы", "kalamkas": "Кулалы"}  # nearest gauges to the Kazakh fields


def main():
    y = ingest.load_monthly()["level_bs"]
    ann = y.groupby(y.index.year).mean()
    expo = pd.read_csv(OUT / "exposure_objects.csv")
    cfg = config.EXPOSURE
    need = cfg["draft_m"] + cfg["ukc_m"]
    datum = config.CHART_DATUM_BS
    since = y[y.index >= pd.Period("2024-01", "M")]
    lvl_now, m_now = float(y.iloc[-1]), str(y.index[-1])
    lvl_min = float(y.min()); m_min = str(y.idxmin())

    rows = []
    for o in cfg["objects"]:
        e = expo[(expo.id == o["id"]) & (expo.depth_basis == "min")].iloc[0]
        d = o["depth_min"]
        yr, yr_label = REF_YEAR[o["id"]]
        lvl_ref = float(ann.loc[yr])
        depth_ref = d + (lvl_ref - datum)
        depth_now = d + (lvl_now - datum)
        depth_minobs = d + (lvl_min - datum)
        depth_series = d + (since - datum)
        months_below = int((depth_series < need).sum())
        months_below_sd = int((depth_series - cfg["setdown_typ_m"] < need).sum())
        g = LOCAL_GAUGE.get(o["id"])
        local = KAZHYDROMET[g]["aug_min"] if g else None
        r = {
            "id": o["id"], "object": o["object"], "operator": o["operator"], "status": o["status"],
            "depth_published_m": f'{o["depth_min"]:g}' + ("" if o["depth_typ"] == o["depth_min"] else f'–{o["depth_typ"]:g}'),
            "depth_source": o["source"],
            "ref_year": yr, "ref_label": yr_label, "level_ref_bs": round(lvl_ref, 2),
            "depth_at_ref_m": round(depth_ref, 2),
            "depth_now_m": round(depth_now, 2), "depth_lost_m": round(depth_ref - depth_now, 2),
            "depth_at_record_low_m": round(depth_minobs, 2),
            "depth_now_strong_setdown_m": round(depth_now - cfg["setdown_max_m"], 2),
            "months_below_need_since_2024": months_below,
            "months_below_need_with_setdown_since_2024": months_below_sd,
            "n_months_since_2024": int(len(since)),
            "local_gauge": g, "depth_at_local_gauge_min_m": None if local is None else round(d + (local - datum), 2),
            "model_depth_oct2028_m": e.depth_fc_min_m, "model_depth_oct2028_p10_m": e.depth_fc_p10_m,
            "model_p24_below_need_setdown_typ": e.p24_setdown_typ,
            "trend_year_below_need_r2019": e.year_calm_r2019, "trend_year_below_need_r2005": e.year_calm_r2005,
        }
        r["at_risk_now"] = bool(depth_now < need or depth_now - cfg["setdown_max_m"] < need)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "at_risk_objects.csv", index=False)

    lv = [[str(m), round(float(v), 3)] for m, v in y.items() if m >= pd.Period("2008-01", "M")]
    data = {"rows": rows, "need": need, "draft": cfg["draft_m"], "ukc": cfg["ukc_m"],
            "setdown_typ": cfg["setdown_typ_m"], "setdown_max": cfg["setdown_max_m"],
            "datum": datum, "level_now": round(lvl_now, 3), "month_now": m_now,
            "level_min": round(lvl_min, 3), "month_min": m_min, "level": lv,
            "kazhydromet": KAZHYDROMET, "n_since": int(len(since))}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = (ROOT / "report" / "at_risk_template.html").read_text()
    (ROOT / "report" / "at_risk.html").write_text(tpl.replace("__DATA__", js))
    print(df[["id", "depth_at_ref_m", "depth_now_m", "depth_lost_m", "depth_at_record_low_m",
              "depth_now_strong_setdown_m", "months_below_need_since_2024",
              "months_below_need_with_setdown_since_2024", "depth_at_local_gauge_min_m", "at_risk_now"]].to_string(index=False))
    print("level now", lvl_now, m_now, "record low", lvl_min, m_min, "months", len(since))


if __name__ == "__main__":
    main()
