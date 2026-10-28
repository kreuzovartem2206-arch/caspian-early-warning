"""Build report/caspian_report.html: a standalone page with the run's data embedded.

    python run.py --reuse && python report/build.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    b = json.loads((OUT / "dashboard_data.json").read_text())
    ens = pd.read_csv(OUT / "forecasts_ensemble_pi.csv")
    y = pd.Series(b["series"]["level"], index=pd.PeriodIndex(b["series"]["month"], freq="M"))
    ann = y.groupby(y.index.year).mean()
    sk = pd.read_csv(OUT / "skill_by_horizon.csv", index_col=0)

    def fan(org):
        g = ens[ens.origin == org]
        return {"origin": org, "rows": [[int(r.h), r.target, round(r.forecast, 3), round(r.pi_lo, 3), round(r.pi_hi, 3),
                                         None if pd.isna(r.actual) else round(r.actual, 3)] for r in g.itertuples()]}

    data = {
        "level": [[m, v] for m, v in zip(b["series"]["month"], b["series"]["level"])],
        "annual": [[int(k), round(float(v), 3)] for k, v in ann.items()],
        "prob": {k: [[o, p] for o, p in zip(v["origin"], v["p"]) if p == p] for k, v in b["prob"].items()},
        "fan2013": fan("2013-10"), "fan2022": fan("2022-01"),
        "now": [list(t) for t in zip(b["now"]["target"], b["now"]["forecast"], b["now"]["pi_lo"], b["now"]["pi_hi"])],
        "results": b["results"], "sweep": b["sensitivity"],
        "skill": [[int(h), *(round(float(sk.loc[h, c]) * 100, 1) for c in ("ensemble", "sarima", "trend_seasonal", "seasonal_naive"))]
                  for h in sk.index],
        "rise": [[int(k), v["rise_m"], v["z"]] for k, v in b["spring_rise"].items()],
        "coverage": b["coverage80"], "coverage_static": b["coverage80_static"], "datum": b["datum"],
    }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tpl = (ROOT / "report" / "template.html").read_text()
    (ROOT / "report" / "caspian_report.html").write_text(tpl.replace("__DATA__", js))
    print("written report/caspian_report.html")


if __name__ == "__main__":
    main()
