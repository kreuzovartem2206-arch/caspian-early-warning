"""Build the standalone MWS-style page report/caspian_shallowing_mws.html.

    python run.py --reuse && python report/build_mws.py

Base data (monthly level, 24-month forecast, objects, Kazhydromet) is in
report/mws_base_data.json; the 30-year outlook is read from outputs/longterm_*.
d3 v7.8.5 is inlined so the page works offline.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    data = json.loads((ROOT / "report" / "mws_base_data.json").read_text())
    lv = pd.read_csv(OUT / "longterm_levels.csv")
    lv = lv[lv.year >= 2027].drop(columns=["observed"])
    meta = json.loads((OUT / "longterm_meta.json").read_text())
    obj = pd.read_csv(OUT / "longterm_objects.csv")
    data["lt"] = {
        "levels": json.loads(lv.round(3).to_json(orient="records")),
        "scen": [{k: s[k] for k in ("id", "group", "label", "source", "rate_cm")} for s in meta["scenarios"]],
        "meta": {k: meta[k] for k in ("mu_cm", "se_mu_cm", "phi", "sigma_cm", "start_year", "start_level", "end_year")},
        "objects": json.loads(obj.to_json(orient="records", force_ascii=False)),
    }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    d3 = (ROOT / "report" / "vendor" / "d3.v7.8.5.min.js").read_text().replace("</script", "<\\/script")
    tpl = (ROOT / "report" / "mws_template.html").read_text()
    (ROOT / "report" / "caspian_shallowing_mws.html").write_text(tpl.replace("__DATA__", js).replace("__D3__", d3))
    print("ok", len(js))


if __name__ == "__main__":
    main()
