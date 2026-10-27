"""Turn crossing probabilities into alerts and score them against what happened."""
import numpy as np
import pandas as pd

from . import config


def first_crossing(y: pd.Series, level: float):
    below = y[y < level]
    return below.index[0] if len(below) else None


def evaluate(y: pd.Series, prob: pd.DataFrame, baseline_min: pd.Series, thr: dict) -> dict:
    level = thr["level_bs"]
    cross = first_crossing(y, level)
    p = prob["p_cross"]
    alert = p >= config.ALERT_PROB

    def lead(a_series):
        if cross is None:
            return None, None
        pre = a_series[a_series.index < cross]
        on = pre[pre]
        if on.empty:
            return None, None
        first = on.index[0]
        # sustained: the alert stayed on for every month from `first` to the crossing
        sustained_from = None
        for t in reversed(pre.index):
            if not pre.loc[t]:
                break
            sustained_from = t
        return first, sustained_from

    first_alert, sustained = lead(alert)
    base_alert = baseline_min < level
    base_first, base_sustained = lead(base_alert)

    # false alarms: alert on, but the level did not go below within the horizon
    fa = 0
    for t in alert[alert].index:
        if cross is not None and t >= cross:
            break
        window = y.loc[t + 1: t + config.HORIZON]
        if not (window < level).any():
            fa += 1

    def months(a, b):
        return None if a is None or b is None else int((b - a).n)

    incident = pd.Period(thr["incident"], "M") if thr.get("incident") else None
    return {
        **{k: thr[k] for k in ("id", "object", "level_bs", "kind", "basis")},
        "first_crossing": str(cross) if cross is not None else None,
        "incident": str(incident) if incident is not None else None,
        "first_alert": str(first_alert) if first_alert is not None else None,
        "sustained_alert_from": str(sustained) if sustained is not None else None,
        "lead_months_first": months(first_alert, cross),
        "lead_months_sustained": months(sustained, cross),
        "lead_to_incident_sustained": months(sustained, incident),
        "baseline_first_alert": str(base_first) if base_first is not None else None,
        "baseline_sustained_from": str(base_sustained) if base_sustained is not None else None,
        "baseline_lead_sustained": months(base_sustained, cross),
        "false_alarm_months_before_crossing": fa,
    }


def spring_rise(y: pd.Series) -> pd.DataFrame:
    """Volga flood proxy: rise of the sea level from March to the June-July peak.
    A large rise means a big flood, which also brings sediment into the delta channel."""
    rows = []
    for yr in sorted(set(y.index.year)):
        try:
            mar = y.loc[pd.Period(f"{yr}-03", "M")]
            peak = max(y.loc[pd.Period(f"{yr}-06", "M")], y.loc[pd.Period(f"{yr}-07", "M")])
        except KeyError:
            continue
        rows.append((yr, float(peak - mar)))
    d = pd.DataFrame(rows, columns=["year", "rise_m"]).set_index("year")
    # z-score against the previous years only (causal)
    z = []
    for yr in d.index:
        past = d.loc[: yr - 1, "rise_m"]
        z.append(np.nan if len(past) < 8 else float((d.loc[yr, "rise_m"] - past.mean()) / past.std()))
    d["z_vs_past"] = z
    return d


def compound_siltation_flag(y: pd.Series, rise: pd.DataFrame, level: float = -29.01,
                            z_min: float = 0.5) -> pd.DataFrame:
    """EXPERIMENTAL, not validated (one event): flag a year when the sea is already below
    `level` before the flood (March) AND the spring rise is above normal. Idea: a low base
    level gives the channel no depth reserve, and a big flood brings sediment to the bar.
    Defined after seeing the 2026 jam, so it is a hypothesis to test, not evidence."""
    rows = []
    for yr, r in rise.iterrows():
        mar = y.get(pd.Period(f"{yr}-03", "M"), np.nan)
        flag = bool(mar < level and (r.z_vs_past or -9) >= z_min)
        rows.append((yr, float(mar), r.z_vs_past, flag))
    return pd.DataFrame(rows, columns=["year", "march_level_bs", "rise_z", "flag"]).set_index("year")
