"""Settings for the Caspian / Volga-Caspian channel early-warning hindcast.

Everything a reviewer may want to challenge lives here, with its origin.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "nasa_gwm_lake000270_monthly.txt"
OUT_DIR = ROOT / "outputs"

# --- Datum -----------------------------------------------------------------
# The altimetry series is relative. We tie it to the Baltic height system (m BS)
# with ONE external anchor: Kazakhstan's Ministry of Water Resources stated the
# level was -29.21 m BS on 22 Apr 2025 (bizmedia.kz, nur.kz).
ANCHOR_MONTH = "2025-04"
ANCHOR_LEVEL_BS = -29.21
# Independent check (not used for fitting): the 1995 peak is reported in the
# literature at about -26.6 to -26.7 m BS.
CHECK_YEAR, CHECK_LEVEL_BS = 1995, -26.66

# --- Backtest --------------------------------------------------------------
FIRST_ORIGIN = "1998-01"      # first forecast issue date (>= 5 years of history)
HORIZON = 24                  # months ahead
TREND_WINDOW = 36             # months of deseasonalised history used for the trend
CLIM_MIN_YEARS = 5            # minimum years to estimate the seasonal cycle
PI_LEVELS = (0.10, 0.90)      # 80% prediction interval from past errors only
ERROR_WINDOW = 120            # most recent verified errors used per horizon (10 years)
ACI_GAMMA = 0.05              # step of the adaptive interval scale (per verified forecast)
ALERT_PROB = 0.5              # raise an alert when P(level < threshold) >= this

# --- Exposure thresholds (m BS) ---------------------------------------------
# A threshold is the sea level below which an object loses its operating margin.
# Origin matters: 'external' thresholds are defined independently of the incidents;
# 'calibrated' ones were chosen to match the first reported incident and must be
# replaced by survey data (Rosmorport soundings, vessel drafts) before real use.
THRESHOLDS = [
    {
        "id": "vkmsk_supply",
        "object": "Суда снабжения ЛУКОЙЛ-Нижневолжскнефть, Волго-Каспийский канал (морская часть)",
        "level_bs": -28.00,
        "kind": "calibrated",
        "basis": "Подобран под первый инцидент: в апреле 2015 суда грузились на 50% (ТАСС). "
                 "Заменить на отметку дна канала по промерам Росморпорта.",
        "incident": "2015-04",
    },
    {
        "id": "historic_min",
        "object": "Каламкас-море / Хазар: подходы через Уральскую седловину",
        "level_bs": -29.01,
        "kind": "external",
        "basis": "Исторический минимум 1977 г. (-29,01 м БС, Казгидромет) — проектная граница, "
                 "ниже которой у мелководной инфраструктуры нет опыта эксплуатации.",
        "incident": None,  # no dated incident: lead time is measured to the actual crossing
    },
]
