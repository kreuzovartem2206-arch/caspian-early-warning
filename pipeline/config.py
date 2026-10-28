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

# --- Forward-looking exposure of objects without incidents -------------------
CHART_DATUM_BS = -28.0  # depths on Caspian charts / project docs ("-28 БСВ")
EXPOSURE = {
    "draft_m": 4.5,          # design draft of the Volga-Caspian channel, the supply route (ko.ru, 2026)
    "ukc_m": 0.5,            # under-keel clearance: assumption
    "setdown_typ_m": 0.64,   # mean significant set-down 1978-1995 (Ivkina et al. 2019)
    "setdown_max_m": 1.17,   # largest set-down 1996-2018, M Peshnoy (Ivkina et al. 2019)
    "datum_shift_m": 1.0,    # if a depth was surveyed at ~-27 m BS instead of -28 m
    "scenarios": [
        {"id": "r2005", "rate_cm": -8.9, "label": "темп 2005–2019 гг., −8,9 см/год"},
        {"id": "r2019", "rate_cm": -19.3, "label": "темп 2019–2025 гг., −19,3 см/год"},
    ],
    # Kazakhstan's Ministry of Water Resources, 22 Apr 2025 (nur.kz): -29.5 to -30.2 m BS by 2030
    "level_2030": (-29.5, -30.2),
    "objects": [
        {"id": "grayfer", "object": "Месторождение им. Грайфера (Ракушечное)", "operator": "ЛУКОЙЛ, 100%",
         "status": "добыча с 2023 г.", "depth_min": 5.0, "depth_typ": 5.4,
         "source": "средняя 5,4 м (−28 м БС) — Neftegaz.RU; 5–8 м — NS Energy"},
        {"id": "khazar", "object": "Хазар", "operator": "Kalamkas-Khazar Operating, ЛУКОЙЛ 50%",
         "status": "проект, первая нефть 2028–2029 гг.", "depth_min": 6.0, "depth_typ": 7.0,
         "source": "около 6–8 м — КазМунайГаз"},
        {"id": "kalamkas", "object": "Каламкас-море", "operator": "Kalamkas-Khazar Operating, ЛУКОЙЛ 50%",
         "status": "проект, первая нефть 2028–2029 гг.", "depth_min": 7.0, "depth_typ": 7.5,
         "source": "7–8 м — КазМунайГаз"},
        {"id": "filanovsky", "object": "Месторождение им. Филановского", "operator": "ЛУКОЙЛ, 100%",
         "status": "добыча с 2016 г.", "depth_min": 7.0, "depth_typ": 9.0,
         "source": "7–11 м — PetroDigest"},
        {"id": "korchagin", "object": "Месторождение им. Корчагина", "operator": "ЛУКОЙЛ, 100%",
         "status": "добыча с 2010 г.", "depth_min": 11.0, "depth_typ": 12.0,
         "source": "11–13 м — национальный доклад РФ, Тегеранская конвенция"},
        {"id": "kuvykin", "object": "Месторождение им. Кувыкина (Сарматское)", "operator": "ЛУКОЙЛ, 100%",
         "status": "проект, ввод около 2029 г.", "depth_min": 15.0, "depth_typ": 15.0,
         "source": "около 15 м — uglevodorody.ru"},
        {"id": "khvalynskoye", "object": "Хвалынское", "operator": "ЛУКОЙЛ, совместно с КазМунайГазом",
         "status": "проект", "depth_min": 25.0, "depth_typ": 27.5,
         "source": "25–30 м — национальный доклад РФ, Тегеранская конвенция"},
    ],
}

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
