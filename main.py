"""Forecast GTA VI launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Model: an unlock-aligned, scale-aware shape model built from the case's own
comparables (past big launches' hourly Twitch curves).

Why not a flat persistence forecast: hour 1 (00:00-01:00 UTC) only has New
Zealand, Australia, central Europe and the UK unlocked. US East unlocks at
05:00 UTC (hour 6) and US West at 08:00 UTC (hour 9), so viewership ramps
strongly through the day. A flat forecast misses that ramp entirely.

Why not a plain ratio-to-hour-1 curve: across past launches the late-day
ratio to hour 1 ranges from ~1x to ~2000x. Hour 1 is a weak anchor for the
absolute level: launches whose hour 1 was tiny (small early-unlock regions,
sparse sampling) still peaked at hundreds of thousands of viewers, while
launches with a huge hour 1 barely grew. What is stable is the *level*
relationship: for each hour h, log(viewers_h) is roughly linear in
log(viewers_1) with a slope that falls from ~1.0 in the early hours to
~0.45 late in the day (regression to the mean across launches). So for every
hour we fit, over the comparables,

    log v_h = alpha_h + beta_h * log v_1

by (recency-weighted) least squares, and read the forecast median from that
line at the case's own hour-1 print. The spread comes from the fit's
residual spread, widened by the usual prediction-leverage term, so hours
where launches disagree get honest, wide quantiles.

Alignment: launches are aligned by hours since the US-East midnight unlock
(04:00Z in daylight time, 05:00Z in standard time), parsed from each
comparable's unlock note and otherwise from the US daylight-saving calendar.
That lines up the US ramp across launches that unlocked in different
seasons. The target case's own unlock hour is derived the same way from its
notes, falling back to the calendar rule.

Filtering: only comparables with a usable hour-1 print of at least 10,000
viewers are used for the fit. Launches with a tiny hour 1 are dominated by
sampling noise and by unlock patterns nothing like a big console launch;
they blow up any ratio curve and add no signal about the level.

Extrapolation guard: GTA VI may print a bigger hour 1 than any comparable.
When the case's log hour-1 print sits above the largest log hour-1 in the
fit, the median is blended toward the plain median-ratio curve in
proportion to how far outside the training range it lies, and the spread is
widened. Inside the training range this changes nothing.

Fallbacks: if too few comparables survive the filter for some hour, that
hour falls back to the median-ratio curve over all comparables; if even
that is empty, to a flat persistence forecast with a widening spread. The
program never crashes on a weird case: it always writes a valid file.

Only the standard library is used. No network, no randomness, no state:
two runs on the same input produce byte-identical output.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- input / output -------------------------------------------------------

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them, with the
# standard normal's quantile at each level.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# --- model constants ------------------------------------------------------

# A comparable only joins the fit if its hour-1 print is at least this big.
MIN_HOUR1 = 10_000.0
# How fast older launches lose weight: 0.85 per year of age.
RECENCY = 0.85
# Smallest spread (log units) any hour is allowed.
SIGMA_FLOOR = 0.15
# Minimum comparables needed for a per-hour fit; fewer falls back to ratios.
MIN_FIT = 6
# Blend toward the ratio curve once log hour-1 exceeds the fit's max by
# this much (log units); full blend one log unit beyond the max.
EXTRAP_RANGE = 1.0


# --- time helpers ---------------------------------------------------------

def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _first_sunday(year, month, day):
    d = datetime(year, month, day, tzinfo=timezone.utc)
    while d.weekday() != 6:  # Sunday
        d += timedelta(days=1)
    return d.date()


def us_east_unlock_hour(day_start, unlock_text, notes=""):
    """UTC hour (1-24) at which the US East coast unlocks, i.e. the hour
    whose [h-1, h) UTC window holds midnight Eastern.

    Prefer an explicit time in the unlock note ("Eastern ... 05:00Z");
    otherwise use the US daylight-saving calendar: daylight time
    (UTC-4) puts the unlock at 04:00Z = hour 5, standard time (UTC-5)
    at 05:00Z = hour 6.
    """
    text = (unlock_text or "") + " " + (notes or "")
    m = re.search(r"[Ee]astern[^0-9]{0,40}?(\d{2}):00Z", text)
    if m:
        return int(m.group(1)) + 1
    for m in re.finditer(r"0([45]):00Z", text):
        return int(m.group(1)) + 1
    day = parse_time(day_start).date()
    start = _first_sunday(day.year, 3, 8)
    end = _first_sunday(day.year, 11, 1)
    return 5 if start <= day < end else 6


# --- comparables ----------------------------------------------------------

def release_day_hours(comp):
    """Map hour 1..24 of the comparable's release day to its viewers."""
    start = parse_time(comp["day_start"])
    hours = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        slot = int((when - start).total_seconds() // 3600) + 1
        if 1 <= slot <= 24:
            hours[slot] = float(viewers)
    return hours


def collect_comparables(case):
    """One record per comparable with a usable release day."""
    rows = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if 1 not in hours:
            continue
        rows.append({
            "hours": hours,
            "v1": hours[1],
            "east": us_east_unlock_hour(comp["day_start"], comp.get("unlock", "")),
            "year": parse_time(comp["day_start"]).year,
        })
    return rows


# --- fitting --------------------------------------------------------------

def _weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sigma)."""
    total = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    beta = sxy / sxx if sxx > 0 else 0.0
    alpha = my - beta * mx
    resid = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    sigma = statistics.pstdev(resid) if len(resid) > 1 else 0.5
    return alpha, beta, sigma, mx, sxx, len(xs)


def _median(values):
    return statistics.median(values)


def fit_hour(rows, aligned_slot, log_v1, target_year):
    """Fit one aligned hour. Returns (mu, sigma, ratio_mu) where mu is the
    log-median forecast from the level regression, sigma its spread, and
    ratio_mu the log-median from the plain ratio curve (for blending)."""
    xs, ys, ws = [], [], []
    ratios = []
    for row in rows:
        hour = aligned_slot + row["east"]
        value = row["hours"].get(hour)
        if value is None:
            continue
        xs.append(math.log(row["v1"]))
        ys.append(math.log(value))
        ws.append(RECENCY ** max(0, target_year - row["year"]))
        ratios.append(math.log(value / row["v1"]))
    ratio_mu = log_v1 + _median(ratios) if ratios else log_v1
    if len(xs) < MIN_FIT:
        return ratio_mu, 0.5, ratio_mu
    alpha, beta, sigma, mx, sxx, n = _weighted_ols(xs, ys, ws)
    mu = alpha + beta * log_v1
    # Prediction spread: residual spread plus leverage for a print far
    # from the fit's centre.
    if sxx > 0:
        sigma *= math.sqrt(1.0 + 1.0 / n + (log_v1 - mx) ** 2 / sxx)
    return mu, max(SIGMA_FLOOR, sigma), ratio_mu


# --- forecasting ----------------------------------------------------------

def _round2(value):
    """Round half up to 2 decimals; whole numbers come back as ints."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100.0
    return int(out) if out.is_integer() else out


def flat_fallback(p1):
    """Persistence forecast around p1 with a widening spread, used only
    when the model itself fails on a case. Always valid output."""
    return {"hours": [
        {"hour": h, "q": [
            _round2(min(1_000_000_000.0, max(0.01,
                     p1 * math.exp(z * 0.15 * math.sqrt(h - 1))))) for z in Z]}
        for h in range(2, 25)]}


def hour1_print(case):
    """Hour 1's print, or 1 when the case carries none."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
    if first is None:
        return 1.0
    try:
        return max(1.0, float(first.get("viewers") or 1.0))
    except (TypeError, ValueError):
        return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    log_v1 = math.log(p1)

    rows = collect_comparables(case)
    fitted = [r for r in rows if r["v1"] >= MIN_HOUR1]
    use_rows = fitted if len(fitted) >= MIN_FIT else rows
    target_year = parse_time(case["day_start"]).year if case.get("day_start") else 2026
    target_east = us_east_unlock_hour(
        case.get("day_start", "2026-11-19T00:00:00Z"), "", case.get("notes", ""))

    # How far outside the fit's hour-1 range this case sits (0 inside).
    if use_rows:
        max_log_v1 = max(math.log(r["v1"]) for r in use_rows)
    else:
        max_log_v1 = log_v1
    outside = max(0.0, min(1.0, (log_v1 - max_log_v1) / EXTRAP_RANGE))

    wanted = [int(r["hour"]) for r in case.get("forecast") or []]
    if not wanted:
        wanted = list(range(2, 25))
    hours_out = []
    for hour in wanted:
        aligned = hour - target_east
        mu, sigma, ratio_mu = fit_hour(use_rows, aligned, log_v1, target_year)
        if outside > 0.0:
            # A record-sized hour 1: hedge the level fit toward the
            # ratio curve, which grows with the print, and widen up.
            mu = (1.0 - outside) * mu + outside * ratio_mu
            sigma *= 1.0 + 0.5 * outside
        quantiles = []
        prev = 0.0
        for z in Z:
            q = math.exp(mu + z * sigma)
            q = min(1_000_000_000.0, max(0.01, q))
            q = _round2(q)
            if q < prev:  # rounding must never break monotonicity
                q = prev
            quantiles.append(q)
            prev = q
        hours_out.append({"hour": hour, "q": quantiles})
    hours_out.sort(key=lambda r: r["hour"])
    return {"hours": hours_out}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            (OUT_DIR / (path.stem + ".out")).write_text(
                json.dumps(flat_fallback(1.0), separators=(",", ":")) + "\n",
                encoding="utf-8")
            continue
        try:
            result = forecast_case(case)
        except Exception:
            # Never leave a case without a file: fall back to a flat
            # persistence forecast around this case's hour-1 print.
            result = flat_fallback(hour1_print(case))
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
