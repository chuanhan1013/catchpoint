"""Shared rounding policy for every PUBLISHED I2 hardware-feasibility number.

POLICY (reviewer rounds 4-5): a published bound must never be better than the source value in the JSON.
  req(v, d)  rounds UP   -> things the hardware must withstand or the walker must supply: loads, tensions,
                            torques, required ratings, resultants, maxima, upper CI bounds, required mass,
                            failure fractions, durations.
  cap(v, d)  rounds DOWN -> things the design can deliver: capacities, safety factors, friction capacity,
                            slide limits, minima, lower CI bounds, envelopes, coverage (success) fractions.
  pct_ci(lo, hi)         -> a confidence interval in percent: lo rounded DOWN, hi rounded UP, 1 decimal.
  desc(v, d)             -> nearest (half up). ONLY for descriptive statistics of the simulated sample
                            (medians, 95th percentiles) and nominal geometry; never for a bound.
Each function returns a float; the *_s variants return a string with exactly d decimals.
Rounding is done in decimal arithmetic after snapping float noise (1e-9), so req(0.30000000000000004, 1) == 0.3.
"""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP

EN_DASH = "–"


def _dec(v):
    return Decimal(repr(round(float(v), 9)))


def _q(v, d, mode):
    return _dec(v).quantize(Decimal(1).scaleb(-d), rounding=mode)


def req(v, d=0):
    """Round UP (conservative for loads, tensions, torques, ratings, resultants, maxima, upper CI bounds)."""
    return float(_q(v, d, ROUND_CEILING))


def cap(v, d=0):
    """Round DOWN (conservative for capacities, safety factors, envelopes, minima, lower CI bounds)."""
    return float(_q(v, d, ROUND_FLOOR))


def desc(v, d=0):
    """Nearest, half up. Descriptive statistics only (median, p95) and nominal geometry; labelled as such."""
    return float(_q(v, d, ROUND_HALF_UP))


def req_s(v, d=0):
    return format(_q(v, d, ROUND_CEILING), "f")


def cap_s(v, d=0):
    return format(_q(v, d, ROUND_FLOOR), "f")


def desc_s(v, d=0):
    return format(_q(v, d, ROUND_HALF_UP), "f")


def pct_cap(k, n, d=1):
    """Coverage / success percentage k/n*100, rounded DOWN."""
    return cap(100.0 * k / n, d)


def pct_req(k, n, d=1):
    """Failure percentage k/n*100, rounded UP."""
    return req(100.0 * k / n, d)


def pct_ci(lo, hi, sep=EN_DASH, d=1):
    """Confidence interval in percent as 'lo-hi': lower bound rounded DOWN, upper bound rounded UP."""
    return f"{cap_s(lo, d)}{sep}{req_s(hi, d)}"
