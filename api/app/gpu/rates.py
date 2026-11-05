"""Realistic CU estimates: per-accelerator hourly rate x expected minutes.

Rates are measured Colab prices (CU per hour). Expected minutes come from the median of the
last completed runs of the same workflow and mode, each scaled back to the default frames x pixels, else the manifest's typical_minutes, else the cap.
Budget caps still reserve the max-runtime worst case; this number is only what the user sees.
"""

from __future__ import annotations

import statistics

HOURLY = {"L4": 1.64, "T4": 1.15, "A100": 5.37, "NONE": 0.07}
T4_HIGHMEM = 1.2


def minute_rate(spec_key: dict | None) -> float:
    key = spec_key or {}
    accel = str(key.get("accelerator") or "NONE").upper()
    if accel == "T4" and "HIGHMEM" in str(key.get("shape", "")).upper():
        return T4_HIGHMEM / 60
    return HOURLY.get(accel, HOURLY["A100"]) / 60


def mode_param(manifest: dict) -> tuple[str, dict] | None:
    """The graph-variant choice (e.g. fast/high-quality), if the workflow has one."""
    params = manifest.get("parameters") or {}
    return next(((k, v) for k, v in params.items() if isinstance(v, dict) and "variants" in v), None)


def expected_minutes(
    repo, owner: str, workflow_key: str | None, manifest: dict, cap: float, prep: bool = False, mode: str | None = None
) -> float:
    picked = mode_param(manifest)
    key = picked[0] if picked else None
    jobs = {j["id"]: j for j in repo.list("jobs", owner)}
    # Each run is normalised back to the default size (a 10 s sample must not make 5 s look dear).
    runs = [
        float(u["elapsed_minutes"]) / (1.0 if prep else length_factor(manifest, (j.get("snapshot") or {}).get("resolved_params") or {}))
        for u in sorted(repo.list("usage", owner), key=lambda u: u.get("created_epoch") or 0)
        if u.get("elapsed_minutes")
        and not u.get("mock")
        and (j := jobs.get(u.get("job_id")))
        and j.get("state") == "completed"
        and bool(j.get("prep")) == prep
        and j.get("workflow_key") == workflow_key
        and (prep or key is None or ((j.get("snapshot") or {}).get("resolved_params") or {}).get(key) == mode)
    ][-5:]
    if runs:
        return statistics.median(runs)
    typical = manifest.get("prep_minutes") if prep else (picked[1].get("typical_minutes", {}).get(mode) if picked else manifest.get("typical_minutes"))
    return float(typical) if typical else float(cap)


def estimate(
    repo, owner: str, workflow_key: str | None, manifest: dict, spec_key: dict | None, cap: float,
    prep: bool = False, mode: str | None = None,
) -> tuple[float, float]:
    """Returns (estimated CU, CU per minute)."""
    rate = minute_rate(spec_key)
    return round(expected_minutes(repo, owner, workflow_key, manifest, cap, prep, mode) * rate, 3), rate


SAMPLING_SHARE = 0.35  # sampling is ~1/3 of a 5 s H3 run; restore/start dominate the rest
LENGTH_EXPONENT = 1.7  # attention cost grows faster than linear with frames


def length_factor(manifest: dict, params: dict) -> float:
    """Work relative to the manifest defaults (frames x pixels); scales the sampling share of a run."""
    spec = manifest.get("parameters") or {}
    ratio = 1.0
    for keys in (("length",), ("width", "height")):
        defaults = [(spec.get(k) or {}).get("default") for k in keys]
        values = [params.get(k) for k in keys]
        if all(isinstance(v, int | float) and v > 0 for v in defaults + values):
            for d, v in zip(defaults, values):
                ratio *= v / d
    return (1 - SAMPLING_SHARE) + SAMPLING_SHARE * ratio ** LENGTH_EXPONENT
