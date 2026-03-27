# TOKIO-LEVEL Deep Audit: `respredict`

Scope: single crate at `libs/scanner/respredict`  
Focus: HTTP response prediction (status code + content type + approximate size), and whether skipping requests is reliable in practice.

## Executive Verdict

- The model **does learn from observed responses**, but it is a lightweight frequency model (majority vote + average size), not an adaptive/robust online learner.
- **Real-world accuracy is unknown** from this crate alone: there is no dataset-driven evaluation, no benchmark harness, and no reported precision/recall.
- It is **potentially useful as a conservative optimization** for very repetitive traffic patterns, but currently too fragile for aggressive skipping in noisy/hostile environments.
- Edge-case resilience (poisoning, backend drift, load balancer variance) is limited; safe use requires strict policy thresholds and external guardrails.

## 1) Does the model actually learn from observed responses?

Yes, in the strict sense.

What it learns:
- `ResponsePredictor::train()` updates three keyed buckets:
  - exact key: method + host + exact path + query key-shape + selected headers
  - family key: method + host + normalized path family (`{int}`, `{uuid}`, `{hex}`) + query key-shape + selected headers
  - host key: method + host + selected headers
- For each bucket, it tracks:
  - status-code frequency map
  - content-type frequency map
  - running size total + min + max + sample count
- `predict()` returns the dominant status/content-type and average size, with confidence:
  - `confidence = 0.7 * status_majority + 0.3 * content_type_majority`
- `should_skip()` requires:
  - `samples >= min_samples`
  - `confidence >= min_confidence`
  - `size_spread_ratio <= max_size_spread_ratio`

Important caveat:
- Learning is purely cumulative; there is no recency weighting, decay, outlier rejection, or per-backend segmentation.

## 2) What’s the accuracy on real-world HTTP traffic?

From this crate: **not measured**.

Evidence:
- Tests verify correctness of API behavior and edge safety, not prediction quality on live traffic.
- No corpus, replay harness, confusion matrix, or benchmark output exists in the crate.
- Examples are synthetic and deterministic.

Implication:
- Any claimed production accuracy would currently be speculative. You need a traffic replay evaluation before trusting skip decisions at scale.

Recommended minimum evaluation protocol:
- Replay representative request traces (per target/service).
- Warm-up phase for training, then holdout phase for prediction.
- Report:
  - status exact-match rate
  - content-type exact-match rate
  - size-within-tolerance rate
  - skip precision (fraction of skipped requests that would truly match)
  - false-skip rate (most important safety metric)
- Slice metrics by endpoint family, host, time window, and backend environment.

## 3) Is this useful in practice or just a demo?

Current state: **useful in narrow scenarios; close to demo for broad internet-scale reliability**.

Where it can help:
- Repeated scanner workflows against stable targets where many URLs follow repetitive templates.
- Quickly identifying likely-static negative paths (e.g., consistent 404 families) when conservative thresholds are used.
- Reducing redundant requests when operational risk of occasional miss is low.

Why it is not broadly production-grade yet:
- No built-in calibration against real traffic.
- No stale-model controls (TTL/versioning/windowing).
- No robust handling for high-variance endpoints.
- No poisoning defenses for untrusted observations.

Bottom line:
- Practical as an optimization layer when run in “safe mode” (high confidence, higher min samples, periodic revalidation), not as a primary truth source.

## 4) Edge-case analysis

### Cache poisoning / training-data poisoning
- Risk: attacker-controlled or transient responses can pollute frequency maps and push model toward wrong dominant status/content type.
- Why vulnerable:
  - all observations are accepted equally
  - no trust tiers, no outlier clipping, no negative weighting
  - no separation by auth context beyond a small header signature
- Impact: false skips and blind spots.
- Mitigations:
  - train only from trusted response sources
  - require stronger skip gates for sensitive paths
  - add anomaly rejection and per-source trust weighting

### Server behavior changes (deploys, feature flags, A/B tests)
- Risk: cumulative historical data lags behind current behavior (concept drift).
- Why vulnerable:
  - no time decay/windowing
  - no model invalidation on drift signals
- Impact: stale predictions, elevated mismatch/false-skip after deploys.
- Mitigations:
  - rolling windows or exponential decay
  - model reset on deploy boundary
  - canary revalidation requests before large skip waves

### Load balancers / multi-origin heterogeneity
- Risk: same host key can map to heterogeneous upstreams with different status/content behavior.
- Why vulnerable:
  - host-level fallback is coarse
  - keying does not include backend identity or routing shard
- Impact: diluted confidence and wrong dominant vote for mixed pools.
- Mitigations:
  - include route/backend fingerprint where available
  - avoid host-level fallback for skip decisions
  - evaluate per-region/POP/backend partitions

### Query/value and header-shape blind spots
- Query shape uses only **sorted keys**, not values.
- Header signature tracks only a fixed subset (`accept`, `authorization`, `content-type`, `x-requested-with`).
- Risk: materially different requests can collide into same bucket.
- Mitigations:
  - optional value-aware features for selected params
  - configurable header features per integration

## Risk Rating

- Functional correctness: **Moderate-High**
- Predictive reliability in uncontrolled real-world traffic: **Unknown to Moderate**
- Adversarial robustness: **Low**
- Safe default for automatic skipping without external controls: **No**

## Practical Go/No-Go Guidance

Use now if:
- you treat it as a probabilistic hint,
- skip policy is conservative,
- and you continuously sample-check skipped requests.

Do not use as-is if:
- missing a response is high impact,
- traffic is highly dynamic/adversarial,
- or you need auditable, measured accuracy guarantees.

## Recommended Next Steps (to reach production confidence)

1. Add an offline replay evaluator and publish benchmark metrics.
2. Introduce time-aware learning (window/decay/TTL) and drift detection.
3. Add poisoning resistance (trusted-source gating, outlier handling, optional robust estimators).
4. Improve feature keying (configurable headers, optional query-value bucketing, backend partition hints).
5. Add safety mechanisms: forced periodic revalidation, endpoint denylist, and false-skip telemetry.

