# Saved sensor observations: teacher behaviour audit

Read-only analysis of 552 saved input snapshots across 12 V5 confirmation and four V7 development cases. These cohorts remain separate; no model inference, new rollout, altered source record or rescoring was performed. The [numeric audit](observation_audit.json) records per-input hashes and the [helper](audit_observations.py) reproduces the extraction. The helper refuses an existing output directory; choose a fresh `OUT` for a re-extraction and retain its new helper hash.

Physical pad contacts are sampled causally at the saved tactile timestamp from a15Hz trace; quantization is at most one trace interval. Positive body contact is not equivalent to active-gel contact, secure retention, or a completed pick. Added gel normal force is the snapshot wrench minus the explicit fixed baseline, using the saved proxy sign convention. The0.1N threshold here is a diagnostic threshold, not a new success rule.

| Cohort/seed | Lifted | Snapshots with bilateral packet contact | Of those: one or more gel pads below0.1N | Of those: neither field mask above0.025 | Median hold forecast after lift, before release |
|---|---:|---:|---:|---:|---:|
|V5 K4 / 904501|True|8|0|0|0.998|
|V5 K4 / 904502|True|4|0|1|0.974|
|V5 K4 / 904503|False|2|2|2|—|
|V5 K4 / 904504|False|5|4|4|—|
|V5 K4 / 904505|True|8|1|1|0.996|
|V5 K4 / 904506|True|8|1|3|0.998|
|V5 K4 / 904507|True|8|5|7|0.028|
|V5 K4 / 904508|False|4|3|4|—|
|V5 K4 / 904509|False|4|3|3|—|
|V5 K4 / 904510|True|20|2|2|0.967|
|V5 K4 / 904511|True|10|2|1|0.999|
|V5 K4 / 904512|False|5|1|3|—|
|V7 K1 / 904301|False|1|1|1|—|
|V7 K1 / 904302|False|46|46|21|—|
|V7 K4 / 904301|True|20|1|1|0.969|
|V7 K4 / 904302|True|8|1|2|0.999|

The five V5 cases without sustained lift frequently have incomplete gel coverage while both physical pad bodies touch the packet. In V7 K1 seed904302, all46 such snapshots have at least one weak/absent active-gel load;21 have neither mask above the training contact threshold. This is consistent with a poor or peripheral grasp. It does not show whether geometry, contact projection, model pose choice or all three caused it; these inputs need comparison with measured pad contact and calibration.

Several failed carries instead have strong hold forecasts and both active-gel inputs. For example V5 seeds904501/505/506/511 have median post-lift hold forecasts0.998/0.996/0.998/0.999. Missing contact observations therefore do not explain all reach-limit failures. Seed904507 is an exception: physical lift/carry accompanies weak field coverage and a low hold forecast, warranting a distinct sensor/contact review.

`p_evt` is the model’s anticipatory event prediction, not a measured state or calibrated confidence score. Its hold probability is expected to fall approaching predicted release. The training contact rule can use contact on either pad; the “one weak pad” statistic alone does not imply no model contact evidence.

The proxy is stateless apart from caller-supplied contact geometry: fixed measured background plus analytic force-dependent deformation. `contact_state` is passed to the model without a learned normalization transform; fields and wrist streams use checkpoint normalization. Physical scalar area units, footprint, shear/slip and pose-dependent wrist bias require measured comparison. The complete real demonstrations also include zero SDK area alongside a substantial field contact mask, so disagreement between those two signals alone is not a simulator bug; see the [measured-data audit](../data/README.md). Do not fabricate missing gel contact from whole-pad/backing loads or change thresholds to turn transient touch into a successful pick.

Suggested discriminating checks: (1) compare measured pad gap/contact-face geometry and real vs simulated area/mask/shear at matched loads, (2) replay identical saved model observations with explicitly labelled, physically plausible one-factor sensor variations and identical noise/previous-contact state, and (3) log the model’s anticipatory and reactive gate components. These are proposals; this audit establishes no causal sensor ablation result.
