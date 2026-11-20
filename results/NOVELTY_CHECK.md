# Paper 8 — careful novelty check of Behaviour-Matched Evaluation (BME) — 20 Nov 2026
Searched: web (extended), OpenAlex, IOPscience/JNE, NeuroImage, GigaScience, arXiv. Queries on performance-matched /
behaviour-matched / confound-controlled evaluation of EEG and fNIRS workload decoders.

## Verdict (honest)
- **No paper found that evaluates workload decoders on behaviour-matched trial pairs, or that shows "accuracy says the
  brain adds nothing beyond performance, but the brain still ranks load at equal performance" across datasets.**
  → The *application + finding* looks new.
- **But the general statistical idea is NOT new.** Controlling a confound by matching / counterbalancing at test time
  is known in neuroimaging ML. So we must present BME as a *workload-specific evaluation protocol and an empirical
  finding*, not as a new statistical method. Claiming more would be rejected.

## Closest prior work (must cite and position against)
| Paper | What it does | How we differ |
|---|---|---|
| Snoek, Miletić, Scholte 2019, NeuroImage 184:741 | Confounds in decoding; post-hoc counterbalancing (subsampling) vs cross-validated confound regression (CVCR). Finds counterbalancing can be **positively biased**; recommends CVCR. | General method, MRI. We apply matching to behaviour in workload BCI. **Their bias warning applies to us → we must run a null simulation and compare with CVCR.** |
| Chyzhyk, Varoquaux, Milham, Thirion 2022, GigaScience | "Confound-isolating cross-validation": test sets where target and confound are independent → confound-free accuracy. | Same spirit as BME. Ours is within-person pair matching with behaviour as the confound; must cite as the general framework. |
| Dinga et al. 2020 (bioRxiv) / Spisak 2023 (partial confound test) | Confound control / testing for ML predictions. | General tools; we can use the partial-confound test as an extra check. |
| Boring et al. 2020, J. Neural Eng. 17:056016 | Continuous EEG cognitive-load decoding; notes load levels also differ in RT; cross-task models to avoid task-specific confounds. | Notes the confound but does not evaluate at matched performance. Good JNE precedent. |
| Kakkos et al. 2021, IEEE JBHI | Cross-task workload discrimination (0.94). | Cross-task, not performance-controlled. |
| Kilgallen et al. 2025 arXiv 2508.00531 | Repeated-stimulus confound inflates EEG decoding 4–7 %. | Same "evaluation is inflated" message type, different confound — supports that JNE-type venues care. |

## What reviewers will ask → extra work needed before writing
1. **Null simulation** (Snoek's bias point): synthetic brain features that encode ONLY behaviour → BME must give ≈ 0.5.
   And features encoding load beyond behaviour → BME > 0.5. Small, laptop OK.
2. **Compare with CVCR** (regress behaviour out of brain features, cross-validated) and confound-isolating CV on the
   same 4 settings. Show BME agrees / where it differs and why matched pairs are easier to read.
3. **Caliper sensitivity** (0.05 / 0.10 / 0.15 / 0.20) and a permutation null.
4. Report the failed discovery rule honestly (already planned).

## JNE chance
About 3 in 10 now; maybe 4 in 10 with items 1–3 done. Fallback: J. Neural Eng. → Journal of Neuroscience Methods /
IEEE TNSRE / BSPC.

Sources: Snoek 2019 https://dare.uva.nl/id/68a1eac7-6da0-44ff-8de4-3d2dd99669fc ; Chyzhyk 2022
https://pmc.ncbi.nlm.nih.gov/articles/PMC8917515 ; Boring 2020 https://iopscience.iop.org/article/10.1088/1741-2552/abb9bc ;
Kakkos 2021 https://doi.org/10.1109/jbhi.2021.3085131 ; Kilgallen 2025 https://arxiv.org/pdf/2508.00531

## Update 20 Nov afternoon — novelty of RB-ME (Reliability-Bounded Matched Evaluation)
- **Key prior (must cite, frame around it):** Westfall & Yarkoni 2016, PLOS ONE 11(3):e0152719, "Statistically controlling
  for confounding constructs is harder than you think": controlling for a NOISY measure of a confound leaves residual
  confounding; ability to control is capped by the measure's reliability; remedy = latent-variable (SEM) models with
  multiple indicators. → Exactly the failure we hit with BME (P8-BMECHK). The problem is KNOWN in social science.
- Snoek 2019 / Chyzhyk 2022: confound control in decoding, but they assume the confound is measured without error.
- Kilgallen 2025, Li 2024 (arXiv 2405.17024, temporal autocorrelation inflates EEG decoding): "evaluation is inflated" type.
- Split-half reliability modelling exists for FC (NeuroImage 2021, 10.1016/j.neuroimage.2021.117842) but not for
  confound-controlled decoder evaluation.
- **No paper found** that evaluates BCI/workload decoders against a reliability-corrected bound of what PERFECT
  performance could explain, using split-half behaviour on matched pairs.
**Positioning:** "We bring Westfall–Yarkoni's residual-confounding problem to neural decoder evaluation and give a
simple, model-free test (split-half matching + attenuation-corrected AUC bound), validated on fake brains, on 3 open
datasets." Novel in neural engineering; built from known statistics → honest framing as a validated evaluation protocol.
