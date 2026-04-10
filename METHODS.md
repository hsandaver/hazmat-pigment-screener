# Methods and Boundaries

This app implements a heuristic screening workflow for hazardous pigments in 19th-century cloth-case bindings. It is designed for triage and safer-handling decisions, not definitive pigment identification.

## What Is Evidence-Based Here

- The app prioritizes green cloth because the current literature and Poison Book Project work show that hazardous green systems are common and operationally important in 19th-century cloth bindings.
- XRF is treated as elemental screening. `As + Cu`, `Pb + Cr`, and `Hg` are meaningful patterns, but they do not establish the exact compound on their own.
- The chrome-yellow-like built-in fallback anchors now use published colorimetry from Otero et al. 2012, who reported reconstructed 19th-century chrome yellow `Lab*` values spanning `76,23,71` to `85,30,89`.
- The app now surfaces class-level evidence strength explicitly. In the current literature, emerald green and chrome green have relatively strong bookcloth-use evidence, but open cloth-specific LAB baselines remain sparse; chrome green also lacks an open mixture-ratio-to-LAB dataset for `lead chromate + Prussian blue` cloth.
- Repeated LAB measurements are averaged to reduce single-point noise on textured cloth.
- Within-set `ΔE00` spread is used to downgrade borderline colour calls when the observed variability crosses the active decision band.
- The app now includes a conservative fading-aware allowance. When fading or browning is flagged, the observed hue stays near the class, and chroma drops relative to the nearest reference, the model may apply a small capped `ΔE00` allowance instead of treating the raw distance as final.
- The app now generates a scenario-based SOP from the screening result. The SOP does not create new evidence; it translates the existing colour, context, fading, and XRF signals into a repeatable handling-and-testing workflow.
- The handling copy distinguishes intrinsic toxicity from likely exposure during handling. Emerald green remains the clearest friability-driven exposure concern, while current project guidance suggests lower transfer risk for chrome yellow and chrome green despite their heavy-metal content.

## What Is Provisional

- The shipped `ΔE00` bands (`2.0`, `5.0`, `8.0`) are working defaults used by this app. This repository does not ship a validation dataset that proves those cutoffs for every device, substrate, and pigment class.
- The emerald-green-like, chrome-green-like, and mercury-red-like built-ins are fallback hue proxies only. They use generic ISCC-NBS swatches rather than chemically confirmed bookcloth references.
- The chrome-yellow-like built-ins are literature-derived historical pigment reconstructions, but they are still not same-device bookcloth measurements.
- The per-class evidence labels are qualitative summaries derived from the report and current source set. They help communicate confidence, but they are not statistical validation metrics.
- A class cannot receive the app's top-tier colour label unless uploaded local references exist for that class.
- The fading-aware allowance is heuristic rather than validated. It is capped at `ΔE00 2.0` against built-in fallback anchors and `ΔE00 1.0` against uploaded local references, and it only reduces the raw colour distance when the observed hue remains near the reference.

## Active Decision Policy

1. Use nearest-reference `ΔE00` as the main colour-distance metric and keep `ΔE76` as a secondary comparison.
2. Start from working bands at `ΔE00 <= 2.0`, `<= 5.0`, and `<= 8.0`.
3. If fading or browning is flagged, hue stays near the class, and chroma is reduced, the app may subtract a small capped fading allowance from the raw nearest-reference `ΔE00`. It reports both the raw and adjusted distance.
4. If repeated measurements are supplied, downgrade the colour band until the band threshold is larger than the observed within-set max `ΔE00`.
5. If a class is running only on built-in fallback anchors, keep the result provisional and do not issue a top-tier local-reference match.
6. Use XRF as elemental support only. Raman, FTIR, XRD, or equivalent analytical methods are still needed for compound-level confirmation.

## Recommended Reference Workflow

- Build a same-device local reference library whenever possible.
- Prefer references measured on chemically confirmed or institutionally trusted analogue cloth.
- Record condition and measurement context because ageing, backing, and weave can shift measured LAB values.

## Built-in Fallback Anchors In This Build

- `chrome-yellow-like` uses two literature-derived historical reconstruction anchors from Otero et al. 2012: `(76.0, 23.0, 71.0)` and `(85.0, 30.0, 89.0)`. These are the only built-ins in this repo that come from published `Lab*` values rather than named hue proxies.
- `emerald-green-like` uses five ISCC-NBS proxy terms from the bundled lookup: `Vivid green (82.8683, -61.3932, 20.8359)`, `Brilliant green (75.5566, -47.1290, 11.7506)`, `Strong green (51.0298, -39.5287, 10.7609)`, `Vivid bluish green (89.1853, -57.7669, 4.8481)`, and `Brilliant bluish green (78.3506, -41.9193, -7.2836)`.
- `chrome-green-like` uses six ISCC-NBS proxy terms from the bundled lookup: `Strong olive green (31.9101, -28.0718, 36.8394)`, `Moderate olive green (36.0115, -16.8489, 29.9763)`, `Dark olive green (19.1706, -16.2798, 22.2503)`, `Grayish olive green (36.9009, -5.6099, 9.7763)`, `Deep yellowish green (38.8698, -40.3240, 34.4110)`, and `Moderate yellowish green (55.8775, -22.6766, 16.2233)`.
- `mercury-red-like` uses six ISCC-NBS proxy terms from the bundled lookup: `Vivid red (46.0524, 67.8760, 32.0885)`, `Strong red (44.2955, 56.0735, 19.8348)`, `Deep red (28.8365, 48.0357, 16.6702)`, `Very deep red (18.0978, 38.0101, 6.1377)`, `Vivid reddish orange (52.2512, 64.3484, 56.4540)`, and `Strong reddish orange (55.2870, 47.3875, 43.4418)`.
- The app's current colour-distance policy bands are `ΔE00 <= 2.0`, `<= 5.0`, and `<= 8.0`. Those are project defaults for triage, not literature-validated universal thresholds for historical bookcloth.

## Provenance Note For The ISCC-NBS Proxy Table

- The chosen proxy names are defensible at the nomenclature level because they come from the ISCC-NBS color-name system described by Kelly and Judd (1955) and refined with centroid notations by Kelly (1958).
- The exact decimal `L*a*b*` values in `data/iscc_nbs_lab_colors.csv` are bundled with this repository and are internally reproducible, but this repo does not yet record the upstream derivation of that specific 267-row `Lab` table.
- Accordingly, the name-level choice is bibliographically defensible, but the exact decimal `Lab` proxies should still be treated as bundled working values until the table is regenerated from a cited source or the original upstream dataset is documented in-repo.

## Sources

- [Kelly and Judd 1955, The ISCC-NBS Method of Designating Colors and a Dictionary of Color Names](https://books.google.com/books/about/The_ISCC_NBS_Method_of_Designating_Color.html?id=5m_xi-CO4RoC)
- [Kelly 1958, Central Notations for the Revised ISCC-NBS Color-name Blocks](https://en.wikisource.org/wiki/Central_Notations_for_the_Revised_ISCC-NBS_Color-name_Blocks)
- [CIE, Colorimetry Part 6: CIEDE2000 Colour-Difference Formula](https://www.cie.co.at/publications/colorimetry-part-6-ciede2000-colour-difference-formula-1)
- [Gil et al. 2023, Detecting emerald green in 19thC book bindings using vis-NIR spectroscopy](https://research-portal.st-andrews.ac.uk/en/publications/detecting-emerald-green-in-19supthsupc-book-bindings-using-vis-ni/)
- [Vermeulen et al. 2023, Journal of Hazardous Materials abstract](https://www.sciencedirect.com/science/article/abs/pii/S0304389423007367)
- [Turner 2023, Lead and mercury in historical books - PubMed](https://pubmed.ncbi.nlm.nih.gov/37414706/)
- [Tedone and Grayburn 2022, Arsenic and Old Bookcloth](https://www.tandfonline.com/doi/abs/10.1080/01971360.2022.2031457)
- [Tedone and Grayburn 2023, Toxic Tomes abstract](https://journals.sagepub.com/doi/abs/10.1177/15501906231159040)
- [Otero et al. 2012, Chrome yellow in nineteenth century art: historic reconstructions of an artists' pigment](https://pubs.rsc.org/en/content/articlehtml/2012/ra/c1ra00614b)
- [Poison Book Project safer handling tips](https://sites.udel.edu/poisonbookproject/handling-and-safety-tips/)
- [Poison Book Project chrome yellow page](https://sites.udel.edu/poisonbookproject/chrome-yellow-bookcloth/)
