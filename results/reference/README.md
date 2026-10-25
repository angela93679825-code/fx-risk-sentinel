# Reference results

Generated with the current modular repository code on original local inputs, separately from the synthetic demo.

- full_rebuilt_metrics.csv: indicators reconstructed and country-namespaced macro fields reselected using the historical full-sample filter.
- supplied_feature_metrics.csv: original archived selected-feature matrices.
- single_asset.csv: Appendix L rules, rounded values match Table 10; no commissions or volatility sizing.
- portfolio_metrics.json: original 365-day arithmetic annualization, explicit asset order and original risk gate.
- target_comparison.json: target alignment comparison from the earlier source audit, unaffected by the feature refactor.
- provenance.json: source hashes; no raw source data are bundled.

These feature variants must not be treated as the same experiment. See docs/REPRODUCTION.md for the source-to-repository changes and look-ahead limitations.
