# Report 2

This folder follows the same multi-file LaTeX structure and page format as
`report/`.

For Overleaf:

1. Upload the complete `report2` folder, including `figures/`.
2. Set `thesis.tex` as the main document.
3. Select **XeLaTeX** as the compiler because the report contains Chinese
   translations.

The plots are generated only from recorded experiment outputs:

```text
python build_figures.py
```

The script expects the repository-level `experiment_metric_summary` folder. The
PNG files are already included, so Overleaf does not need Python or the source
CSV files.
