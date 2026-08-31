# AI4PowerGrids workshop paper

Main source: `neurips_2026.tex`

Compile with pdfLaTeX and BibTeX:

```text
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

`build_paper_figures.py` regenerates the focused five-controller comparison
from `../experiment_metric_summary/selected_experiment_metrics_full.csv`.

The source uses the official NeurIPS 2026 `dblblindworkshop` option and is
anonymized for review. Chinese review translations are LaTeX comments placed
after the corresponding English paragraphs, so they do not appear in the PDF.
The AI4PowerGrids five-item domain checklist is included after the appendix.
The standard NeurIPS main-track `checklist.tex` is intentionally not included.

Before submission:

1. Confirm that the main text through the generative-AI disclosure occupies no
   more than four pages. References, appendices, and the domain checklist are
   outside that limit.
2. Verify that the PDF contains no author names, affiliations, acknowledgments,
   grant identifiers, repository usernames, or identifying links.
3. Confirm that checklist item 5 matches the anonymized supplementary material
   actually uploaded to OpenReview.
4. Keep the generative-AI disclosure immediately before the references and
   update it if additional tools are used.
5. Upload one PDF through OpenReview. Track 1 (methods with rigorous evaluation)
   is the recommended primary track for the current framing.
