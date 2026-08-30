# AI4PowerGrids workshop paper

Main file: `neurips_2026.tex`

Compile with pdfLaTeX and BibTeX:

```text
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

The source uses the official NeurIPS 2026 `dblblindworkshop` option and is
anonymized for review. The AI4PowerGrids-specific five-item domain checklist is
in `domain_checklist.tex` and is included after the appendix. Do not input the
official main-track `checklist.tex`: the workshop call explicitly says that the
standard NeurIPS paper checklist is not required.

Before submission:

1. Compile and confirm that the main text through `Conclusion` occupies no more
   than four pages. References, appendices, and the domain checklist are outside
   that limit.
2. Verify that the PDF contains no author names, affiliations, acknowledgments,
   grant identifiers, repository usernames, or identifying links.
3. Confirm the code/data release statement in checklist item 5 matches what is
   actually uploaded to OpenReview.
4. Keep the generative-AI disclosure immediately before the references and
   update it if additional tools are used.
5. Upload one PDF through OpenReview and select the primary workshop track that
   best matches the final framing. Track 1 (methods with rigorous evaluation) is
   the current recommended choice; Track 4 is also plausible if the paper is
   reframed primarily around routing collapse and tail-comfort failure.
