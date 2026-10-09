# Contributing

Before a change, describe the concrete paper-reading failure it fixes. Use an original PDF fixture or a document you have permission to redistribute. Include an expected page, quote, table cell or refusal behavior.

Run `python -m unittest discover -s tests -v`. Parser, retrieval and citation-validation changes need regression evidence; documentation-only edits do not need new tests. Do not add unmeasured accuracy claims to the README.

Useful contributions include table-row provenance, multi-column regression fixtures, cross-language retrieval evaluation, and held-out answerability annotations. Keep model and parser integrations optional until their performance and costs have been measured.
