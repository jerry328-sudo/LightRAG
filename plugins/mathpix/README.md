# Mathpix for LightRAG

This plugin registers `mathpix` as a PDF parser in LightRAG 1.5.9 and later.
Install it in the environment that runs the API server:

```sh
pip install ./plugins/mathpix
```

Mathpix OCR API credentials are an App ID and an App Key. Put
`MATHPIX_APP_ID` and `MATHPIX_APP_KEY` in the server's environment. With both
values set, an upload named `paper.[mathpix-iteP].pdf` selects Mathpix for that
file. To route all new PDFs to Mathpix, prepend `pdf:mathpix-iteP` to the existing
`LIGHTRAG_PARSER` rules. Use one of the existing document re-enqueue flows for
documents already bound to a different parser.

Optional settings are `MAX_PARALLEL_PARSE_MATHPIX` (default `2`),
`MATHPIX_POLL_INTERVAL_SECONDS` (default `3`), and
`MATHPIX_POLL_TIMEOUT_SECONDS` (default `600`).

The plugin uses the standard Mathpix `v3/pdf` API and requests a Markdown ZIP
with its images. It keeps the task ID and validated ZIP in the document's
`*.parsed/.mathpix/` directory. A timeout, local cancellation, or failed result
download keeps the task ID for the next retry. A successful parse can reuse
the cached ZIP. A terminal OCR failure allows a new upload on the next
explicit retry. If Mathpix accepted an upload but the submission connection
failed before returning a PDF ID, the plugin has no ID to reconcile.

For papers with images, the plugin also downloads and caches the same task's
`lines.json`. It attaches figure captions when the page geometry and Markdown
agree, and attaches footnotes with an explicit figure reference or layout link.
Ambiguous matches and unavailable layout leave the Markdown-based parse usable.
