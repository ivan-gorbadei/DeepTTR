"""Charts of a training run: a page to follow it in a browser, and static pictures of it.

- `python -m deepttr.charts RUN_ROOT [--port 8090] [--host 127.0.0.1]`: the chart page (`server`)
- `python -m deepttr.charts.export RUN_ROOT OUTDIR`: SVG files of the charts that show each phase's
  skill being learned (`export`)
- `deepttr.charts.data`: what both draw, read from the run's files; it needs nothing beyond
  Python's standard library
"""
