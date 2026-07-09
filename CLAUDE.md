# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`JIN_pylib` is a Python library by Dr. Ge Jin (Colorado School of Mines /
Alaska DFOS & USGS research) providing shared data structures and file
readers for processing Distributed Acoustic Sensing (DAS), Distributed
Temperature Sensing (DTS), pump-curve, and related fiber-optic sensing data.
Per `README.md`: "developed ... for research and education purposes. This is
not developed for public use."

There is no package manifest (no `setup.py`/`pyproject.toml`), no build
system, and no test suite — this is a flat collection of modules imported
directly by consuming notebooks/scripts. It is not pip-installed; consumers
do `sys.path.append('/path/to/parent/of/JIN_pylib')` and then
`from JIN_pylib import Data2D_XT, Spool, gjsignal, ...`. `JIN_pylib/__init__.py`
is empty, so always import submodules directly rather than expecting
anything re-exported at the package level.

This library is developed as its own repo
(`github.com/jinwar/JIN_pylib`) but is also vendored/used from at least one
downstream research repo (the Alaska DFOS notebook workspace at the parent
directory), where it is imported with `%autoreload 2` so edits here take
effect without restarting a kernel. Keep that consumption pattern in mind —
changes here can affect running notebooks elsewhere on disk that don't share
this git history.

## Common commands

- **Push changes**: `./git_push.sh "commit message"` — a thin wrapper around
  `git commit -a -m "$1" && git push origin master`. Note it uses
  `commit -a`, i.e. it stages and commits *all* tracked-file changes, not
  just what you intend — check `git status`/`git diff` first if you didn't
  write every pending change yourself.
- **No lint/test/build commands exist.** There is no CI, linter config, or
  test runner in this repo. Validate changes by exercising them through the
  example notebooks in `examples/` (`jin_pylib_demonstration.ipynb`, `Spool
  Processing.ipynb`, `FK Velocity Analysis.ipynb`, `dispersion
  analysis.ipynb`, `Data3D examples.ipynb`), e.g.:
  ```bash
  jupyter nbconvert --to notebook --execute --inplace "examples/Spool Processing.ipynb"
  ```
  or inspect a notebook's logic without running it:
  ```bash
  jupyter nbconvert --to script --stdout "examples/Spool Processing.ipynb"
  ```

## Core architecture

- **`BasicClass.BasicClass`** — mixin providing `save`/`load` (pickle) and
  `copy` (deepcopy). Used by `Data1D.PumpCurve` and the `FracHitPick`
  classes.
- **`Data2D_XT.Data2D`** — the central data object: a 2D `data` array
  (channel/depth × time), `taxis` (seconds from `start_time`), `daxis`
  (aliased as `.mds` via a property for backwards compatibility), `chans`,
  `attrs`, and a `history` list that filtering/processing methods append
  human-readable descriptions to. Key method groups:
  - Filtering: `bp_filter`/`hp_filter`/`lp_filter` (via `gjsignal`,
    Butterworth, optional `edge_taper`), `median_filter`, `down_sample`
    (anti-alias low-pass then decimate).
  - Windowing: `select_time`/`select_depth` (return-copy or in-place via
    `makecopy`), `window_data_time`/`window_data_depth` (in-place only).
  - Gaps/regularization: `fill_gap_zeros`, `fill_gap_interp`,
    `remove_duplicate_time`, `interp_time`.
  - Derived transforms: `take_gradient`, `take_time_diff`,
    `apply_gauge_length` (differencing across channels, e.g. Treble strain
    from velocity), `cumsum`.
  - Plotting: `plot_waterfall` (imshow with optional real-timestamp x-axis
    via `VizUtil.PrecisionDateFormatter`), `plot_wiggle`, `plot_simple_waterfall`.
  - IO: `saveh5`/`loadh5` (module-level `load_h5`) write/read one `Data2D` as
    an HDF5 file — `data` plus its attrs live in one dataset, every other
    `__dict__` attribute (taxis, daxis, chans, history, ...) is saved as a
    sibling dataset. `Patch_to_Data2D` converts from a dascore-style `Patch`.
  - `merge_data2D(data_list, daxis=None)` — module-level function that
    concatenates multiple `Data2D` objects in time, sorting by `start_time`;
    if `daxis` is given (array or an index into `data_list`), each patch is
    depth-interpolated onto a common axis first (`scipy.interpolate.interp1d`)
    before concatenation. This is the core stitching primitive `Spool` uses
    internally.
- **`Data3D_XTF.Data3D`** — 3D (channel × time × frequency) sibling of
  `Data2D`, used for spectrogram-like data. Reuses `Data2D`'s `saveh5`/`loadh5`
  via direct method assignment (`Data3D.saveh5 = Data2D_XT.Data2D.saveh5`).
  `get_FBA(freq_min, freq_max)` collapses the frequency axis (band average)
  into a `Data2D`; `get_channel_spectrum()` collapses the channel axis into a
  `Data2D` whose `taxis` is actually `faxis`. `get_FBA_from_files` batches
  `get_FBA` across files and merges with `merge_data2D`.
- **`Data1D.PumpCurve`** (`BasicClass` subclass) — wraps a pandas DataFrame of
  pump/pressure curve columns plus a time axis, with multi-column/twin-axis
  plotting (`plot_multi_cols`, `plot_single_col`) for co-plotting against DAS
  waterfalls (see `VizUtil.CoPlot_Waterfall_Pumping`).
- **`Spool.spool`** — a lazy, multi-file dataset index "mimicking dascore's
  spool structure". Backed by a DataFrame (`_df`: `file`, `start_time`,
  `end_time`) plus a `reader` callable, with an internal `OrderedDict` LRU
  cache (`_cashe`, size-limited in GB via `set_cashe_size`). Two read modes,
  chosen by `support_partial_reading`:
  - `False` (`_get_data_nopl`): `reader(filename)` loads a *whole* file into
    the cache, then `Data2D.select_time` windows it in memory.
  - `True` (`_get_data_pl`): `reader(filename, bgtime, edtime)` reads only
    the needed window directly (no caching needed/used).
  `get_data(bgtime, edtime)` dispatches to one of these and calls
  `merge_data2D` to stitch results spanning multiple files.
  `get_time_segments(max_dt=None)` finds contiguous-coverage runs (gap
  threshold defaults to 1.5× the median inter-file gap); `get_chunks(length,
  overlap, is_partial)` breaks those runs into fixed-length windows for batch
  processing. `save_pickle`/`load_pickle` (module-level `load_pickle`
  function too) persist only the file index, not raw data — regenerate
  after raw data moves. `spool.__add__` concatenates two spools' indices.
  - **`sp_process`** / **`sp_process_pipeline`** (bottom of `Spool.py`) —
    batch-process an entire spool chunk-by-chunk: `get_data` →
    `pre_process` → `process_fun` → `post_process` → accumulate → flush to
    HDF5 once buffered output exceeds `save_file_size` MB. `_pipeline` is the
    same thing overlapping the next chunk's IO with the current chunk's
    processing via a `ThreadPoolExecutor` pair (IO executor has
    `max_workers=1`, so IO is still serialized, just overlapped with CPU
    work).
- **`readers/` package** — per-instrument-format readers that all follow the
  same three-function contract so they can back a `Spool.spool`:
  - `get_time_range(file) -> (start_time, end_time)` — cheap, metadata-only.
  - `reader(file)` or `reader(file, bgtime, edtime) -> Data2D` — depending on
    whether the format supports partial reads.
  - `create_spool(datapath, ...)` — walks `datapath` with a glob pattern,
    calls `get_time_range` on every match, and builds a `Spool.spool` from
    the results. Implemented via the shared helper
    `readers/reader_utils.create_spool_common(datapath, get_time_range,
    reader, search_pattern, search_subdirs, support_partial_reading)` — new
    readers should call this rather than reimplementing the indexing loop.
  Existing readers: `HAL_DAS_Reader.py` (also exposes
  `create_spool_from_files` for an explicit file list rather than a glob),
  `ONYX_DAS_Reader.py`, `Silixa_DAS_Reader.py`, `Data2D_DAS_Reader.py`
  (generic Data2D-shaped HDF5, whole-file reads only),
  `HFTS2_FiberSegy_Reader.py`, `HAL_DTS_CSV_Reader.py`, and
  `readers/Terra15_Reader.py` (added most recently — reads Terra15's
  `data_product/{posix_time,data}` HDF5 layout, partial-read capable). When
  adding a new vendor format, follow this same three-function contract
  rather than inventing a new interface.
  - **Exception**: root-level `TrebleReader.py` (`read_Treble`, `Treble_io`,
    `Treble_io_colab`) predates and does not follow the `readers/` contract —
    it's class-based and builds its own file database rather than a
    `Spool.spool`. Treat it as legacy/one-off rather than a pattern to copy
    for new readers.
- **`ProcessUtil.py`** — FK (frequency-wavenumber) transform/filtering
  (`Data2D_fft2`, `Data2D_fkfilter_maskgen`/`_applymask`/`_velocity`),
  velocity/dispersion analysis (`fk_velocity_analysis[_viz]`,
  `dispersion_analysis`, `plot_disp_result`), and STA/LTA event-detection
  helpers (`tranform_STALTA`, `select_stalta_triggers`) — all operate on
  `Data2D` objects and return either transformed `Data2D`s or plain
  arrays/dicts for the `_viz`/plotting helpers to consume.
- **`gjsignal.py`** — general-purpose DSP building blocks (Butterworth
  filter design/application, amplitude spectrum, cross-correlation &
  time-shift estimation, quadratic interpolation matrices, STA/LTA on 1D/2D
  arrays, unit conversions) that `Data2D_XT` and `ProcessUtil` build on top
  of. Prefer adding new generic signal-processing primitives here rather
  than inline in a class method.
- **`VizUtil.py`** — matplotlib/ipympl interactive plotting: `CoPlot_Simple`
  / `CoPlot_Waterfall_Pumping` (DAS waterfall + `PumpCurve` co-plots with
  ipywidgets controls), `Interactive_Waterfall`, `TDSlice` (linked
  time/depth slice viewer), and `PrecisionDateFormatter` (sub-second-aware
  matplotlib date tick formatter used throughout `Data2D_XT`/`Data1D`
  plotting methods).
- **`FracHitPick.py` / `FracHitPick_Simple.py`** — interactive
  (matplotlib-event-driven) hydraulic-fracturing "frac hit" picking/labeling
  tools; pickle-persistable, `FracHitPick.py`'s classes via `BasicClass`.
- **`NB_util.py`**, **`Optasense_Util.py`**, **`IOUtil.py`**,
  **`TimeUtil.py`** — smaller format-specific or IO helper utilities (NB CSV
  reading, OptaSense folder indexing by filename timestamp, SEGY export
  (`IOUtil.savesegy`) and HDF5 structure inspection, timestamp-string
  parsing).

## Conventions to preserve when editing

- `Data2D` methods that mutate in place also append a short descriptive
  string to `self.history` — do this for any new processing method so
  provenance stays reconstructable from `print_info()`/`history`.
- Time is generally represented two ways in this codebase: an absolute
  `start_time` (Python `datetime`) plus a relative `taxis` (float seconds
  from `start_time`). Prefer this pair over storing absolute timestamps
  per-sample; use `get_datetime()`/`get_datetime64()`/`get_mdates_taxis()` to
  materialize absolute timestamps only when needed (e.g. for plotting).
- `daxis`/`mds` are the same underlying attribute (property alias for
  backward compatibility) — don't reintroduce a separate `mds` field.
