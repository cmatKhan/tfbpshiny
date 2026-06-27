"""
Styled UI component library for TFBPShiny.

This module is the single source of truth for all reusable, styled Shiny UI
elements.  Every component here maps to one or more CSS classes in ``app.css``
and the mapping is documented in the function's docstring.

Purpose
-------
When the application logic is stable and it is time to focus on appearance,
changes to visual design should be made here (or in ``app.css``) rather than
scattered across individual module ``ui.py`` files.  Keeping structure and
styling in one place means:

  - A class rename in ``app.css`` requires only one call-site change here.
  - New structural variants (e.g. a compact sidebar row) can be added as
    keyword arguments without touching caller code.
  - The component list serves as living documentation of which CSS classes are
    in active use.

Maintenance rules
-----------------
- **Add a component** whenever a new CSS class is introduced in ``app.css`` and
  used in more than one place in the app.
- **Update a component** whenever its underlying CSS class is renamed or its
  structural HTML changes (e.g. a new wrapper div is added).
- **Do not** put business logic or reactive code here — components are pure
  ``ui.Tag`` factories.
- Components that require a Shiny ``id`` accept it as the first positional
  argument; purely structural wrappers use ``*children``.

CSS variable reference (from ``app.css`` ``:root``)
----------------------------------------------------
--color-primary        #2C7A7B
--color-primary-dark   #1A5456
--color-border         #E2E8F0
--color-text           #1A202C
--radius-sm            6px
--font-size-label      0.875rem
--transition-fast      150ms ease

"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import faicons as fa
from shiny import ui

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

try:
    _version = version("tfbpshiny")
except PackageNotFoundError:
    _version = "dev"

_GITHUB_URL = "https://github.com/BrentLab/tfbpshiny"

_GITHUB_SVG_PATH = (
    "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 "
    "7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-"
    "2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
    "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-"
    "1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-"
    "1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
    "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56"
    ".82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 "
    "0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 "
    "8c0-4.42-3.58-8-8-8z"
)


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------


def tooltip(
    trigger: ui.Tag,
    text: str,
    *,
    placement: Literal["auto", "top", "right", "bottom", "left"] = "right",
) -> ui.Tag:
    """
    Wrap a UI element with a Bootstrap tooltip shown on hover.

    CSS: ``.tooltip-inner`` — overrides Bootstrap defaults to set
    ``max-width: 300px`` and ``text-align: left``.

    This is a simplified wrapper around ``ui.tooltip`` that exposes only
    ``trigger``, ``text``, and ``placement``.  If you need ``id``,
    ``options``, or extra tag attributes, extend this function rather than
    bypassing it.

    :param trigger: The element the user hovers over to reveal the tooltip.
    :param text: Plain-text content displayed inside the tooltip bubble.
    :param placement: Where the tooltip appears relative to the trigger.
        Defaults to ``"right"`` (instead of Shiny's ``"auto"``) to suit
        the sidebar layout where tooltips are most commonly used.

    """
    return ui.tooltip(trigger, text, placement=placement)


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------


def sidebar_label(text: str) -> ui.Tag:
    """
    Block-level section label for grouping controls within a sidebar.

    Uses Bootstrap's ``form-label`` class for consistent typography and
    spacing with other form elements, plus ``mt-3`` for separation between
    sections (Bootstrap's spacing utilities handle responsive scaling).
    ``mb-1`` keeps the gap to the control tight.

    CSS: Bootstrap ``form-label mt-3 mb-1``

    """
    return ui.p({"class": "form-label mt-3 mb-1"}, text)


# ---------------------------------------------------------------------------
# Workspace typography
# ---------------------------------------------------------------------------


def workspace_heading(text: str) -> ui.Tag:
    """
    Primary workspace page title (``h1``-level, 20 px, 600 weight).

    CSS: ``.workspace-header h1``

    """
    return ui.h1(text)


# ---------------------------------------------------------------------------
# Empty / placeholder states
# ---------------------------------------------------------------------------


def empty_state(*children: Any, compact: bool = False) -> ui.Tag:
    """
    Centred placeholder shown when there is nothing to display yet.

    CSS: ``.empty-state`` / ``.empty-state.compact``

    :param compact: Use reduced vertical padding (20 px vs 64 px) for inline
        placeholders inside a partially-populated workspace.

    """
    cls = "empty-state compact" if compact else "empty-state"
    return ui.div({"class": cls}, *children)


# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------


def github_badge() -> ui.Tag:
    """
    GitHub repo link with version pill, displayed at the right end of the nav bar.

    CSS: ``.github-badge``, ``.github-badge-version``

    """
    return ui.a(
        {"class": "github-badge", "href": _GITHUB_URL, "target": "_blank"},
        ui.tags.svg(
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "width": "16",
                "height": "16",
                "viewBox": "0 0 16 16",
                "fill": "currentColor",
                "style": "vertical-align:middle; margin-right:5px;",
            },
            ui.Tag("path", d=_GITHUB_SVG_PATH),
        ),
        ui.tags.span(
            {"style": "vertical-align:middle; margin-right:6px;"},
            "BrentLab/tfbpshiny",
        ),
        ui.tags.span({"class": "github-badge-version"}, f"v{_version}"),
    )


# ---------------------------------------------------------------------------
# Dataset selection row (Select Datasets sidebar)
# ---------------------------------------------------------------------------


def dataset_row(toggle: ui.Tag, label: str, filter_button: ui.Tag) -> ui.Tag:
    """
    Single dataset row: toggle switch + display name + filter button on one line.

    CSS: ``.dataset-row``, ``.dataset-row-label``, ``.dataset-item``

    :param toggle: A fully-constructed ``ui.input_switch`` element.
    :param label: Human-readable dataset display name.
    :param filter_button: A fully-constructed ``ui.input_action_button`` element
        (use :func:`filter_button` to build it).
    """
    return ui.div(
        {"class": "dataset-row dataset-item"},
        ui.div(
            {"class": "dataset-row-left"},
            toggle,
            ui.span({"class": "dataset-row-label sidebar-text"}, label),
        ),
        filter_button,
    )


def dataset_list(*rows: ui.Tag) -> ui.Tag:
    """
    Vertical stack of ``dataset_row`` elements with a small gap between them.

    CSS: ``.dataset-list``

    """
    return ui.div({"class": "dataset-list"}, *rows)


def filter_button(id: str) -> ui.Tag:
    """
    Small "Filter" button at the right of each dataset row.

    CSS: ``.btn-filter-dataset``

    """
    return ui.input_action_button(
        id,
        "Filter",
        class_="btn btn-sm btn-outline-secondary btn-filter-dataset",
    )


# ---------------------------------------------------------------------------
# Filter modal building blocks
# ---------------------------------------------------------------------------


def filter_option_card(title: str, *controls: ui.Tag) -> ui.Tag:
    """
    Bordered card containing a single filter control (slider, selectize, switch).

    Uses Bootstrap ``.card`` / ``.card-body`` for structure. The title is rendered
    as bold text in a flex header row; controls appear below it.

    :param title: Field name shown in bold at the top of the card.
    :param controls: One or more Shiny input elements placed below the header.

    """
    return ui.div(
        {"class": "card"},
        ui.div(
            {"class": "card-body p-2"},
            ui.div(
                {
                    "class": "d-flex align-items-center "
                    "justify-content-between gap-2 mb-2"
                },
                ui.span({"class": "fw-bold small"}, title),
            ),
            *controls,
        ),
    )


def modal_section(*cards: ui.Tag) -> ui.Tag:
    """
    Vertical stack of ``filter_option_card`` elements inside a modal column.

    Uses Bootstrap ``d-flex flex-column gap-2``.

    """
    return ui.div({"class": "d-flex flex-column gap-2"}, *cards)


# ---------------------------------------------------------------------------
# Intersection matrix table
# ---------------------------------------------------------------------------


def matrix_cell_button(id: str, label: str, *, tooltip: str | None = None) -> ui.Tag:
    """
    Full-width, borderless button that fills a matrix table cell.

    Emits a plain ``<button>`` that calls ``Shiny.setInputValue`` with
    ``{priority: "event"}`` on click instead of using
    ``ui.input_action_button``.  This avoids reactive loops when the button is
    rendered inside a ``render.ui`` output: Shiny action buttons re-register on
    every render and trigger their reactive listeners, whereas plain buttons
    only fire when the user physically clicks them.

    CSS: ``.matrix-cell-button``

    :param id: Shiny input ID written by ``Shiny.setInputValue`` on click.
    :param label: Text displayed inside the button.
    :param tooltip: When provided, sets the native ``title`` attribute so
        browsers show a hover tooltip.

    """
    attrs: dict[str, str | None] = {
        "class": "matrix-cell-button",
        "onclick": f"Shiny.setInputValue('{id}', Math.random(), {{priority: 'event'}})",
    }
    if tooltip is not None:
        attrs["title"] = tooltip
    return ui.tags.button(label, **attrs)


def matrix_header_cell(label: str, *, row: bool = False) -> ui.Tag:
    """
    Header cell (``<th>``) for the intersection matrix.

    CSS:

    - ``row=False`` (default) — column header: ``.matrix-col-header``,
      ``.matrix-header-name``. Used for each dataset column in the top row.
    - ``row=True`` — row header: ``.matrix-row-header``. Used for the first
      ``<th>`` in the header row (typically labelled ``"Dataset"``).

    :param label: Text shown in the header cell.
    :param row: When ``True``, renders as a row header rather than a column header.

    """
    if row:
        return ui.tags.th({"class": "matrix-row-header"}, label)
    return ui.tags.th(
        {"class": "matrix-col-header"},
        ui.div({"class": "matrix-header-name"}, label),
    )


def matrix_row_label(label: str | ui.Tag) -> ui.Tag:
    """
    Row label cell (``<td>``) showing the dataset name at the start of each row.

    CSS: ``.matrix-row-label``

    :param label: Dataset display name or a tag (e.g. an anchor link).

    """
    return ui.tags.td({"class": "matrix-row-label"}, label)


def matrix_cell(
    kind: Literal["empty", "diagonal", "interactive"],
    button: ui.Tag | None = None,
    *,
    active: bool = False,
    pending: bool = False,
) -> ui.Tag:
    """
    Data cell (``<td>``) in the intersection matrix.

    CSS by ``kind``:

    - ``"empty"`` — lower-triangle placeholder: ``.matrix-cell-empty``.
      No button; ``button`` argument is ignored.
    - ``"diagonal"`` — on-diagonal cell showing regulator/sample counts for one
      dataset: ``.matrix-cell-diagonal``. Wraps a ``matrix_cell_button``.
    - ``"interactive"`` — upper-triangle cell showing the common-regulator count
      for a dataset pair: ``.matrix-cell-interactive``. When ``active=True``
      adds ``.matrix-cell-active`` for the committed regulator filter pair;
      when ``pending=True`` adds ``.matrix-cell-pending`` for a queued but
      uncommitted pair. ``active`` takes precedence over ``pending``.

    :param kind: One of ``"empty"``, ``"diagonal"``, or ``"interactive"``.
    :param button: A ``matrix_cell_button`` element. Required for ``"diagonal"``
        and ``"interactive"``; ignored for ``"empty"``.
    :param active: Marks the selected item (e.g. committed regulator filter pair
        or currently selected correlation pair).
    :param pending: Marks a queued regulator filter pair not yet committed.

    """
    if kind == "empty":
        return ui.tags.td({"class": "matrix-cell-empty"}, "")
    if kind == "diagonal":
        return ui.tags.td({"class": "matrix-cell-diagonal"}, button)
    # interactive
    if active:
        cls = "matrix-cell-interactive matrix-cell-active"
    elif pending:
        cls = "matrix-cell-interactive matrix-cell-pending"
    else:
        cls = "matrix-cell-interactive"
    return ui.tags.td({"class": cls}, button)


def pending_regulator_banner(
    display_a: str,
    display_b: str,
    n_common: int,
) -> ui.Tag:
    """
    Persistent inline notification shown when a regulator filter is queued.

    Appears above the matrix table. Not a modal — clicking outside has no effect.
    The Cancel button emits ``input.cancel_pending_regulator``.

    CSS: ``.pending-regulator-banner``

    :param display_a: Human-readable name of the first dataset.
    :param display_b: Human-readable name of the second dataset.
    :param n_common: Number of common regulators in the pending filter.

    """
    return ui.div(
        {"class": "pending-regulator-banner"},
        ui.div(
            {"class": "pending-regulator-banner-body"},
            ui.span(
                {"class": "pending-regulator-banner-text"},
                ui.strong(f"{n_common:,} common regulators"),
                f" between {display_a} and {display_b} queued as a filter. "
                "Click Apply in the sidebar to commit.",
            ),
            ui.input_action_button(
                "cancel_pending_regulator",
                "Cancel",
                class_="btn btn-sm btn-outline-secondary",
            ),
        ),
    )


def matrix_table(header_row: ui.Tag, *body_rows: ui.Tag) -> ui.Tag:
    """
    Full intersection matrix ``<table>``.

    CSS: ``.matrix-summary-table``

    :param header_row: A ``<tr>`` built from ``matrix_row_header`` and
        ``matrix_col_header`` cells.
    :param body_rows: One ``<tr>`` per active dataset, built from
        ``matrix_row_label``, ``matrix_cell_empty``, ``matrix_cell_diagonal``,
        and ``matrix_cell_interactive`` cells.

    """
    return ui.tags.table(
        {"class": "matrix-summary-table"},
        ui.tags.thead(header_row),
        ui.tags.tbody(*body_rows),
    )


def export_download_button(id: str) -> ui.Tag:
    """
    Full-width download button for exporting selected datasets as a tarball.

    CSS: ``.btn-export-datasets``

    :param id: Shiny download ID (paired with a ``@render.download`` handler).

    """
    return ui.download_button(
        id,
        "Export Selected Datasets",
        icon=fa.icon_svg("download", width="14px", height="14px"),
        class_="btn-export-datasets",
    )


__all__ = [
    # tooltips
    "tooltip",
    # typography
    "sidebar_label",
    "workspace_heading",
    # states
    "empty_state",
    # nav
    "github_badge",
    # dataset selection
    "dataset_row",
    "dataset_list",
    "filter_button",
    # filter modal
    "filter_option_card",
    "modal_section",
    # matrix
    "matrix_cell_button",
    "matrix_header_cell",
    "matrix_row_label",
    "matrix_cell",
    "matrix_table",
    "pending_regulator_banner",
    # export
    "export_download_button",
]
