"""UI functions for the Perturbation analysis page."""

from __future__ import annotations

from shiny import module, ui

from tfbpshiny.components import sidebar_label


@module.ui
def perturbation_ui() -> ui.Tag:
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h2("Perturbation"),
            ui.output_ui("execute_pending_style"),
            ui.input_action_button(
                "execute_analysis",
                "Execute Analysis",
                class_="btn-danger w-100",
            ),
            sidebar_label("Column"),
            ui.input_radio_buttons(
                "col_preference",
                label=None,
                choices={
                    "log10pval": ui.tooltip(
                        ui.span("-log10(p-value)"),
                        "Negative log10 of the p-value. "
                        "Values below 1e-10 are capped at 10.",
                    ),
                    "effect": ui.tooltip(
                        ui.span("Effect"),
                        "Raw effect size (e.g. log2 fold-change).",
                    ),
                    "pvalue": ui.tooltip(
                        ui.span("P-value"),
                        "Raw p-value. Smaller is more significant.",
                    ),
                },
                selected="effect",
                inline=True,
            ),
            sidebar_label("Correlation"),
            ui.input_radio_buttons(
                "corr_type",
                label=None,
                choices={"pearson": "Pearson", "spearman": "Spearman"},
                selected="pearson",
                inline=True,
            ),
            id="perturbation_sidebar",
            width=320,
            open="open",
        ),
        ui.h1("Perturbation Analysis"),
        ui.div(
            {"class": "sidebar-text"},
            ui.p(
                "Select score and correlation method in the sidebar. "
                "Correlations update automatically as selections change; "
                "click Execute Analysis to force a refresh."
            ),
            ui.p(
                "The Correlation Matrix tab shows median correlation for each "
                "dataset pair. Click a cell to select that pair."
            ),
            ui.p(
                "The Pair Distribution tab shows the per-regulator correlation "
                "distribution for the selected pair."
            ),
        ),
        ui.output_ui("hackett_pvalue_warning"),
        ui.output_ui("analysis_status"),
        ui.navset_tab(
            ui.nav_panel(
                "Correlation Matrix",
                ui.output_ui("corr_matrix_container"),
            ),
            ui.nav_panel(
                "Pair Distribution",
                ui.output_ui("regulator_selector_box"),
                ui.output_ui("pair_box_status"),
                ui.output_ui("pair_box_container"),
            ),
            id="perturbation_view_tabs",
        ),
    )


__all__ = ["perturbation_ui"]
