from logging import Logger

from shiny import Inputs, Outputs, Session, module, reactive, req, ui

from ..rank_response.expression_source_table_module import (
    expression_source_table_server,
    expression_source_table_ui,
)
from ..rank_response.main_table_module import (
    main_table_server,
    main_table_ui,
)
from ..rank_response.replicate_plot_module import (
    rank_response_replicate_plot_server,
    rank_response_replicate_plot_ui,
)
from ..utils.create_accordion_panel import create_accordion_panel

# This sets, statically but with definitions that will appear as tooltips, the
# selectable columns which may be displayed in the replicate details table
_rr_column_metadata = {
    "single_binding": (
        "Single Binding",
        "Unique ID for a single replicate; NA if composite.",
    ),
    "composite_binding": (
        "Composite Binding",
        "Unique ID for composite replicate; NA if single.",
    ),
    "expression_time": (
        "Expression Time",
        "Time point of McIsaac overexpression assay.",
    ),
    "univariate_rsquared": (
        "R²",
        "R² of model perturbed ~ binding.",
    ),
    "univariate_pvalue": (
        "P-value",
        "P-value of model perturbed ~ binding.",
    ),
    "binding_rank_threshold": (
        "Binding Rank Threshold",
        "binding rank with most significant DTO overlap.",
    ),
    "perturbation_rank_threshold": (
        "Perturbation Rank Threshold",
        "perturbation rank with most significant DTO overlap.",
    ),
    "binding_set_size": (
        "Binding Set Size",
        (
            "Gene count in binding set in DTO overlap. "
            "May be larger than rank due to ties."
        ),
    ),
    "perturbation_set_size": (
        "Perturbation Set Size",
        (
            "Gene count in perturbation set in DTO overlap. "
            "May be larger than rank due to ties."
        ),
    ),
    "dto_fdr": (
        "DTO FDR",
        "False discovery rate from DTO.",
    ),
    "dto_empirical_pvalue": (
        "DTO Empirical P-value",
        "Empirical p-value from DTO.",
    ),
    "rank_25": (
        "Rank at 25",
        "Responsive fraction in top 25 bound genes.",
    ),
    "rank_50": (
        "Rank at 50",
        "Responsive fraction in top 50 bound genes.",
    ),
}

# Convert to dictionary: {value: HTML label}
rr_choices_dict = {
    key: ui.span(label, title=desc)
    for key, (label, desc) in _rr_column_metadata.items()
}

# Initial selection for the replicate details table
_init_rr_selected = [
    "univariate_rsquared",
    "dto_fdr",
    "dto_empirical_pvalue",
    "rank_25",
]


@module.ui
def individual_regulator_compare_ui():

    general_ui_panel = create_accordion_panel(
        "General",
        ui.input_switch(
            "symbol_locus_tag_switch", label="Symbol/Locus Tag", value=False
        ),
        ui.input_select(
            "regulator",
            label="Select Regulator",
            selectize=True,
            selected=None,
            choices=[],
        ),
    )

    rr_columns_panel = create_accordion_panel(
        "Replicate Details Columns",
        ui.input_checkbox_group(
            "rr_columns",
            label="Replicate Metrics",
            choices=rr_choices_dict,
            selected=_init_rr_selected,
        ),
        ui.input_action_button(
            "update_table",
            "Update Table",
            class_="btn-primary mt-2",
            style="width: 100%;",
        ),
    )

    option_panels = [
        general_ui_panel,
        rr_columns_panel,
    ]

    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                *option_panels,
                id=module.resolve_id("input_accordion"),
                open=None,
                multiple=True,
            ),
            width="400px",
        ),
        ui.div(
            ui.div(
                ui.p(
                    "This page displays the rank response plots and associated "
                    "tables for a single regulator. Use the sidebar to select the "
                    "regulator of interest and the columns to be displayed in the ",
                    ui.tags.b("Replicate Details Tables"),
                    ". Hover over any of the column names "
                    "for information on what the column represents. ",
                    "Select row(s) in the ",
                    ui.tags.b("Main Selection Table"),
                    " on the left "
                    "to isolate a sample/samples in the plots and highlight the "
                    "corresponding rows in the replicate details table.",
                ),
                ui.tags.ul(
                    ui.tags.li(
                        ui.tags.b("Colored Lines: "),
                        "Each colored line represents a different binding dataset "
                        "replicate for the currently selected regulator.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Random Lines: "),
                        "The random expectation is calculated as the number of "
                        "responsive target genes divided by the total number of "
                        "target genes.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Gray Shaded Area: "),
                        "This area represents the 95% binomial distribution "
                        "confidence interval.",
                    ),
                ),
            ),
            ui.div(
                rank_response_replicate_plot_ui("rank_response_replicate_plot"),
                style="max-width: 100%; overflow-x: auto;",
            ),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Main Selection Table"),
                        main_table_ui("main_table"),
                        ui.card_footer(
                            ui.p(
                                ui.tags.b("How to use: "),
                                "Select rows in this table to filter and highlight "
                                "corresponding data in the replicate details table "
                                "and plots. "
                                "Multiple rows can be selected by holding Ctrl/Cmd "
                                "while clicking.",
                                style="margin: 0; font-size: 0.9em; color: #666;",
                            )
                        ),
                        style="height: 100%;",
                    ),
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("Replicate Details Table"),
                        ui.navset_tab(
                            ui.nav_panel(
                                "TFKO",
                                expression_source_table_ui("tfko_table"),
                            ),
                            ui.nav_panel(
                                "Overexpression",
                                expression_source_table_ui("overexpression_table"),
                            ),
                            id="expression_source_tabs",
                        ),
                        ui.card_footer(
                            ui.p(
                                ui.tags.b("Note: "),
                                "Rows corresponding to your main table selection are "
                                "automatically highlighted in orange. "
                                "This table shows detailed metrics for the selected "
                                "expression source. "
                                "Switch between tabs to view different expression "
                                "conditions.",
                                style="margin: 0; font-size: 0.9em; color: #666;",
                            )
                        ),
                        style="height: 100%;",
                    ),
                ),
                style="margin-top: 20px;",
            ),
        ),
    )


@module.server
def individual_regulator_compare_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    *,
    rank_response_metadata: reactive.ExtendedTask,
    bindingmanualqc_result: reactive.ExtendedTask,
    logger: Logger,
) -> None:
    """
    This function produces the reactive/render functions necessary to producing the
    individual_regulator_compare module which inclues the rank response plots and
    associated QC tables.

    :param rank_response_metadata: This is the result from a reactive.extended_task.
        Result can be retrieved with .result()
    :param bindingmanualqc_result: This is the result from a reactive.extended_task.
        Result can be retrieved with .result()
    :param logger: A logger object

    """
    # this reactive stores the selected promotersetsigs from the main table server.
    # This needs to be a reactive.value at the top in order to be shared across modules
    # which must be coded before the main table server is called.
    selected_promotersetsigs_reactive: reactive.value[set] = reactive.Value(set())

    # This reactive stores the columns selected from the side bar for
    # the rank response table
    selected_rr_columns: reactive.value[list] = reactive.Value(_init_rr_selected)

    @reactive.effect
    def _():
        """Update the regulator ui drop down selector based on the
        rank_response_metadata."""
        rank_response_metadata_local = rank_response_metadata.result()

        input_switch_value = input.symbol_locus_tag_switch.get()

        logger.info("switch: %s", input_switch_value)

        regulator_col = (
            "regulator_symbol" if input_switch_value else "regulator_locus_tag"
        )

        logger.info("regulator_col: %s", regulator_col)

        regulator_dict = (
            rank_response_metadata_local[["regulator_id", regulator_col]]
            .drop_duplicates()
            .sort_values(by=regulator_col)
            .set_index("regulator_id")
            .to_dict(orient="dict")
        )

        logger.info("regulator_dict: %s", regulator_dict.keys())

        ui.update_select("regulator", choices=regulator_dict)

    rr_metadata = rank_response_replicate_plot_server(
        "rank_response_replicate_plot",
        selected_regulator=input.regulator,
        selected_promotersetsigs=selected_promotersetsigs_reactive,
        logger=logger,
    )

    @reactive.calc
    def has_column_changes():
        """Reactive to check if there are changes in column selection."""
        current_selection = set(input.rr_columns.get() or [])
        confirmed_selection = set(selected_rr_columns.get())
        return current_selection != confirmed_selection

    @reactive.effect
    def _():
        """Update column choices for replicate details."""
        selected = list(input.rr_columns.get())

        ui.update_checkbox_group("rr_columns", selected=selected)

    # Update button appearance based on changes
    @reactive.effect
    def _():
        has_changes = has_column_changes()

        if has_changes:
            # Active state
            ui.update_action_button(
                "update_table",
                label="Update Table",
                disabled=False,
            )

        else:
            # Disabled state
            ui.update_action_button(
                "update_table",
                label="Update Table",
                disabled=True,
            )

    # Update the confirmed column selections when the button is clicked
    @reactive.effect
    @reactive.event(input.update_table)
    def _():
        req(input.rr_columns)
        selected_cols = list(input.rr_columns.get())
        selected_rr_columns.set(selected_cols)
        logger.debug("Updated table columns: %s", selected_cols)

    # Main table with selection capability
    selected_promotersetsigs = main_table_server(
        "main_table",
        rr_metadata=rr_metadata,
        bindingmanualqc_result=bindingmanualqc_result,
        logger=logger,
    )

    # Update the reactive value when main table selection changes
    @reactive.effect
    def _():
        selected_promotersetsigs_local = selected_promotersetsigs()
        selected_promotersetsigs_reactive.set(selected_promotersetsigs_local)
        logger.debug(
            "selected_promotersetsigs_reactive: %s", selected_promotersetsigs_reactive()
        )

    # Expression source tables
    expression_source_table_server(
        "tfko_table",
        rr_metadata=rr_metadata,
        expression_source="kemmeren_tfko",
        selected_promotersetsigs=selected_promotersetsigs_reactive,
        selected_columns=selected_rr_columns,
        logger=logger,
    )

    expression_source_table_server(
        "overexpression_table",
        rr_metadata=rr_metadata,
        expression_source="mcisaac_oe",
        selected_promotersetsigs=selected_promotersetsigs_reactive,
        selected_columns=selected_rr_columns,
        logger=logger,
    )
