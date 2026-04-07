"""Splash page shown on initial load."""

from shiny import ui


def _nav_link(label: str, target_id: str) -> ui.Tag:
    """
    Bold link that clicks a nav button to navigate to a page.

    :param label: Display text for the link.
    :param target_id: The Shiny input ID of the nav button to click.

    """
    return ui.a(
        label,
        href="#",
        onclick=f"document.getElementById('{target_id}').click(); return false;",
        style="font-weight: bold; color: var(--color-nav);",
    )


def _feature_card(
    title: str,
    target_id: str,
    description: str,
    *,
    image: str | None = None,
) -> ui.Tag:
    """
    Feature card for the home page grid using Bootstrap card classes.

    :param title: Card heading (rendered as a nav link).
    :param target_id: Shiny input ID of the nav button to navigate to.
    :param description: Short description text below the title.
    :param image: Optional filename in ``www/`` to display above the title.

    """
    content = ui.div(
        ui.div(
            {"class": "fw-bold fs-5 mb-1"},
            _nav_link(title, target_id),
        ),
        ui.div(description),
    )
    if image is not None:
        return ui.div(
            {"class": "card mb-3"},
            ui.div(
                {"class": "card-body d-flex align-items-center gap-4"},
                ui.img(
                    src=image,
                    alt=title,
                    style="width:100px; height:100px; "
                    "object-fit:contain; flex-shrink:0;",
                ),
                content,
            ),
        )
    return ui.div(
        {"class": "card mb-3"},
        ui.div({"class": "card-body"}, content),
    )


def home_ui() -> ui.Tag:
    return ui.div(
        {"class": "p-4"},
        ui.div(
            {"class": "alert alert-warning", "role": "alert"},
            ui.strong("Under development: "),
            "excuse the mess. Projected release: April, 2026.",
        ),
        ui.h2("Welcome to the TF Binding and Perturbation Explorer"),
        ui.p(
            "Explore datasets of transcription factor (TF) binding and gene "
            "expression responses following TF perturbation. Compare growth "
            "conditions, experimental techniques, or analytic techniques. "
            "Currently, all datasets are for ",
            ui.em("Saccharomyces cerevisiae"),
            " (yeast).",
        ),
        ui.h3("How to"),
        ui.p(
            "The tabs above take you to pages for selecting and comparing " "datasets."
        ),
        ui.div(
            {"class": "mt-3"},
            _feature_card(
                "Dataset selection",
                "selection",
                "Choose which binding and perturbation datasets to include "
                "in your analysis.",
            ),
            _feature_card(
                "Binding",
                "binding",
                "Compare TF binding targets in the selected binding " "datasets.",
                image="binding.png",
            ),
            _feature_card(
                "Perturbation",
                "perturbation",
                "Compare transcriptional responses to TF perturbations in "
                "the selected perturbation datasets.",
                image="perturbation.png",
            ),
            _feature_card(
                "Comparison",
                "comparison",
                "Compare selected binding datasets to selected perturbation "
                "datasets.",
            ),
        ),
        ui.h3("Getting Started"),
        ui.p(
            "Begin with ",
            _nav_link("Dataset selection", "selection"),
            " to choose and filter the datasets you want to analyse, "
            "then navigate to the other tabs to explore the results.",
        ),
    )
