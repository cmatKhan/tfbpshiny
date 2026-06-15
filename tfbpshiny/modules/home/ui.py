"""Splash page shown on initial load."""

from shiny import ui

# Maps card link id -> nav panel title string used in main_nav.
HOME_CARD_NAV_TARGETS: dict[str, str] = {
    "home_nav_selection": "Dataset selection",
    "home_nav_binding": "Binding",
    "home_nav_perturbation": "Perturbation",
    "home_nav_comparison": "Binding/Perturbation Comparisons",
}


def _feature_card(
    title: str, description: str, link_id: str | None = None, img_src: str | None = None
) -> ui.Tag:
    """
    Feature card for the home page grid using Bootstrap card classes.

    :param title: Card heading.
    :param description: Short description text below the title.
    :param link_id: Optional input id for an ``action_link`` wrapping the title.
        When provided, clicking the title navigates to the corresponding tab.
    :param img_src: Optional path to an image shown at the left of the card body.

    """
    body_children: list[ui.Tag] = []
    if img_src is not None:
        body_children.append(
            ui.img(
                {
                    "src": img_src,
                    "style": "width:64px; height:64px; object-fit:contain;"
                    " flex-shrink:0; margin-right:1rem;",
                }
            )
        )
    title_tag: ui.Tag = (
        ui.input_action_link(link_id, title, class_="fw-bold fs-5 mb-1 d-block")
        if link_id is not None
        else ui.div({"class": "fw-bold fs-5 mb-1"}, title)
    )
    body_children += [
        ui.div(
            title_tag,
            ui.div(description),
        )
    ]
    return ui.div(
        {"class": "card mb-3"},
        ui.div(
            {
                "class": "card-body d-flex align-items-center",
            },
            *body_children,
        ),
    )


def home_ui() -> ui.Tag:
    return ui.div(
        {"class": "p-4"},
        ui.h2("Welcome to the TF Binding and Perturbation Explorer"),
        ui.p(
            "Explore datasets of transcription factor (TF) binding and gene "
            "expression responses following TF perturbation. Compare growth "
            "conditions, experimental techniques, or analytic techniques. "
            "Currently, all datasets are for ",
            ui.em("Saccharomyces cerevisiae"),
            " (yeast).",
        ),
        ui.h3("Getting Started"),
        ui.p(
            "Use the tabs above to navigate between pages. "
            "Start with Dataset selection to choose which datasets to analyse."
        ),
        ui.div(
            {"class": "mt-3"},
            _feature_card(
                "Dataset selection",
                "Begin here to choose and filter the datasets you want to "
                "analyse, then navigate to the other tabs to explore the results.",
                link_id="home_nav_selection",
            ),
            _feature_card(
                "Binding",
                "Compare TF binding targets in the selected binding datasets.",
                link_id="home_nav_binding",
                img_src="binding.png",
            ),
            _feature_card(
                "Perturbation",
                "Compare transcriptional responses to TF perturbations in "
                "the selected perturbation datasets.",
                link_id="home_nav_perturbation",
                img_src="perturbation.png",
            ),
            _feature_card(
                "Comparison",
                "Compare selected binding datasets to selected perturbation datasets.",
                link_id="home_nav_comparison",
            ),
        ),
    )
