from shiny import module, ui


@module.ui
def home_ui():
    return ui.div(
        ui.h2("Welcome to the TF Binding and Perturbation Explorer"),
        ui.p(
            "This application provides an interactive interface for exploring "
            "datasets of transcription factor (TF) binding and gene expression "
            "responses following TF perturbation in yeast."
        ),
        ui.h3("How to Use This App"),
        ui.p(
            "Navigate through the tabs above to interact "
            "with different visualizations:"
        ),
        ui.tags.ul(
            ui.tags.li(
                ui.strong("Binding: "),
                (
                    "View TF binding profiles in multiple datasets and "
                    "compare the datasets to each other."
                ),
            ),
            ui.tags.li(
                ui.strong("Perturbation Response: "),
                (
                    "View transcriptional responses to TF perturbations "
                    "(gene deletion, gene overexpression, and TF degradation) "
                    "in multiple datasets and compare the datasets to each other."
                ),
            ),
            ui.tags.li(
                ui.strong(
                    "Compare binding datasets to perturbation response datasets: "
                ),
                ("This tab focuses on global statistics for many TFs."),
            ),
            ui.tags.li(
                ui.strong(
                    "Compare binding profiles to perturbation response profiles: "
                ),
                ("This tab focuses on individual TFs."),
            ),
        ),
        ui.h3("Getting Started"),
        ui.p(
            "Begin by selecting a tab to load a dataset and explore visual summaries "
            "of binding and expression response relationships."
        ),
        class_="home-content p-4",
    )
