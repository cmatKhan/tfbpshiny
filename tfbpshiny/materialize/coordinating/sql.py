"""
SQL generators for the coordinating layer of the materialized DuckDB schema.

All functions return pure SQL strings (no side effects) so they can be called
from a Jupyter notebook to inspect output before running the full pipeline.
The coordinator is the only code that calls ``.execute()``.
"""

from __future__ import annotations

from labretriever import VirtualDB

from tfbpshiny.utils.vdb_init import HIDDEN_FILTER_FIELDS

# ---------------------------------------------------------------------------
# Mapping: db_name → (hf_repo, hf_config)
# Derived from brentlab_yeast_collection.yaml.
# ---------------------------------------------------------------------------

DATASET_HF_COORDS: dict[str, tuple[str, str]] = {
    "callingcards": ("BrentLab/callingcards", "2026_analysis_set"),
    "callingcards_mindel": (
        "BrentLab/callingcards",
        "annotated_feature_reprocess_mindel_analysis",
    ),
    "callingcards_500bp": (
        "BrentLab/callingcards",
        "annotated_feature_reprocess_start_codon_500bp_analysis",
    ),
    "callingcards_intergenic": (
        "BrentLab/callingcards",
        "annotated_feature_reprocess_intergenic_analysis",
    ),
    "harbison": ("BrentLab/harbison_2004", "harbison_2004"),
    "rossi": ("BrentLab/rossi_2021", "rossi_2021_af_combined"),
    "rossi_mindel": ("BrentLab/rossi_2021", "rossi_2021_af_combined_mindel"),
    "rossi_500bp": ("BrentLab/rossi_2021", "rossi_2021_af_combined_start_codon_500bp"),
    "rossi_intergenic": ("BrentLab/rossi_2021", "rossi_2021_af_combined_intergenic"),
    "rossi_peaks": ("BrentLab/rossi_2021", "yep_filtered_peaks_combined"),
    "chec_m2025": (
        "BrentLab/mahendrawada_2025",
        "chec_mahendrawada_m2025_af_combined",
    ),
    "chec_m2025_mindel": (
        "BrentLab/mahendrawada_2025",
        "chec_mahendrawada_m2025_af_combined_mindel",
    ),
    "chec_m2025_500bp": (
        "BrentLab/mahendrawada_2025",
        "chec_mahendrawada_m2025_af_combined_start_codon_500bp",
    ),
    "chec_m2025_intergenic": (
        "BrentLab/mahendrawada_2025",
        "chec_mahendrawada_m2025_af_combined_intergenic",
    ),
    "chec_m2025_peaks": ("BrentLab/mahendrawada_2025", "mahendrawada_chec_seq"),
    "kemmeren": ("BrentLab/kemmeren_2014", "kemmeren_2014"),
    "degron": ("BrentLab/mahendrawada_2025", "rnaseq_reprocessed"),
    "hackett": ("BrentLab/hackett_2020", "hackett_2020_analysis_set"),
    "hu_reimand": ("BrentLab/hu_2007_reimand_2010", "hu_2007_reimand_2010"),
    "hughes_overexpression": ("BrentLab/hughes_2006", "overexpression"),
    "hughes_knockout": ("BrentLab/hughes_2006", "knockout"),
}


def promoter_sets_sql() -> str:
    """
    Return SQL to create and populate the ``promoter_sets`` table.

    :returns: ``CREATE TABLE`` + ``INSERT`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE promoter_sets (
    promoter_set_id  VARCHAR  PRIMARY KEY,
    display_name     VARCHAR  NOT NULL,
    description      VARCHAR
);

INSERT INTO promoter_sets VALUES
    ('kang',
     'Kang',
     '700 bp upstream of each start codon, truncated when a feature lies within 700 bp of the ORF'),
    ('mindel',
     'Mindel',
     'Start codon to >= 700 bp upstream of the TSS (Park 2014 / Pelechano 2013 / Policastro 2020); start codon used when no TSS is defined'),
    ('500bp',
     '500 bp',
     'Exactly 500 bp upstream of the start codon; no truncation or extension'),
    ('intergenic',
     'Intergenic',
     'Full intergenic region upstream of the 5'' end of the feature; 1 410 of 6 040 features are divergently transcribed'),
    ('peaks',
     'Peaks',
     'Regions as called by the original authors'' peak-calling pipeline; not a fixed promoter window');
"""


def binding_methods_sql() -> str:
    """
    Return SQL to create and populate the ``binding_methods`` table.

    :returns: ``CREATE TABLE`` + ``INSERT`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE binding_methods (
    binding_method_id  VARCHAR  PRIMARY KEY,
    display_name       VARCHAR  NOT NULL
);

INSERT INTO binding_methods VALUES
    ('promoter_enrichment', 'Promoter Enrichment'),
    ('peak_calling',        'Peak Calling');
"""


def dataset_registry_sql() -> str:
    """
    Return SQL to create and populate the ``dataset_registry`` table.

    One row per ``db_name`` covering all binding and perturbation datasets.
    Includes HuggingFace coordinates, display metadata, and promoter-set /
    method references for binding datasets.

    Two INSERTs are used: primary rows (``primary_db_name IS NULL``) first,
    variant rows second, so that the self-referential FK constraint is
    satisfied when DuckDB checks it row-by-row.

    :returns: ``CREATE TABLE`` + two ``INSERT`` SQL blocks.
    :rtype: str

    """
    return """
CREATE TABLE dataset_registry (
    db_name              VARCHAR  PRIMARY KEY,
    hf_repo              VARCHAR  NOT NULL,
    hf_config            VARCHAR  NOT NULL,
    data_type            VARCHAR  NOT NULL,
    assay                VARCHAR,
    display_name         VARCHAR,
    base_label           VARCHAR,
    is_primary           BOOLEAN  NOT NULL,
    is_active_default    BOOLEAN  NOT NULL,
    primary_db_name      VARCHAR  REFERENCES dataset_registry(db_name),
    promoter_set_id      VARCHAR  REFERENCES promoter_sets(promoter_set_id),
    binding_method_id    VARCHAR  REFERENCES binding_methods(binding_method_id)
);

-- Pass 1: primary rows (primary_db_name IS NULL) inserted first so the
-- self-referential FK is satisfied when Pass 2 variant rows reference them.
INSERT INTO dataset_registry VALUES
-- binding primaries
('callingcards',
 'BrentLab/callingcards', '2026_analysis_set',
 'binding', 'CallingCards',
 '2026 Calling Cards', '2026 Calling Cards',
 TRUE, TRUE, NULL, 'kang', 'promoter_enrichment'),
('harbison',
 'BrentLab/harbison_2004', 'harbison_2004',
 'binding', 'ChIP-chip',
 '2004 ChIP-chip (Harbison)', '2004 ChIP-chip',
 TRUE, FALSE, NULL, 'kang', 'promoter_enrichment'),
('rossi',
 'BrentLab/rossi_2021', 'rossi_2021_af_combined',
 'binding', 'ChIPexo',
 '2021 ChIP-exo (Rossi)', '2021 ChIP-exo',
 TRUE, TRUE, NULL, 'kang', 'promoter_enrichment'),
('chec_m2025',
 'BrentLab/mahendrawada_2025', 'chec_mahendrawada_m2025_af_combined',
 'binding', 'ChEC-seq',
 '2025 ChEC-seq (Mahendrawada)', '2025 ChEC-seq',
 TRUE, TRUE, NULL, 'kang', 'promoter_enrichment'),
-- perturbation primaries (all have primary_db_name = NULL)
('kemmeren',
 'BrentLab/kemmeren_2014', 'kemmeren_2014',
 'perturbation', 'TFKO',
 '2014 TFKO (Kemmeren)', '2014 TFKO',
 TRUE, TRUE, NULL, NULL, NULL),
('degron',
 'BrentLab/mahendrawada_2025', 'rnaseq_reprocessed',
 'perturbation', 'RNA-seq',
 '2025 Degron (Mahendrawada)', '2025 Degron',
 TRUE, TRUE, NULL, NULL, NULL),
('hackett',
 'BrentLab/hackett_2020', 'hackett_2020_analysis_set',
 'perturbation', 'overexpression',
 '2020 Overexpression (Hackett)', '2020 Overexpression',
 TRUE, TRUE, NULL, NULL, NULL),
('hu_reimand',
 'BrentLab/hu_2007_reimand_2010', 'hu_2007_reimand_2010',
 'perturbation', 'TFKO',
 '2007 TFKO (Hu)', '2007 TFKO',
 TRUE, FALSE, NULL, NULL, NULL),
('hughes_overexpression',
 'BrentLab/hughes_2006', 'overexpression',
 'perturbation', 'overexpression',
 '2006 Overexpression (Hughes)', '2006 Overexpression',
 TRUE, FALSE, NULL, NULL, NULL),
('hughes_knockout',
 'BrentLab/hughes_2006', 'knockout',
 'perturbation', 'TFKO',
 '2006 Knockout (Hughes)', '2006 Knockout',
 TRUE, FALSE, NULL, NULL, NULL);

-- Pass 2: variant rows — primary rows above now exist so FK is satisfied
INSERT INTO dataset_registry VALUES
-- callingcards variants
('callingcards_mindel',
 'BrentLab/callingcards', 'annotated_feature_reprocess_mindel_analysis',
 'binding', 'CallingCards',
 '2026 Calling Cards (Mindel)', '2026 Calling Cards',
 FALSE, FALSE, 'callingcards', 'mindel', 'promoter_enrichment'),
('callingcards_500bp',
 'BrentLab/callingcards', 'annotated_feature_reprocess_start_codon_500bp_analysis',
 'binding', 'CallingCards',
 '2026 Calling Cards (500bp)', '2026 Calling Cards',
 FALSE, FALSE, 'callingcards', '500bp', 'promoter_enrichment'),
('callingcards_intergenic',
 'BrentLab/callingcards', 'annotated_feature_reprocess_intergenic_analysis',
 'binding', 'CallingCards',
 '2026 Calling Cards (Intergenic)', '2026 Calling Cards',
 FALSE, FALSE, 'callingcards', 'intergenic', 'promoter_enrichment'),
-- rossi variants
('rossi_mindel',
 'BrentLab/rossi_2021', 'rossi_2021_af_combined_mindel',
 'binding', 'ChIPexo',
 '2021 ChIP-exo (Rossi, Mindel)', '2021 ChIP-exo',
 FALSE, FALSE, 'rossi', 'mindel', 'promoter_enrichment'),
('rossi_500bp',
 'BrentLab/rossi_2021', 'rossi_2021_af_combined_start_codon_500bp',
 'binding', 'ChIPexo',
 '2021 ChIP-exo (Rossi, 500bp)', '2021 ChIP-exo',
 FALSE, FALSE, 'rossi', '500bp', 'promoter_enrichment'),
('rossi_intergenic',
 'BrentLab/rossi_2021', 'rossi_2021_af_combined_intergenic',
 'binding', 'ChIPexo',
 '2021 ChIP-exo (Rossi, Intergenic)', '2021 ChIP-exo',
 FALSE, FALSE, 'rossi', 'intergenic', 'promoter_enrichment'),
('rossi_peaks',
 'BrentLab/rossi_2021', 'yep_filtered_peaks_combined',
 'binding', 'ChIPexo',
 '2021 ChIP-exo Peaks', '2021 ChIP-exo',
 FALSE, FALSE, 'rossi', 'peaks', 'peak_calling'),
-- chec_m2025 variants
('chec_m2025_mindel',
 'BrentLab/mahendrawada_2025', 'chec_mahendrawada_m2025_af_combined_mindel',
 'binding', 'ChEC-seq',
 '2025 ChEC-seq (Mahendrawada, Mindel)', '2025 ChEC-seq',
 FALSE, FALSE, 'chec_m2025', 'mindel', 'promoter_enrichment'),
('chec_m2025_500bp',
 'BrentLab/mahendrawada_2025', 'chec_mahendrawada_m2025_af_combined_start_codon_500bp',
 'binding', 'ChEC-seq',
 '2025 ChEC-seq (Mahendrawada, 500bp)', '2025 ChEC-seq',
 FALSE, FALSE, 'chec_m2025', '500bp', 'promoter_enrichment'),
('chec_m2025_intergenic',
 'BrentLab/mahendrawada_2025', 'chec_mahendrawada_m2025_af_combined_intergenic',
 'binding', 'ChEC-seq',
 '2025 ChEC-seq (Mahendrawada, Intergenic)', '2025 ChEC-seq',
 FALSE, FALSE, 'chec_m2025', 'intergenic', 'promoter_enrichment'),
('chec_m2025_peaks',
 'BrentLab/mahendrawada_2025', 'mahendrawada_chec_seq',
 'binding', 'ChEC-seq',
 '2025 ChEC-seq Peaks (Mahendrawada)', '2025 ChEC-seq',
 FALSE, FALSE, 'chec_m2025', 'peaks', 'peak_calling');
"""


def comparative_registry_sql() -> str:
    """
    Return SQL to create and populate the ``comparative_dataset_registry`` table.

    :returns: ``CREATE TABLE`` + ``INSERT`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE comparative_dataset_registry (
    analysis_name    VARCHAR  PRIMARY KEY,
    provenance       VARCHAR  NOT NULL,
    description      VARCHAR,
    hf_repo          VARCHAR,
    hf_config        VARCHAR
);

INSERT INTO comparative_dataset_registry VALUES
    ('dto',
     'hf_parquet',
     'Directional transcription overlap (DTO) empirical p-values',
     'BrentLab/yeast_comparative_analysis',
     'dto'),
    ('topn_results',
     'computed',
     'Top-N-by-binding responsive ratio for (binding, perturbation, regulator) triples',
     NULL,
     NULL),
    ('correlations',
     'computed',
     'Pairwise Pearson or Spearman correlations between samples within the same data type',
     NULL,
     NULL);
"""


def column_metadata_sql(vdb: VirtualDB) -> str:
    """
    Return SQL to create and populate the ``dataset_column_metadata`` table.

    Queries VirtualDB for each dataset's column metadata, applies
    :data:`~tfbpshiny.utils.vdb_init.HIDDEN_FILTER_FIELDS`, and classifies
    columns as ``'condition'`` or ``'upstream'``.  The resulting table replaces
    ``vdb.get_column_metadata()`` in the Phase 2 app startup.

    :param vdb: VirtualDB instance with all dataset views registered.
    :returns: ``CREATE TABLE`` + ``INSERT`` SQL string.
    :rtype: str

    """
    hidden_global = HIDDEN_FILTER_FIELDS.get("*", set())
    rows: list[str] = []

    for db_name in vdb.get_datasets():
        db_meta = vdb.get_column_metadata(db_name) or {}
        hidden = hidden_global | HIDDEN_FILTER_FIELDS.get(db_name, set())

        condition_cols = [
            col
            for col, m in db_meta.items()
            if m.role == "experimental_condition"
            and m.level_definitions is not None
            and col not in hidden
        ]
        upstream_cols = [
            col
            for col, m in db_meta.items()
            if col not in condition_cols
            and col not in hidden
            and col != "sample_id"
            and m.role not in ("regulator_identifier", "target_identifier")
            and m.level_definitions is None
        ]
        for col in condition_cols:
            safe_col = col.replace("'", "''")
            safe_db = db_name.replace("'", "''")
            rows.append(f"('{safe_db}', '{safe_col}', 'condition')")
        for col in upstream_cols:
            safe_col = col.replace("'", "''")
            safe_db = db_name.replace("'", "''")
            rows.append(f"('{safe_db}', '{safe_col}', 'upstream')")

    values_clause = ",\n    ".join(rows) if rows else "('__placeholder__', '__none__', 'condition')"
    return f"""
CREATE TABLE dataset_column_metadata (
    db_name     VARCHAR NOT NULL,
    column_name VARCHAR NOT NULL,
    role        VARCHAR NOT NULL,
    PRIMARY KEY (db_name, column_name)
);

INSERT INTO dataset_column_metadata VALUES
    {values_clause};
"""
