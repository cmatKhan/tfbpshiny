# Database Schema

## Overview

Rather than perform analysis type functions at the target level in the 
app, these types of analyses will now be carried out offline. The result
will be saved into a .duckdb file based database, and that is what will 
be used to provide data to the app.  

In addition to the comparison type analyses, eg topn, dto or correlations,
the file based database will also store sample level metadata (including 
conditions, etc) and provide some 

**Layers:**

| Layer | Tables | Description |
|-------|--------|-------------|
| Coordinating | `promoter_sets`, `binding_methods`, `dataset_registry`, `comparative_dataset_registry` | Registry + display metadata; single source of truth for the dicts currently hardcoded in `vdb_init.py` and `comparison/queries.py` |
| Metadata | `{db_name}_meta` (one per dataset) | Materialized verbatim from VirtualDB `_meta` views; schema is dataset-specific |
| Comparison — HF-sourced | `{analysis_name}` (one per configured comparative dataset) | Materialized verbatim from the raw HuggingFace Parquet; composite `source_sample` IDs preserved |
| Comparison — computed | `topn_results`, `correlations` | Pairwise analysis results computed at materialization time; same `source_sample` format |

**What is intentionally excluded:**

- Target-level measurement data (`{db_name}` views — `target_locus_tag`,
  `enrichment`, `log2FoldChange`, etc.). These remain in the HuggingFace
  Parquet files and are accessed at runtime only when needed.
- Field definition / alias / mapping tables from the previous design. That
  mapping is handled inside labretriever / VirtualDB and is not re-encoded here.

---

## Coordinating Layer

### `promoter_sets`

Reference table for genomic region definitions used by binding datasets.
Factored out of the previous `binding_datasets.promoter_set` column and the
hardcoded `PROMOTER_SET_MAP` in `comparison/queries.py`.

The descriptions come from the `genome_resources` block in
`brentlab_yeast_collection.yaml`.

```sql
CREATE TABLE promoter_sets (
    promoter_set_id  VARCHAR  PRIMARY KEY,  -- 'kang' | 'mindel' | '500bp' | 'intergenic' | 'peaks'
    display_name     VARCHAR  NOT NULL,
    description      VARCHAR
);
```

| promoter_set_id | display_name | description |
|---|---|---|
| `kang` | Kang | 700 bp upstream of each start codon, truncated when a feature lies within 700 bp of the ORF |
| `mindel` | Mindel | Start codon to ≥ 700 bp upstream of the TSS (Park 2014 / Pelechano 2013 / Policastro 2020); start codon used when no TSS is defined |
| `500bp` | 500 bp | Exactly 500 bp upstream of the start codon; no truncation or extension |
| `intergenic` | Intergenic | Full intergenic region upstream of the 5′ end of the feature; 1 410 of 6 040 features are divergently transcribed |
| `peaks` | Peaks | Regions as called by the original authors' peak-calling pipeline; not a fixed promoter window |

---

### `binding_methods`

Distinguishes how binding signal was quantified. Factored out of the display
labels in `SCORING_VARIANT_MAP` in `comparison/queries.py`.

```sql
CREATE TABLE binding_methods (
    binding_method_id  VARCHAR  PRIMARY KEY,  -- 'promoter_enrichment' | 'peak_calling'
    display_name       VARCHAR  NOT NULL
);
```

| binding_method_id | display_name |
|---|---|
| `promoter_enrichment` | Promoter Enrichment |
| `peak_calling` | Peak Calling |

`peak_calling` datasets use the original authors' peak annotations (`_peaks`
variants); all others use counts aggregated over a fixed promoter window.

---

### `dataset_registry`

One row per `db_name`. Consolidates the two former registry tables
(`binding_datasets`, `perturbation_datasets`) and the hardcoded Python
dictionaries (`BINDING_LABEL_MAP`, `PERTURBATION_LABEL_MAP`,
`BINDING_BASE_LABEL_MAP`, `PROMOTER_SET_MAP`, `PRIMARY_DATASETS`,
`DEFAULT_ACTIVE_DATASETS`, `PROMOTER_VARIANT_PAIRS`).

```sql
CREATE TABLE dataset_registry (
    db_name              VARCHAR  PRIMARY KEY,
    hf_repo              VARCHAR  NOT NULL,     -- e.g. 'BrentLab/callingcards'
    hf_config            VARCHAR  NOT NULL,     -- e.g. '2026_analysis_set'
    data_type            VARCHAR  NOT NULL,     -- 'binding' | 'perturbation'
    assay                VARCHAR,               -- 'CallingCards' | 'ChIP-chip' | 'ChIPexo' | 'ChEC-seq' | 'TFKO' | 'overexpression'
    display_name         VARCHAR,               -- full label, e.g. '2026 Calling Cards (Mindel)'
    base_label           VARCHAR,               -- label without promoter-set suffix, e.g. '2026 Calling Cards'
    is_primary           BOOLEAN  NOT NULL,     -- TRUE → shown in main dataset selector
    is_active_default    BOOLEAN  NOT NULL,     -- TRUE → toggle on at startup
    -- binding-only (NULL for perturbation datasets)
    primary_db_name      VARCHAR  REFERENCES dataset_registry(db_name),
    promoter_set_id      VARCHAR  REFERENCES promoter_sets(promoter_set_id),
    binding_method_id    VARCHAR  REFERENCES binding_methods(binding_method_id)
);
```

**`primary_db_name`** is NULL for canonical datasets; set for promoter-set or
method variants (e.g. `callingcards_mindel → callingcards`). Together with
`base_label` and `promoter_set_id`, this replaces the `PROMOTER_VARIANT_PAIRS`
and `PEAKS_VARIANT_MAP` dicts.

**Rows:**

| db_name | data_type | display_name | base_label | is_primary | is_active_default | primary_db_name | promoter_set_id | binding_method_id |
|---|---|---|---|---|---|---|---|---|
| `callingcards` | binding | 2026 Calling Cards | 2026 Calling Cards | TRUE | TRUE | NULL | kang | promoter_enrichment |
| `callingcards_mindel` | binding | 2026 Calling Cards (Mindel) | 2026 Calling Cards | FALSE | FALSE | callingcards | mindel | promoter_enrichment |
| `callingcards_500bp` | binding | 2026 Calling Cards (500bp) | 2026 Calling Cards | FALSE | FALSE | callingcards | 500bp | promoter_enrichment |
| `callingcards_intergenic` | binding | 2026 Calling Cards (Intergenic) | 2026 Calling Cards | FALSE | FALSE | callingcards | intergenic | promoter_enrichment |
| `harbison` | binding | 2004 ChIP-chip (Harbison) | 2004 ChIP-chip | TRUE | FALSE | NULL | kang | promoter_enrichment |
| `rossi` | binding | 2021 ChIP-exo (Rossi) | 2021 ChIP-exo | TRUE | TRUE | NULL | kang | promoter_enrichment |
| `rossi_mindel` | binding | 2021 ChIP-exo (Rossi, Mindel) | 2021 ChIP-exo | FALSE | FALSE | rossi | mindel | promoter_enrichment |
| `rossi_500bp` | binding | 2021 ChIP-exo (Rossi, 500bp) | 2021 ChIP-exo | FALSE | FALSE | rossi | 500bp | promoter_enrichment |
| `rossi_intergenic` | binding | 2021 ChIP-exo (Rossi, Intergenic) | 2021 ChIP-exo | FALSE | FALSE | rossi | intergenic | promoter_enrichment |
| `rossi_peaks` | binding | 2021 ChIP-exo Peaks | 2021 ChIP-exo | FALSE | FALSE | rossi | peaks | peak_calling |
| `chec_m2025` | binding | 2025 ChEC-seq (Mahendrawada) | 2025 ChEC-seq | TRUE | TRUE | NULL | kang | promoter_enrichment |
| `chec_m2025_mindel` | binding | 2025 ChEC-seq (Mahendrawada, Mindel) | 2025 ChEC-seq | FALSE | FALSE | chec_m2025 | mindel | promoter_enrichment |
| `chec_m2025_500bp` | binding | 2025 ChEC-seq (Mahendrawada, 500bp) | 2025 ChEC-seq | FALSE | FALSE | chec_m2025 | 500bp | promoter_enrichment |
| `chec_m2025_intergenic` | binding | 2025 ChEC-seq (Mahendrawada, Intergenic) | 2025 ChEC-seq | FALSE | FALSE | chec_m2025 | intergenic | promoter_enrichment |
| `chec_m2025_peaks` | binding | 2025 ChEC-seq Peaks (Mahendrawada) | 2025 ChEC-seq | FALSE | FALSE | chec_m2025 | peaks | peak_calling |
| `kemmeren` | perturbation | 2014 TFKO (Kemmeren) | 2014 TFKO | TRUE | TRUE | NULL | NULL | NULL |
| `degron` | perturbation | 2025 Degron (Mahendrawada) | 2025 Degron | TRUE | TRUE | NULL | NULL | NULL |
| `hackett` | perturbation | 2020 Overexpression (Hackett) | 2020 Overexpression | TRUE | TRUE | NULL | NULL | NULL |
| `hu_reimand` | perturbation | 2007 TFKO (Hu) | 2007 TFKO | TRUE | FALSE | NULL | NULL | NULL |
| `hughes_overexpression` | perturbation | 2006 Overexpression (Hughes) | 2006 Overexpression | TRUE | FALSE | NULL | NULL | NULL |
| `hughes_knockout` | perturbation | 2006 Knockout (Hughes) | 2006 Knockout | TRUE | FALSE | NULL | NULL | NULL |

### `comparative_dataset_registry`

One row per configured comparative analysis. Records provenance so the
materialization pipeline knows whether to copy data from a HuggingFace Parquet
or compute it locally. Both sources use the same `source_sample` composite ID
format, so the same queries work regardless of origin.

```sql
CREATE TABLE comparative_dataset_registry (
    analysis_name    VARCHAR  PRIMARY KEY,  -- table name, e.g. 'dto', 'topn_results', 'correlations'
    provenance       VARCHAR  NOT NULL,     -- 'hf_parquet' | 'computed'
    description      VARCHAR,
    -- hf_parquet only (NULL for computed analyses)
    hf_repo          VARCHAR,              -- e.g. 'BrentLab/yeast_comparative_analysis'
    hf_config        VARCHAR               -- e.g. 'dto'
);
```

| analysis_name | provenance | hf_repo | hf_config |
|---|---|---|---|
| `dto` | `hf_parquet` | `BrentLab/yeast_comparative_analysis` | `dto` |
| `topn_results` | `computed` | NULL | NULL |
| `correlations` | `computed` | NULL | NULL |

**Notes:**

- Additional HuggingFace comparative datasets declared in the VirtualDB config
  appear here automatically at materialization time.
- An analysis that currently lives in HuggingFace (e.g. `dto`) could be
  recalculated locally; changing `provenance` to `computed` signals that the
  stored table was produced locally rather than copied from Parquet.

---

## Metadata Layer

Each registered dataset's VirtualDB `{db_name}_meta` view is materialized
verbatim as a DuckDB table. There is no unified schema across datasets; each
table's columns are determined by the dataset's HuggingFace datacard and the
field mappings in `brentlab_yeast_collection.yaml`.

**Universal columns** (present in every `_meta` table):

| Column | Description |
|---|---|
| `sample_id` | Primary key within the dataset |
| `regulator_locus_tag` | Systematic gene identifier |
| `regulator_symbol` | Gene name; absent for some datasets |

**Dataset-specific columns** (representative; verify against the actual
datacard for the authoritative list):

| Table | Notable columns |
|---|---|
| `callingcards_meta` | `background_total_hops`, `experiment_total_hops`, `carbon_source`, `temperature_celsius` |
| `harbison_meta` | `condition` (YPD, YP-galactose, …) |
| `rossi_meta` | `antibody`, `growth_media`, `treatment`, `carbon_source`, `temperature_celsius` |
| `chec_m2025_meta` | `condition`, `mahendrawada_symbol`, `carbon_source`, `temperature_celsius` |
| `hackett_meta` | `time`, `date`, `mechanism`, `restriction`, `strain` |
| `hu_reimand_meta` | `average_od_of_replicates`, `heat_shock`, `carbon_source`, `temperature_celsius` |
| `hughes_overexpression_meta` | `del_passed_qc`, `sgd_description`, `carbon_source`, `temperature_celsius` |
| `hughes_knockout_meta` | `oe_passed_qc`, `sgd_description`, `carbon_source`, `temperature_celsius` |
| `kemmeren_meta` | `carbon_source`, `temperature_celsius` |
| `degron_meta` | `env_condition`, `timepoint` |

**Notes:**

- Hidden filter fields (from `HIDDEN_FILTER_FIELDS` in `vdb_init.py`) are
  present in these tables but suppressed in the UI at runtime. They are not
  removed at materialization time.
- `FIELD_TYPE_OVERRIDES` in `vdb_init.py` governs how the UI interprets column
  types (e.g. treating `time` as categorical numeric). This remains
  application-level logic, not encoded in the schema.
- Promoter-set variants (`callingcards_mindel`, `rossi_500bp`, etc.) each have
  their own `_meta` table. Their metadata columns are identical to the primary
  dataset's; only the measurement data differs.

---

## Comparison Layer

All comparison tables follow the labretriever `comparative` dataset format (see
[`docs/huggingface_datacard.md`](../../labretriever/docs/huggingface_datacard.md)
in the labretriever reference). Each row is an observation involving two or
more samples, identified by `source_sample` composite strings in the format:

```
"hf_repo;hf_config;sample_id"
```

This is the same format used in the HuggingFace Parquet files stored in
`BrentLab/yeast_comparative_analysis`, so data from either source can be
queried identically.

**Joining any `source_sample` column back to `dataset_registry`:**

```sql
-- resolves the db_name for a source_sample column
(
    SELECT db_name
    FROM dataset_registry
    WHERE hf_repo   = split_part(<source_sample_col>, ';', 1)
      AND hf_config = split_part(<source_sample_col>, ';', 2)
)
```

VirtualDB also exposes `{analysis_name}_expanded` views that parse the
composite IDs into `{link_field}_source` (mapped to `db_name`) and
`{link_field}_id` (sample_id component). The materialized tables store the
raw composite strings; use `split_part` or join to `dataset_registry` at
query time to recover the parsed form.

---

## HuggingFace-Sourced Comparative Tables

One table per entry in `comparative_dataset_registry` with
`provenance = 'hf_parquet'`. Materialized verbatim from the raw HuggingFace
Parquet (not from the VirtualDB `_expanded` view), so the composite
`source_sample` identifiers are preserved exactly as stored upstream.

The schema for each table is dataset-specific. The authoritative definition
lives in the HuggingFace datacard for that repo. The sections below document
the currently configured datasets.

### `dto`

Directional transcription overlap (DTO) empirical p-values, pre-computed by
the Brent Lab and stored in `BrentLab/yeast_comparative_analysis;dto`.

Each row is a (binding sample, perturbation sample) pair for which a DTO score
was computed. The `binding_id` and `perturbation_id` columns are `source_sample`
composite identifiers.

```sql
CREATE TABLE dto (
    binding_id              VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    perturbation_id         VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    dto_empirical_pvalue    DOUBLE,
    dto_fdr                 DOUBLE,
    binding_set_size        INTEGER,
    perturbation_set_size   INTEGER,
    pr_ranking_column       VARCHAR,            -- e.g. 'log2fc'; used to filter analysis subsets
    PRIMARY KEY (binding_id, perturbation_id, pr_ranking_column)
);
```

**Grain:** one row per `(binding_sample, perturbation_sample, pr_ranking_column)`.

**`links` mapping** (from `brentlab_yeast_collection.yaml`):

| composite ID field | maps to |
|---|---|
| `binding_id` | harbison, callingcards, rossi, chec_m2025 |
| `perturbation_id` | kemmeren, hackett, hughes_overexpression, hughes_knockout, degron |

**Notes:**

- The VirtualDB `dto_expanded` view parses `binding_id` into
  `binding_id_source` (the `db_name`) and `binding_id_id` (the sample_id
  component), and likewise for `perturbation_id`. At query time you can use
  either the composite strings (with `split_part`) or join to the
  `dto_expanded` view — both resolve the same data.
- Not all binding × perturbation combinations have a DTO score; only pairs
  in the original analysis are present.
- `pr_ranking_column = 'log2fc'` is the only value currently populated;
  the column exists to support future ranking variants.

---

## Locally Computed Comparison Tables

These tables are produced at `tfbpshiny materialize` time by executing the
queries in `comparison/queries.py` against the VirtualDB views. They use the
same `source_sample` composite ID format as the HuggingFace-sourced tables.

---

### `topn_results`

Top-N-by-binding responsive ratio for one (binding sample, perturbation
sample, regulator) triple. Computed by `topn_responsive_ratio` in
`comparison/queries.py`.

```sql
CREATE TABLE topn_results (
    binding_source_sample       VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    perturbation_source_sample  VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    regulator_locus_tag         VARCHAR  NOT NULL,
    top_n                       INTEGER  NOT NULL,   -- N used for ranking cutoff
    rank_col                    VARCHAR  NOT NULL,   -- column used to rank binding targets
    rank_asc                    BOOLEAN  NOT NULL,   -- TRUE → lowest value = rank 1 (p-values)
                                                     -- FALSE → highest = rank 1 (enrichment)
    effect_threshold            DOUBLE   NOT NULL,   -- |effect| > threshold → responsive
    pvalue_threshold            DOUBLE   NOT NULL,   -- padj/pval < threshold → responsive
    n                           INTEGER  NOT NULL,   -- targets in top-N present in both datasets
    n_responsive                INTEGER  NOT NULL,
    responsive_ratio            DOUBLE   NOT NULL,   -- n_responsive / n
    PRIMARY KEY (
        binding_source_sample,
        perturbation_source_sample,
        regulator_locus_tag,
        top_n, rank_col, rank_asc,
        effect_threshold, pvalue_threshold
    )
);
```

**Grain:** one row per `(binding_sample, perturbation_sample, regulator,
analysis_config)`.

**Notes:**

- Self-interactions (regulator == target) are excluded by the query.
- The callingcards target blacklist (`YOR201C`, `YOR202W`, `YOR203W`,
  `YCL018W`, `YEL021W`) is applied at query time; it is a fixed constant in
  `comparison/queries.py` and is not a column here.
- Multiple rows can exist for the same sample pair and regulator when different
  `(top_n, rank_col, effect_threshold, pvalue_threshold)` combinations are
  pre-computed.
- Defaults: `top_n = 25`, `effect_threshold = 0.0`, `pvalue_threshold = 0.05`.

---

---

### `correlations`

Pairwise Pearson or Spearman correlations between samples within the same
data type.

```sql
CREATE TABLE correlations (
    source_sample_a      VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    source_sample_b      VARCHAR  NOT NULL,  -- 'hf_repo;hf_config;sample_id'
    regulator_locus_tag  VARCHAR  NOT NULL,
    comparison_type      VARCHAR  NOT NULL,  -- 'binding' | 'perturbation'
    method               VARCHAR  NOT NULL,  -- 'pearson' | 'spearman'
    score_col_a          VARCHAR  NOT NULL,  -- column used from sample_a's dataset
    score_col_b          VARCHAR  NOT NULL,  -- column used from sample_b's dataset
    correlation          DOUBLE   NOT NULL,
    n_shared_targets     INTEGER  NOT NULL,  -- targets used; always >= 3
    PRIMARY KEY (
        source_sample_a, source_sample_b,
        regulator_locus_tag,
        method, score_col_a, score_col_b
    )
);
```

**Pair ordering:** `source_sample_a ≤ source_sample_b` (lexicographic) so each
unordered pair is stored exactly once.

**Notes:**

- `comparison_type` lets you filter to binding or perturbation rows without
  parsing sample IDs.
- `score_col_*` allows multiple scoring variants for the same dataset pair to
  coexist (e.g. `poisson_pval` and `callingcards_enrichment` for callingcards).
- Rows are only written when `COUNT(shared targets) >= 3`.

---

## Entity–Relationship Diagram

```
promoter_sets ──< dataset_registry >── binding_methods
                        │ (primary_db_name self-ref for variant rows)

{db_name}_meta tables — one per dataset, schema dataset-specific,
                         joined to dataset_registry via
                         hf_repo + hf_config → db_name at query time

comparative_dataset_registry — one row per analysis (hf_parquet or computed)

HuggingFace-sourced tables (provenance = 'hf_parquet'):
  dto.binding_id               ─┐
  dto.perturbation_id          ─┤→ dataset_registry (via split_part → hf_repo;hf_config)

Locally computed tables (provenance = 'computed'):
  topn_results.binding_source_sample       ─┐
  topn_results.perturbation_source_sample  ─┤→ dataset_registry (via split_part → hf_repo;hf_config)
  correlations.source_sample_a             ─┤
  correlations.source_sample_b             ─┘
```

---

## Materialization Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | required | Path to VirtualDB YAML config |
| `--output` | `brentlab_yeast.duckdb` | Output `.duckdb` file path |
| `--methods` | `pearson,spearman` | Correlation methods to compute |
| `--top-n` | `25` | N for top-N analysis (repeatable) |
| `--effect-threshold` | `0.0` | Effect size threshold for responsiveness (repeatable) |
| `--pvalue-threshold` | `0.05` | Adjusted p-value threshold for responsiveness (repeatable) |
| `--skip-correlations` | false | Skip correlation computation |
| `--skip-topn` | false | Skip top-N computation |
| `--token` | env `HF_TOKEN` | HuggingFace token for private repos |

Repeatable options (`--top-n`, `--effect-threshold`, `--pvalue-threshold`) can
be supplied multiple times to pre-compute several analysis configurations in one
pass.

---

## Example Queries

### Which analyses are available and where they came from

```sql
SELECT analysis_name, provenance, hf_repo, hf_config
FROM comparative_dataset_registry
ORDER BY provenance, analysis_name;
```

### All primary binding datasets with their promoter set and method

```sql
SELECT
    dr.db_name,
    dr.display_name,
    ps.display_name  AS promoter_set,
    bm.display_name  AS binding_method
FROM dataset_registry dr
LEFT JOIN promoter_sets ps ON ps.promoter_set_id = dr.promoter_set_id
LEFT JOIN binding_methods bm ON bm.binding_method_id = dr.binding_method_id
WHERE dr.data_type = 'binding'
  AND dr.is_primary = TRUE
ORDER BY dr.base_label;
```

### All promoter-set variants of a primary binding dataset

```sql
SELECT db_name, display_name, promoter_set_id, binding_method_id
FROM dataset_registry
WHERE primary_db_name = 'callingcards'
   OR db_name         = 'callingcards'
ORDER BY promoter_set_id;
```

### Top-N results for callingcards → kemmeren, joined to display names

```sql
SELECT
    t.regulator_locus_tag,
    split_part(t.binding_source_sample,       ';', 3) AS binding_sample_id,
    split_part(t.perturbation_source_sample,  ';', 3) AS pert_sample_id,
    t.responsive_ratio,
    t.n,
    t.n_responsive
FROM topn_results t
WHERE t.binding_source_sample      LIKE 'BrentLab/callingcards;%'
  AND t.perturbation_source_sample LIKE 'BrentLab/kemmeren_2014;%'
  AND t.top_n              = 25
  AND t.rank_col           = 'poisson_pval'
  AND t.rank_asc           = TRUE
  AND t.effect_threshold   = 0.0
  AND t.pvalue_threshold   = 0.05
ORDER BY t.responsive_ratio DESC;
```

### Correlation matrix for a binding dataset pair, glucose only

```sql
SELECT
    c.source_sample_a,
    c.source_sample_b,
    c.regulator_locus_tag,
    c.correlation,
    c.n_shared_targets
FROM correlations c
WHERE c.source_sample_a   LIKE 'BrentLab/callingcards;%'
  AND c.source_sample_b   LIKE 'BrentLab/harbison_2004;%'
  AND c.comparison_type   = 'binding'
  AND c.method            = 'spearman'
  AND c.score_col_a       = 'callingcards_enrichment'
  AND c.score_col_b       = 'effect'
  -- filter binding sample to glucose via its _meta table
  AND split_part(c.source_sample_a, ';', 3) IN (
      SELECT CAST(sample_id AS VARCHAR)
      FROM callingcards_meta
      WHERE carbon_source = 'glucose'
  );
```

### DTO results for a specific binding × perturbation pair

```sql
-- callingcards (any sample) × kemmeren, log2fc ranking only
SELECT
    split_part(binding_id,       ';', 3) AS binding_sample_id,
    split_part(perturbation_id,  ';', 3) AS pert_sample_id,
    dto_empirical_pvalue,
    dto_fdr,
    binding_set_size,
    perturbation_set_size
FROM dto
WHERE binding_id      LIKE 'BrentLab/callingcards;%'
  AND perturbation_id LIKE 'BrentLab/kemmeren_2014;%'
  AND pr_ranking_column = 'log2fc'
ORDER BY dto_empirical_pvalue;
```

### Dataset metadata for a specific dataset (harbison)

```sql
SELECT *
FROM harbison_meta
LIMIT 10;
```

### All variants grouped by method for method-comparison tab

```sql
SELECT
    dr.base_label,
    dr.db_name,
    dr.display_name,
    bm.display_name  AS method,
    ps.display_name  AS promoter_set
FROM dataset_registry dr
LEFT JOIN binding_methods bm ON bm.binding_method_id = dr.binding_method_id
LEFT JOIN promoter_sets   ps ON ps.promoter_set_id   = dr.promoter_set_id
WHERE dr.data_type = 'binding'
  AND dr.primary_db_name IN ('rossi', 'chec_m2025')
ORDER BY dr.base_label, dr.binding_method_id, dr.promoter_set_id;
```
