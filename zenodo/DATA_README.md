# Reproducibility data for “Structure alone supports efficient visual computation in the Drosophila visual system”

This is version 1.0.0 of the data and source-data record accompanying the manuscript by Eudald Correig-Fraga, Roger Guimerà, and Marta Sales-Pardo. The version-specific DOI is https://doi.org/10.5281/zenodo.21549559.

## Scope

The record contains the exact processed biological graph and four randomized graph instances analyzed in the manuscript, the matching neuron annotations and retinal-neuron coordinate mapping, trial-level predictions for the numerical-discrimination analysis, and the numerical values used to create the panels in Fig. 3. The exact graph files are included because they are the authoritative randomized instances used in the reported analyses.

The raw FlyWire synapse release is not duplicated. The biological graph was derived from `proofread_connections_783.feather` in the FlyWire Whole-brain Connectome Connectivity Data v783 record (https://doi.org/10.5281/zenodo.10676866). The annotation table is an exact copy of `Supplemental_file1_neuron_annotations.tsv` from FlyWire annotations release v2.1.0 (https://github.com/flyconnectome/flywire_annotations/releases/tag/v2.1.0).

## Files

- `connections_biological.csv.gz`: biological FlyWire v783 graph, grouped by presynaptic and postsynaptic neuron.
- `connections_unconstrained.csv.gz`: exact unconstrained randomized graph used in the manuscript.
- `connections_connection_pruned.csv.gz`: exact connection-pruned randomized graph used in the manuscript.
- `connections_binned.csv.gz`: exact synapse-bin randomized graph used in the manuscript.
- `connections_neuron_binned.csv.gz`: exact neuron-bin randomized graph used in the manuscript.
- `neuron_annotations_flywire_v2.1.0.tsv.gz`: FlyWire v783 neuron annotations used for cell types and spatial coordinates.
- `right_visual_positions_all_neurons.csv`: retinal-neuron mapping and projected eye coordinates used by the main configuration.
- `predictions_numerical_*.csv`: trial-level numerical-discrimination predictions for each graph ensemble. Machine-specific prefixes in the original image paths have been normalized to portable relative paths; the image filenames and all predictions are unchanged.
- `source_data_figure_3_task_accuracy.csv`: values and uncertainty measures used for Fig. 3a, b, and e.
- `source_data_figure_3d_transfer.csv`: values used for the train/test transfer panel.
- `source_data_figure_3f_weber.csv`: trial counts and accuracy summaries by graph ensemble and Weber ratio.
- `software_versions.csv`: exact repository commits associated with this record.
- `DATA_DICTIONARY.md`: field-level documentation.
- `MANIFEST.csv`: filenames, sizes, media types, row counts where applicable, and SHA-256 checksums.
- `SHA256SUMS`: checksums in standard `sha256sum` format.

## Code versions

- `connectome`: commit `93476a27692cc5a2e8e4c0ea0f1ec398ab5ae50d`, package version 0.2.0, https://github.com/eudald-seeslab/connectome
- `train-your-fly`: commit `733a8bdb80cb68089d63f8af803a6180c0c67405`, package version 0.1.0, https://github.com/eudald-seeslab/train-your-fly
- `cogstim`: commit `e61ebf1d59d929af0fee2bc5e670bbbaf5e89f1f`, package version 0.8.1, https://github.com/eudald-seeslab/cogstim

## Loading compressed tables

Pandas reads the compressed graph and annotation tables directly:

```python
import pandas as pd

connections = pd.read_csv("connections_biological.csv.gz")
annotations = pd.read_csv("neuron_annotations_flywire_v2.1.0.tsv.gz", sep="\t")
```

## Integrity

Run `sha256sum --check SHA256SUMS` after downloading all files. `MANIFEST.csv` contains the same checksums together with file sizes and table metadata.

## Licence and attribution

The record is released under CC BY 4.0. The processed connectivity and annotations derive from the credited FlyWire and Schlegel et al. resources, which are also distributed under CC BY 4.0 in their cited archival records/releases. Please cite both this record and the upstream resources when reusing those data.
