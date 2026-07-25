# Data dictionary

## Connectome graph tables

Applies to `connections_biological.csv.gz`, `connections_unconstrained.csv.gz`, `connections_connection_pruned.csv.gz`, `connections_binned.csv.gz`, and `connections_neuron_binned.csv.gz`.

- `pre_root_id`: FlyWire v783 identifier of the presynaptic neuron.
- `post_root_id`: FlyWire v783 identifier of the postsynaptic neuron.
- `syn_count`: number of synapses represented by the directed edge.

## Neuron annotations

`neuron_annotations_flywire_v2.1.0.tsv.gz` is the v2.1.0 FlyWire annotation table. Its full field documentation is supplied by Schlegel et al. and the FlyWire annotation repository. The fields used directly by this study are:

- `root_id`: FlyWire v783 neuron identifier.
- `pos_x`, `pos_y`, `pos_z`: anchor coordinates in FlyWire voxel coordinates.
- `soma_x`, `soma_y`, `soma_z`: soma coordinates in FlyWire voxel coordinates.
- `cell_type`: hierarchical cell-type annotation used to identify retinal and Kenyon-cell populations.

## Retinal-neuron coordinate mapping

Applies to `right_visual_positions_all_neurons.csv`.

- `root_id`: FlyWire v783 neuron identifier.
- `cell_type`: photoreceptor cell type.
- `x`, `y`, `z`: three-dimensional neuron coordinates.
- `PC1`, `PC2`: first and second principal-component coordinates.
- `x_axis`, `y_axis`: projected coordinates used for eye-plane/Voronoi mapping.

## Trial-level numerical predictions

Applies to `predictions_numerical_*.csv`.

- `Image`: normalized relative path containing the original stimulus filename.
- `Model outputs`: two class scores/probabilities serialized as a comma-separated pair.
- `Prediction`: predicted binary class.
- `True label`: target binary class (`0` for blue and `1` for yellow).
- `Is correct`: `1` when prediction equals the target and `0` otherwise.
- `yellow_count`: number of yellow dots parsed from the stimulus filename.
- `blue_count`: number of blue dots parsed from the stimulus filename.
- `weber_ratio`: larger dot count divided by smaller dot count.
- `equalized`: whether the two colors were equalized for total surface area.

## Figure source-data tables

`source_data_figure_3_task_accuracy.csv` contains task, network, mean accuracy, uncertainty type/value, available trial count, and the immediate source used by the figure code.

`source_data_figure_3d_transfer.csv` contains the training- and test-set accuracy values shown in the retinotopic transfer panel.

`source_data_figure_3f_weber.csv` contains network, Weber ratio, trial count, correct count, mean accuracy, standard deviation, and standard error for surface-equalized trials.
