# flake8: noqa
# Configuration file for jupyter-notebook.

# timeout of each cell
c.ExecutePreprocessor.timeout = 60 * 15

# Remove metadata
c.ClearMetadataPreprocessor.enabled = True
c.ClearMetadataPreprocessor.clear_cell_metadata = True
