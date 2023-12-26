## Docs for selection experiments

Files involved:
* `launch_selection_on_source.py` - launches experiments on datasets from particular source. Source is the tag of a ClearML dataset. Firstly, gets ids of all the datasets matching source. Then for each dataset creates  ClearML Task and enques it. To run specify the `source` varibale in script. 
* `experiments/run_selection_methods.py` - performs one experiment on a specified dataset. Iterates through all selection methods and computes scores for each. 