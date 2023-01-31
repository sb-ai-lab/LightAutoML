print("Create feature selector")
model01 = BoostLGBM(
    default_params={
        "learning_rate": 0.05,
        "num_leaves": 64,
        "seed": 42,
        "num_threads": 5,
    }
)
model02 = BoostLGBM(
    default_params={
        "learning_rate": 0.05,
        "num_leaves": 64,
        "seed": 42,
        "num_threads": 5,
    }
)
pipe0 = LGBSimpleFeatures()
pie = NpPermutationImportanceEstimator()
pie1 = ModelBasedImportanceEstimator()
sel1 = ImportanceCutoffSelector(pipe0, model01, pie1, cutoff=0)
sel2 = NpIterativeFeatureSelector(pipe0, model02, pie, feature_group_size=1, max_features_cnt_in_result=15)
selector = ComposedSelector([sel1, sel2])
