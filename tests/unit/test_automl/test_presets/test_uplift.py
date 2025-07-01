from sklearn.metrics import roc_auc_score

# from tests.unit.test_automl.test_presets.presets_utils import check_pickling

import copy
from lightautoml.addons.uplift.base import AutoUplift
from lightautoml.addons.uplift.metrics import (
    calculate_min_max_uplift_auc,
    calculate_uplift_auc,
)


class TestAutoUpliftPreset:
    def test_fit_predict(self, uplift_data_train_test, sampled_app_roles, binary_task):
        # load and prepare data
        train, test, test_target, test_treatment = uplift_data_train_test

        # run automl
        autouplift = AutoUplift(
            binary_task,
            metric="adj_qini",
            has_report=True,
            test_size=0.2,
            timeout=200,
            cpu_limit=1,
            # gpu_ids=["0"]
            # timeout_metalearner=5
        )

        uplift_data_roles = copy.deepcopy(sampled_app_roles)
        uplift_data_roles["treatment"] = "CODE_GENDER"

        autouplift.fit(train, uplift_data_roles, verbose=1)

        best_metalearner = autouplift.create_best_metalearner(
            update_metalearner_params={"timeout": None}, update_baselearner_params={"timeout": 30}
        )
        best_metalearner.fit(train, uplift_data_roles)
        _ = best_metalearner.predict(test)

        uplift_pred, treatment_pred, control_pred = best_metalearner.predict(test)
        uplift_pred = uplift_pred.to_numpy()

        # calculate scores
        roc_auc_treatment = roc_auc_score(test_target[test_treatment == 1], treatment_pred[test_treatment == 1])
        roc_auc_control = roc_auc_score(test_target[test_treatment == 0], control_pred[test_treatment == 0])

        uplift_auc_algo = calculate_uplift_auc(test_target, uplift_pred, test_treatment, normed=False)
        uplift_auc_algo_normed = calculate_uplift_auc(test_target, uplift_pred, test_treatment, normed=True)
        auc_base, auc_perfect = calculate_min_max_uplift_auc(test_target, test_treatment)

        print("--- Check scores ---")
        print('OOF scores "ROC_AUC":')
        print(f"\tTreatment = {roc_auc_treatment:.5f}")
        print(f"\tControl   = {roc_auc_control:.5f}")
        print('Uplift score of test group (default="adj_qini"):')
        print(f"\tBaseline      = {auc_base:.5f}")
        print(f"\tAlgo (Normed) = {uplift_auc_algo:.5f} ({uplift_auc_algo_normed:.5f})")
        print(f"\tPerfect       = {auc_perfect:.5f}")

        # Uplift score of test group (default="adj_qini"):
        #         Baseline      = 0.01340
        #         Algo (Normed) = 0.03012 (0.20648)
        #         Perfect       = 0.09438

        # checks
        assert roc_auc_treatment > 0.68  # 0.69535
        assert roc_auc_control > 0.71  # 0.73022

        # check_pickling(autouplift, ho_score, binary_task, test, target_name)
