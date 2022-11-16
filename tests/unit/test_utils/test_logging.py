#!/usr/bin/env python
# coding: utf-8

import os

import pytest

from lightautoml.automl.presets.tabular_presets import TabularAutoML


@pytest.mark.parametrize(
    "sampled_app_train_test, verbose, log_file",
    [
        (1000, 0, "log_file.log"),
        # (10, 'log_file.log'),
    ],
    indirect=["sampled_app_train_test"],
)
def test_logging(
    capsys,
    tmpdir,
    sampled_app_train_test,
    sampled_app_roles,
    binary_task,
    verbose,
    log_file,
):
    train, _ = sampled_app_train_test

    if log_file:
        log_file = os.path.join(tmpdir, "log_file.log")

    automl = TabularAutoML(
        task=binary_task,
        tuning_params={"max_tuning_iter": 3, "max_tuning_time": 30},
        lgb_params={"default_params": {"num_trees": 5}},
    )

    automl.fit_predict(
        train,
        roles=sampled_app_roles,
        verbose=verbose,
        log_file=log_file,
    )

    sys_out, sys_err = capsys.readouterr()

    if log_file:
        assert os.path.exists(log_file)

    if verbose == 0:
        assert sys_out == ""
        assert sys_err == ""

    # If log_file contains exact same that in stdout at max verbose value
    # if (verbose >= 4) and (log_file is not None):
    #     sys_out_lines = sys_out.split('\n')
    #     with open(log_file) as f:
    #         for line_file, line_stdout in zip(f, sys_out_lines):
    #             # remove message prefixes and compare
    #             assert re.split(r'^(?:[^\t\r\n]+\t){5}([01])(?:\t|$)', line_file) == re.split(r'\s(.*)', line_stdout)


# def test_logging_verbose_switching():
# def test_logging_custom_pipeline():
