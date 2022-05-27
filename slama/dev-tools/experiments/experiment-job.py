import datetime
import glob
import itertools
import json
import logging
import os
import random
import subprocess
import time
import uuid
from copy import copy
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Iterator, Optional

import yaml
from tqdm import tqdm

from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT


DEFAULT_SPARK_CONFIG_SETTINGS = "dev-tools/config/experiments/default-spark-conf.yaml"
EXP_PY_FILES_DIR = "dev-tools/experiments/"
MARKER = "EXP-RESULT:"

statefile_path = "/tmp/exp-job"
results_path = "/tmp/exp-job"

# cfg_path = "./dev-tools/config/experiments/experiment-config-spark-cluster-automl-lgb-load-pipeline.yaml"
cfg_path = "./dev-tools/config/experiments/experiment-config-spark-cluster-automl-lgb.yaml"
all_results_path = "/tmp/exp-job/results.txt"


ExpInstanceConfig = Dict[str, Any]


logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class ExpInstanceProc:
    exp_instance: ExpInstanceConfig
    p: subprocess.Popen
    outfile: str
    id: str = field(init=False)

    def __post_init__(self):
        self.id = f'{uuid.uuid4()}'

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other):
        return other.id == self.id


def read_config(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    return config_data


def process_state_file(config_data: Dict[str, Any]) -> Set[str]:
    state_file_mode = config_data["state_file"]
    state_file_path = f"{statefile_path}/state_file.json"

    if state_file_mode == "use" and os.path.exists(state_file_path):
        with open(state_file_path, "r") as f:
            exp_instances = [json.loads(line) for line in f.readlines()]
        exp_instances_ids = {exp_inst["instance_id"] for exp_inst in exp_instances}
    elif state_file_mode == "use" and not os.path.exists(state_file_path):
        exp_instances_ids = set()
    elif state_file_mode == "delete":
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
        exp_instances_ids = set()
    else:
        raise ValueError(f"Unsupported mode for state file: {state_file_mode}")

    logger.info(f"Found {len(exp_instances_ids)} existing experiments "
                f"in state file {state_file_path}. "
                f"Exp instance ids: \n{exp_instances_ids}")

    return exp_instances_ids


def generate_experiments(config_data: Dict) -> List[ExpInstanceConfig]:
    logger.info(f"Starting to generate experiments configs. Config file: \n\n{config_data}")
    experiments = config_data["experiments"]

    existing_exp_instances_ids = process_state_file(config_data)

    with open(DEFAULT_SPARK_CONFIG_SETTINGS, "r") as f:
        default_spark_config = yaml.safe_load(f)['default_spark_config']

    exp_instances = []
    for experiment in experiments:
        name = experiment["name"]
        repeat_rate = experiment.get("repeat_rate", 1)
        libraries = experiment["library"]

        # Make all possible experiment AutoML params
        keys_exps, values_exps = zip(*experiment["params"].items())
        param_sets = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

        if "spark" in libraries:
            assert "spark_config" in experiment, f"No spark_config set (even empty one) for experiment {name}"
            keys_exps, values_exps = zip(*experiment['spark_config'].items())
            spark_param_sets = [dict(zip(keys_exps, v)) for v in itertools.product(*values_exps)]

            spark_configs = []
            for spark_params in spark_param_sets:
                spark_config = copy(default_spark_config)
                spark_config.update(spark_params)
                spark_config['spark.cores.max'] = \
                    int(spark_config['spark.executor.cores']) * int(spark_config['spark.executor.instances'])
                spark_configs.append(spark_config)
            spark_instances = itertools.product(["spark"], param_sets, spark_configs)
        else:
            spark_instances = []

        if "lama" in libraries:
            lama_instances = itertools.product(["lama"], param_sets, [None])
        else:
            lama_instances = []

        instances = (
            (i, el) for el in itertools.chain(spark_instances, lama_instances)
            for i in range(repeat_rate)
        )

        for repeat_seq_id, (library, params, spark_config) in instances:
            params = copy(params)
            spark_config = copy(spark_config)

            instance_id = f"{name}-{uuid.uuid4()}"[:50]

            if instance_id in existing_exp_instances_ids:
                continue

            if library == "spark":
                params["spark_config"] = spark_config

            exp_instance = {
                "exp_name": name,
                "instance_id": instance_id,
                "params": params,
                "calculation_script": config_data["calculation_scripts"][library]
            }

            exp_instances.append(exp_instance)

    return exp_instances


# Run subprocesses with Spark jobs
def run_experiments(experiments_configs: List[ExpInstanceConfig]) \
        -> Iterator[ExpInstanceProc]:
    logger.info(f"Starting to run experiments. Experiments count: {len(experiments_configs)}")
    for exp_instance in experiments_configs:
        exp_name = exp_instance["exp_name"]
        instance_id = exp_instance["instance_id"]
        launch_script_name = exp_instance["calculation_script"]
        jobname = instance_id[:50].strip('-')

        instance_config_path = f"/tmp/{jobname}/config.yaml"
        os.makedirs(os.path.dirname(instance_config_path), exist_ok=True)

        spark_master = exp_instance['params']['spark_config']['spark.master']

        py_files = glob.glob(os.path.join(EXP_PY_FILES_DIR, '*.py'))
        py_files = ','.join(py_files)

        with open(instance_config_path, "w+") as f:
            yaml.dump(exp_instance["params"], f, default_flow_style=False)

        outfile = os.path.abspath(f"{results_path}/Results_{instance_id}.log")

        confs = [f'{setting}={value}' for setting, value in exp_instance['params']['spark_config'].items()]
        conf_args = [el for c in confs for el in ['--conf', c]]
        conf_args.extend([
            '--conf', f'spark.kubernetes.driver.label.appname={instance_id}',
            '--conf', f'spark.kubernetes.executor.label.appname={instance_id}',
            '--conf', f'spark.kubernetes.driver.label.expname={exp_name}',
            '--conf', f'spark.kubernetes.executor.label.expname={exp_name}',
            '--py-files', py_files,
            '--files', instance_config_path
        ])

        custom_env = os.environ.copy()
        custom_env["PYSPARK_PYTHON"] = "python3"
        p = subprocess.Popen(
            ["spark-submit", '--master', spark_master, '--deploy-mode', 'cluster',  *conf_args, launch_script_name],
            env=custom_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info(f"Started process with instance id {instance_id} and args {p.args}. ")

        yield ExpInstanceProc(exp_instance=exp_instance, p=p, outfile=outfile)


def limit_procs(it: Iterator[ExpInstanceProc],
                max_parallel_ops: int = 1,
                check_period_secs: float = 1.0) \
        -> Iterator[ExpInstanceProc]:
    assert max_parallel_ops > 0

    exp_procs: Set[ExpInstanceProc] = set()

    def try_to_remove_finished() -> Optional[ExpInstanceProc]:
        proc_to_remove = None
        for exp_p in exp_procs:
            if exp_p.p.poll() is not None:
                proc_to_remove = exp_p
                break

        if proc_to_remove is not None:
            msg = f"Removing proc {proc_to_remove.p.pid} " \
                  f"because it is ended (exit code {proc_to_remove.p.returncode}) " \
                  f"for instance id {proc_to_remove.exp_instance['instance_id']}"
            if proc_to_remove.p.returncode != 0:
                logger.warning(msg)
            else:
                logger.debug(msg)

            exp_procs.remove(proc_to_remove)

        return proc_to_remove

    for el in it:
        exp_procs.add(el)

        while len(exp_procs) >= max_parallel_ops:
            exp_proc = try_to_remove_finished()

            if exp_proc:
                yield exp_proc
            else:
                time.sleep(check_period_secs)

    while len(exp_procs) > 0:
        exp_proc = try_to_remove_finished()

        if exp_proc:
            yield exp_proc
        else:
            time.sleep(check_period_secs)


def process_outfile(exp_instance: ExpInstanceConfig, outfile: str, result_file: str) -> None:
    if not os.path.exists(outfile):
        logger.error(f"No result file found on path {outfile} for exp with instance id {exp_instance['instance_id']}")
        return

    with open(outfile, "r") as f:
        res_lines = [line for line in f.readlines() if MARKER in line]

    if len(res_lines) == 0:
        logger.error(f"No result line found for exp with instance id {exp_instance['instance_id']} in {outfile}")
        return

    if len(res_lines) > 1:
        logger.warning(f"More than one results ({len(res_lines)}) are found "
                       f"for exp with instance id {exp_instance['instance_id']} in {outfile}")

    result_line = res_lines[-1]
    result_str = result_line[result_line.index(MARKER) + len(MARKER):].strip('\n \t')
    if len(result_str) == 0:
        logger.error(f"Found result line for exp with instance id {exp_instance['instance_id']} in {outfile} is empty")

    curr_time = datetime.datetime.now()

    record = {
        "time": curr_time.strftime('%Y-%m-%d %H:%M:%S'),
        "timestamp": curr_time.timestamp(),
        "exp_instance": exp_instance,
        "outfile": outfile,
        "result": result_str
    }

    with open(result_file, "a") as f:
        record = json.dumps(record)
        f.write(f"{record}{os.linesep}")


def register_results(exp_procs: Iterator[ExpInstanceProc], total: int):
    os.makedirs(statefile_path, exist_ok=True)

    state_file_path = f"{statefile_path}/state_file.json"

    exp_proc: ExpInstanceProc
    for exp_proc in tqdm(exp_procs, desc="Experiment", total=total):
        # Mark exp_instance as completed in the state file
        instance_id = exp_proc.exp_instance['instance_id']
        logger.info(f"Registering finished process with instance id: {instance_id}")

        # kubectl -n spark-lama-exps logs -l spark-role=driver,spark-app-selector=spark-ba10619fe245432d9153bc0fcd96abae
        kube_ns = exp_proc.exp_instance['params']['spark_config']['spark.kubernetes.namespace']
        with open(exp_proc.outfile, 'w') as f:
            logs_fetcher = subprocess.Popen(
                ['kubectl', '-n', kube_ns, 'logs', '-l', f'spark-role=driver,appname={instance_id}'],
                stdout=f,
                stderr=f
            )
            logs_fetcher.wait()

        process_outfile(exp_proc.exp_instance, exp_proc.outfile, all_results_path)

        with open(state_file_path, "a") as f:
            record = copy(exp_proc.exp_instance)
            record["outfile"] = exp_proc.outfile
            record = json.dumps(record)
            f.write(f"{record}{os.linesep}")

    logger.info("Finished exp_procs proccessing")


def print_all_results_file():
    print("All results file content:\n\n")
    if not os.path.exists(all_results_path):
        logger.error("No file with results. Nothing to print.")
        return

    with open(all_results_path, "r") as f:
        content = f.read()

    print(content)


def main():
    logger.info("Starting experiments")

    cfg = read_config(cfg_path)
    exp_cfgs = generate_experiments(cfg)
    exp_procs = limit_procs(run_experiments(exp_cfgs), max_parallel_ops=1)
    register_results(exp_procs, total=len(exp_cfgs))
    print_all_results_file()

    logger.info("Finished processes")


if __name__ == "__main__":
    main()
