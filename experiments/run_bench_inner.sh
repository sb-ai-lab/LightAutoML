# usage: run_bench_openml.sh <project>

# activate venv with clearml

cd "$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ..

for DATASET in "sampled_app_train"
do
    clearml-task --project $1 --name $2 --script experiments/run_tabular.py \
    --args dataset=$DATASET --queue gpu_queue --requirements experiments/requirements.txt \
    --docker python:3.7.13-bullseye --docker_args "--cpus=8 --memory=16g"
    # TODO set release version as a label
done
