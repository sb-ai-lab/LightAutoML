DOCKER_NETWORK = docker-hadoop_default
ENV_FILE = hadoop.env
current_branch := $(shell git rev-parse --abbrev-ref HEAD)
build:
	docker build -t bde2020/hadoop-base:$(current_branch) ./base
	docker build -t bde2020/hadoop-namenode:$(current_branch) ./namenode
	docker build -t bde2020/hadoop-datanode:$(current_branch) ./datanode
	docker build -t bde2020/hadoop-resourcemanager:$(current_branch) ./resourcemanager
	docker build -t bde2020/hadoop-nodemanager:$(current_branch) ./nodemanager
	docker build -t bde2020/hadoop-historyserver:$(current_branch) ./historyserver
	docker build -t bde2020/hadoop-submit:$(current_branch) ./submit

wordcount:
	docker build -t hadoop-wordcount ./submit
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} bde2020/hadoop-base:$(current_branch) hdfs dfs -mkdir -p /input/
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} bde2020/hadoop-base:$(current_branch) hdfs dfs -copyFromLocal -f /opt/hadoop-3.2.1/README.txt /input/
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-wordcount
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} bde2020/hadoop-base:$(current_branch) hdfs dfs -cat /output/*
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} bde2020/hadoop-base:$(current_branch) hdfs dfs -rm -r /output
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} bde2020/hadoop-base:$(current_branch) hdfs dfs -rm -r /input

build-nodemanager-with-python:
	docker build -t bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8-py3.9 ./nodemanager

build-image-to-spark-submit:
	docker build -t spark-submit:1.0 ./spark-submit