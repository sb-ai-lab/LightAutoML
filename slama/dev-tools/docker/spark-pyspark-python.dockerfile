FROM python:3.9

# tag example: spark-pyspark-python:3.9-3.2.0

ARG SCALA_VERSION=2.12.10
ARG SPARK_VERSION=3.2.0
ARG HADOOP_VERSION=3.2
ARG SPARK_HOME=/spark

RUN apt-get update && \
	apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN	wget http://scala-lang.org/files/archive/scala-${SCALA_VERSION}.deb && \
	dpkg -i scala-${SCALA_VERSION}.deb

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
	tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
	mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark && \
	rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

RUN pip install pyspark==${SPARK_VERSION} pyarrow

COPY jars_cache /root/.ivy2/cache
