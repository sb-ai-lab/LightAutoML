FROM spark-py:pyspark-executor-3.2.0

ARG spark_jars_cache=jars_cache

USER root

RUN mkdir -p /src

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt
RUN pip install torchvision==0.9.1

COPY dist/LightAutoML-0.3.0-py3-none-any.whl /tmp/LightAutoML-0.3.0-py3-none-any.whl
RUN pip install /tmp/LightAutoML-0.3.0-py3-none-any.whl

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

USER ${spark_id}

COPY jars jars