FROM spark-pyspark-python:3.9-3.2.0

ARG spark_jars_cache=jars_cache

WORKDIR /src

RUN pip install poetry
RUN poetry config virtualenvs.create false --local

# we need star here to make copying of poetry.lock conditional
COPY requirements.txt /src

# workaround to make poetry not so painly slow on dependency resolution
# before this image building: poetry export -f requirements.txt > requirements.txt
RUN pip install -r requirements.txt

RUN pip install torchvision==0.9.1

COPY dist/LightAutoML-0.3.0-py3-none-any.whl /tmp/LightAutoML-0.3.0-py3-none-any.whl
RUN pip install /tmp/LightAutoML-0.3.0-py3-none-any.whl

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

COPY jars jars


