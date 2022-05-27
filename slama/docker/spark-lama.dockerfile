FROM python:3.9.9

RUN pip install poetry
WORKDIR /code
#COPY poetry.lock pyproject.toml /code/
COPY pyproject.toml /code/

RUN poetry config virtualenvs.create false --local
RUN poetry install

RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
RUN tar -xvf openjdk-11+28_linux-x64_bin.tar.gz
RUN mv jdk-11 /usr/local/lib/jdk-11
RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java

RUN pip install pyarrow

COPY .. /code
RUN poetry build
RUN pip install dist/LightAutoML-0.3.0-py3-none-any.whl
#COPY ivy2_cache /root/.ivy2/cache
