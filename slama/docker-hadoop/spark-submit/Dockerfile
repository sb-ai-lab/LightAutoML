FROM python:3.9.9


RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
RUN tar -xvf openjdk-11+28_linux-x64_bin.tar.gz
RUN mv jdk-11 /usr/local/lib/jdk-11
RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java

RUN pip install pyspark

# COPY bin/yarn-site.xml /etc/hadoop/yarn-site.xml
# COPY bin/core-site.xml /etc/hadoop/core-site.xmb

ENV YARN_CONF_DIR="/etc/hadoop"
ENV HADOOP_CONF_DIR="/etc/hadoop"

WORKDIR /LightAutoML

CMD ["sleep", "infinity"]