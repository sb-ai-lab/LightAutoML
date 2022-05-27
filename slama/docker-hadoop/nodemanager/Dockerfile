FROM bde2020/hadoop-base:2.0.0-hadoop3.2.1-java8

MAINTAINER Ivan Ermilov <ivan.s.ermilov@gmail.com>

HEALTHCHECK CMD curl -f http://localhost:8042/ || exit 1

ADD run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8042

RUN apt update

RUN apt install build-essential libncurses5-dev zlib1g-dev libnss3-dev libgdbm-dev libssl-dev libsqlite3-dev libffi-dev libreadline-dev curl libbz2-dev -y

RUN apt install wget

RUN wget https://www.python.org/ftp/python/3.9.12/Python-3.9.12.tgz

RUN tar -xvf Python-3.9.12.tgz

RUN cd Python-3.9.12 && ./configure --enable-optimizations

RUN cd Python-3.9.12 && make altinstall

RUN ln -s /usr/local/bin/python3.9 /usr/bin/python3

ADD LightAutoML-0.3.0-py3-none-any.whl /LightAutoML-0.3.0-py3-none-any.whl

RUN python3 -m pip install /LightAutoML-0.3.0-py3-none-any.whl

CMD ["/run.sh"]
