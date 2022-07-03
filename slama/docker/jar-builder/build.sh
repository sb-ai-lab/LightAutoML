#!/bin/bash
# both /src and /jars assumed to be mounted

set -ex

# JAR file will be located at: /src/target/scala-2.12/spark-lightautoml_2.12-{VERSION}.jar
sbt clean && sbt package

cp target/scala-2.12/*.jar /jars/
chmod 777 /jars/*.jar

sbt clean
