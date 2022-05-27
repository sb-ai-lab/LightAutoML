#!/usr/bin/env bash

set -ex

BASE_IMAGE_TAG="lama-v3.2.0"

if [[ -z "${KUBE_NAMESPACE}" ]]
then
  KUBE_NAMESPACE=default
fi

if [[ -z "${IMAGE_TAG}" ]]
then
  IMAGE_TAG=${BASE_IMAGE_TAG}
fi


if [[ -z "${REPO}" ]]
then
  echo "REPO var is not defined!"
  REPO=""
  IMAGE=spark-py-lama:${IMAGE_TAG}
  BASE_SPARK_IMAGE=spark-py:${BASE_IMAGE_TAG}
else
  IMAGE=${REPO}/spark-py-lama:${IMAGE_TAG}
  BASE_SPARK_IMAGE=${REPO}/spark-py:${BASE_IMAGE_TAG}
fi


function build_jars() {
  cur_dir=$(pwd)

  echo "Building docker image for lama-jar-builder"
  docker build -t lama-jar-builder -f docker/jar-builder/scala.dockerfile docker/jar-builder

  echo "Building jars"
  docker run -it \
    -v "${cur_dir}/scala-lightautoml-transformers:/scala-lightautoml-transformers" \
    -v "${cur_dir}/jars:/jars" \
    lama-jar-builder
}

function build_pyspark_images() {
  export SPARK_VERSION=3.2.0
  export HADOOP_VERSION=3.2

  mkdir -p /tmp/spark-build-dir
  cd /tmp/spark-build-dir

  wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

  # create images with names:
  # - ${REPO}/spark:${BASE_IMAGE_TAG}
  # - ${REPO}/spark-py:${BASE_IMAGE_TAG}
  # the last is equal to BASE_SPARK_IMAGE

  if [[ ! -z "${REPO}" ]]
  then
    ./spark/bin/docker-image-tool.sh -r ${REPO} -t ${BASE_IMAGE_TAG} \
      -p spark/kubernetes/dockerfiles/spark/bindings/python/Dockerfile \
      build

    ./spark/bin/docker-image-tool.sh -r ${REPO} -t ${BASE_IMAGE_TAG} push
  else
      ./spark/bin/docker-image-tool.sh -t ${BASE_IMAGE_TAG} \
      -p spark/kubernetes/dockerfiles/spark/bindings/python/Dockerfile \
      build
  fi
}

function build_lama_dist() {
  # shellcheck disable=SC2094
  poetry export -f requirements.txt > requirements.txt
  poetry build
}

function build_lama_image() {
  # shellcheck disable=SC2094
  poetry export -f requirements.txt > requirements.txt
  poetry build

  docker build \
    --build-arg base_image=${BASE_SPARK_IMAGE} \
    -t ${IMAGE} \
    -f docker/spark-lama/spark-py-lama.dockerfile \
    .

  if [[ ! -z "${REPO}" ]]
  then
    docker push ${IMAGE}
  fi

  rm -rf dist
}

function build_dist() {
    build_jars
    build_pyspark_images
    build_lama_image
}

function submit_job() {
  APISERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

  script_path=$1

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

  spark-submit \
    --master k8s://${APISERVER} \
    --deploy-mode cluster \
    --py-files "examples/spark/examples_utils.py" \
    --conf 'spark.kryoserializer.buffer.max=512m' \
    --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
    --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
    --conf 'spark.executor.extraClassPath=/root/.ivy2/jars/com.azure_azure-ai-textanalytics-5.1.4.jar:/root/.ivy2/jars/com.azure_azure-core-1.22.0.jar:/root/.ivy2/jars/com.azure_azure-core-http-netty-1.11.2.jar:/root/.ivy2/jars/com.azure_azure-storage-blob-12.14.2.jar:/root/.ivy2/jars/com.azure_azure-storage-common-12.14.1.jar:/root/.ivy2/jars/com.azure_azure-storage-internal-avro-12.1.2.jar:/root/.ivy2/jars/com.beust_jcommander-1.27.jar:/root/.ivy2/jars/com.chuusai_shapeless_2.12-2.3.2.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-annotations-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-core-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-databind-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.dataformat_jackson-dataformat-xml-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.datatype_jackson-datatype-jsr310-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.module_jackson-module-jaxb-annotations-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.woodstox_woodstox-core-6.2.4.jar:/root/.ivy2/jars/com.github.vowpalwabbit_vw-jni-8.9.1.jar:/root/.ivy2/jars/com.jcraft_jsch-0.1.54.jar:/root/.ivy2/jars/com.linkedin.isolation-forest_isolation-forest_3.2.0_2.12-2.0.8.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-cognitive_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-core_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-deep-learning_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-lightgbm_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-opencv_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-vw_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.cntk_cntk-2.4.jar:/root/.ivy2/jars/com.microsoft.cognitiveservices.speech_client-jar-sdk-1.14.0.jar:/root/.ivy2/jars/com.microsoft.ml.lightgbm_lightgbmlib-3.2.110.jar:/root/.ivy2/jars/com.microsoft.onnxruntime_onnxruntime_gpu-1.8.1.jar:/root/.ivy2/jars/commons-codec_commons-codec-1.10.jar:/root/.ivy2/jars/commons-logging_commons-logging-1.2.jar:/root/.ivy2/jars/io.netty_netty-buffer-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-dns-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-http2-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-http-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-socks-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-common-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-handler-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-handler-proxy-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-dns-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-dns-native-macos-4.1.68.Final-osx-x86_64.jar:/root/.ivy2/jars/io.netty_netty-tcnative-boringssl-static-2.0.43.Final.jar:/root/.ivy2/jars/io.netty_netty-transport-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-transport-native-epoll-4.1.68.Final-linux-x86_64.jar:/root/.ivy2/jars/io.netty_netty-transport-native-kqueue-4.1.68.Final-osx-x86_64.jar:/root/.ivy2/jars/io.netty_netty-transport-native-unix-common-4.1.68.Final.jar:/root/.ivy2/jars/io.projectreactor.netty_reactor-netty-core-1.0.11.jar:/root/.ivy2/jars/io.projectreactor.netty_reactor-netty-http-1.0.11.jar:/root/.ivy2/jars/io.projectreactor_reactor-core-3.4.10.jar:/root/.ivy2/jars/io.spray_spray-json_2.12-1.3.2.jar:/root/.ivy2/jars/jakarta.activation_jakarta.activation-api-1.2.1.jar:/root/.ivy2/jars/jakarta.xml.bind_jakarta.xml.bind-api-2.3.2.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpclient-4.5.6.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpcore-4.4.10.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpmime-4.5.6.jar:/root/.ivy2/jars/org.apache.spark_spark-avro_2.12-3.2.0.jar:/root/.ivy2/jars/org.beanshell_bsh-2.0b4.jar:/root/.ivy2/jars/org.codehaus.woodstox_stax2-api-4.2.1.jar:/root/.ivy2/jars/org.openpnp_opencv-3.2.0-1.jar:/root/.ivy2/jars/org.reactivestreams_reactive-streams-1.0.3.jar:/root/.ivy2/jars/org.scalactic_scalactic_2.12-3.0.5.jar:/root/.ivy2/jars/org.scala-lang_scala-reflect-2.12.4.jar:/root/.ivy2/jars/org.slf4j_slf4j-api-1.7.32.jar:/root/.ivy2/jars/org.spark-project.spark_unused-1.0.0.jar:/root/.ivy2/jars/org.testng_testng-6.8.8.jar:/root/.ivy2/jars/org.tukaani_xz-1.8.jar:/root/.ivy2/jars/org.typelevel_macro-compat_2.12-1.1.1.jar:/root/jars/spark-lightautoml_2.12-0.1.jar' \
    --conf 'spark.driver.extraClassPath=/root/.ivy2/jars/com.azure_azure-ai-textanalytics-5.1.4.jar:/root/.ivy2/jars/com.azure_azure-core-1.22.0.jar:/root/.ivy2/jars/com.azure_azure-core-http-netty-1.11.2.jar:/root/.ivy2/jars/com.azure_azure-storage-blob-12.14.2.jar:/root/.ivy2/jars/com.azure_azure-storage-common-12.14.1.jar:/root/.ivy2/jars/com.azure_azure-storage-internal-avro-12.1.2.jar:/root/.ivy2/jars/com.beust_jcommander-1.27.jar:/root/.ivy2/jars/com.chuusai_shapeless_2.12-2.3.2.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-annotations-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-core-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.core_jackson-databind-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.dataformat_jackson-dataformat-xml-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.datatype_jackson-datatype-jsr310-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.jackson.module_jackson-module-jaxb-annotations-2.12.5.jar:/root/.ivy2/jars/com.fasterxml.woodstox_woodstox-core-6.2.4.jar:/root/.ivy2/jars/com.github.vowpalwabbit_vw-jni-8.9.1.jar:/root/.ivy2/jars/com.jcraft_jsch-0.1.54.jar:/root/.ivy2/jars/com.linkedin.isolation-forest_isolation-forest_3.2.0_2.12-2.0.8.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-cognitive_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-core_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-deep-learning_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-lightgbm_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-opencv_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.azure_synapseml-vw_2.12-0.9.5.jar:/root/.ivy2/jars/com.microsoft.cntk_cntk-2.4.jar:/root/.ivy2/jars/com.microsoft.cognitiveservices.speech_client-jar-sdk-1.14.0.jar:/root/.ivy2/jars/com.microsoft.ml.lightgbm_lightgbmlib-3.2.110.jar:/root/.ivy2/jars/com.microsoft.onnxruntime_onnxruntime_gpu-1.8.1.jar:/root/.ivy2/jars/commons-codec_commons-codec-1.10.jar:/root/.ivy2/jars/commons-logging_commons-logging-1.2.jar:/root/.ivy2/jars/io.netty_netty-buffer-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-dns-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-http2-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-http-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-codec-socks-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-common-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-handler-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-handler-proxy-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-dns-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-resolver-dns-native-macos-4.1.68.Final-osx-x86_64.jar:/root/.ivy2/jars/io.netty_netty-tcnative-boringssl-static-2.0.43.Final.jar:/root/.ivy2/jars/io.netty_netty-transport-4.1.68.Final.jar:/root/.ivy2/jars/io.netty_netty-transport-native-epoll-4.1.68.Final-linux-x86_64.jar:/root/.ivy2/jars/io.netty_netty-transport-native-kqueue-4.1.68.Final-osx-x86_64.jar:/root/.ivy2/jars/io.netty_netty-transport-native-unix-common-4.1.68.Final.jar:/root/.ivy2/jars/io.projectreactor.netty_reactor-netty-core-1.0.11.jar:/root/.ivy2/jars/io.projectreactor.netty_reactor-netty-http-1.0.11.jar:/root/.ivy2/jars/io.projectreactor_reactor-core-3.4.10.jar:/root/.ivy2/jars/io.spray_spray-json_2.12-1.3.2.jar:/root/.ivy2/jars/jakarta.activation_jakarta.activation-api-1.2.1.jar:/root/.ivy2/jars/jakarta.xml.bind_jakarta.xml.bind-api-2.3.2.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpclient-4.5.6.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpcore-4.4.10.jar:/root/.ivy2/jars/org.apache.httpcomponents_httpmime-4.5.6.jar:/root/.ivy2/jars/org.apache.spark_spark-avro_2.12-3.2.0.jar:/root/.ivy2/jars/org.beanshell_bsh-2.0b4.jar:/root/.ivy2/jars/org.codehaus.woodstox_stax2-api-4.2.1.jar:/root/.ivy2/jars/org.openpnp_opencv-3.2.0-1.jar:/root/.ivy2/jars/org.reactivestreams_reactive-streams-1.0.3.jar:/root/.ivy2/jars/org.scalactic_scalactic_2.12-3.0.5.jar:/root/.ivy2/jars/org.scala-lang_scala-reflect-2.12.4.jar:/root/.ivy2/jars/org.slf4j_slf4j-api-1.7.32.jar:/root/.ivy2/jars/org.spark-project.spark_unused-1.0.0.jar:/root/.ivy2/jars/org.testng_testng-6.8.8.jar:/root/.ivy2/jars/org.tukaani_xz-1.8.jar:/root/.ivy2/jars/org.typelevel_macro-compat_2.12-1.1.1.jar:/root/jars/spark-lightautoml_2.12-0.1.jar' \
    --conf 'spark.driver.cores=4' \
    --conf 'spark.driver.memory=16g' \
    --conf 'spark.executor.instances=1' \
    --conf 'spark.executor.cores=8' \
    --conf 'spark.executor.memory=16g' \
    --conf 'spark.cores.max=8' \
    --conf 'spark.memory.fraction=0.6' \
    --conf 'spark.memory.storageFraction=0.5' \
    --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
    --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
    --conf "spark.kubernetes.container.image=${IMAGE}" \
    --conf 'spark.kubernetes.namespace='${KUBE_NAMESPACE} \
    --conf 'spark.kubernetes.authenticate.driver.serviceAccountName=spark' \
    --conf 'spark.kubernetes.memoryOverheadFactor=0.4' \
    --conf "spark.kubernetes.driver.label.appname=${filename}" \
    --conf "spark.kubernetes.executor.label.appname=${filename}" \
    --conf 'spark.kubernetes.executor.deleteOnTermination=false' \
    --conf 'spark.kubernetes.container.image.pullPolicy=Always' \
    --conf 'spark.kubernetes.driverEnv.SCRIPT_ENV=cluster' \
    --conf 'spark.kubernetes.file.upload.path=/mnt/nfs/spark_upload_dir' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.claimName=spark-lama-data' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.options.storageClass=local-hdd' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.path=/opt/spark_data/' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.spark-lama-data.mount.readOnly=true' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.claimName=mnt-nfs' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass=nfs' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.path=/mnt/nfs/' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly=false' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.claimName=mnt-nfs' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.options.storageClass=nfs' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.path=/mnt/nfs/' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.mnt-nfs.mount.readOnly=false' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.pvc-tmp.options.claimName=pvc-tmp' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.pvc-tmp.options.storageClass=local-hdd' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.pvc-tmp.mount.path=/tmp/spark_results/' \
    --conf 'spark.kubernetes.driver.volumes.persistentVolumeClaim.pvc-tmp.mount.readOnly=false' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.pvc-tmp.options.claimName=pvc-tmp' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.pvc-tmp.options.storageClass=local-hdd' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.pvc-tmp.mount.path=/tmp/spark_results/' \
    --conf 'spark.kubernetes.executor.volumes.persistentVolumeClaim.pvc-tmp.mount.readOnly=false' \
    ${script_path}
}

function port_forward() {
  script_path=$1
  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')
  spark_app_selector=$(kubectl -n spark-lama-exps get pod -l appname=${filename} -l spark-role=driver -o jsonpath='{.items[0].metadata.labels.spark-app-selector}')

  svc_name=$(kubectl -n ${KUBE_NAMESPACE} get svc -l spark-app-selector=${spark_app_selector} -o jsonpath='{.items[0].metadata.name}')
  kubectl -n spark-lama-exps port-forward svc/${svc_name} 9040:4040
}

function port_forward_by_expname() {
  expname=$1
  port=$2
  spark_app_selector=$(kubectl -n spark-lama-exps get pod -l spark-role=driver,expname=${expname} -o jsonpath='{.items[0].metadata.labels.spark-app-selector}')
  svc_name=$(kubectl -n ${KUBE_NAMESPACE} get svc -l spark-app-selector=${spark_app_selector} -o jsonpath='{.items[0].metadata.name}')
  kubectl -n spark-lama-exps port-forward svc/${svc_name} ${port}:4040
}

function logs_by_expname() {
  expname=$1

  kubectl -n spark-lama-exps logs -l spark-role=driver,expname=${expname}
}

function logs_ex_by_expname() {
  expname=$1

  kubectl -n spark-lama-exps logs -l spark-role=executor,expname=${expname}
}

function submit_job_yarn() {
  py_files=$1

  script_path=$2

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

  spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
    --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
    --conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
    --conf 'spark.kryoserializer.buffer.max=512m' \
    --conf 'spark.driver.cores=4' \
    --conf 'spark.driver.memory=5g' \
    --conf 'spark.executor.instances=8' \
    --conf 'spark.executor.cores=8' \
    --conf 'spark.executor.memory=5g' \
    --conf 'spark.cores.max=8' \
    --conf 'spark.memory.fraction=0.6' \
    --conf 'spark.memory.storageFraction=0.5' \
    --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
    --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
    --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
    --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
    --jars jars/spark-lightautoml_2.12-0.1.jar \
    --py-files ${py_files} ${script_path}
}

function submit_job_spark() {
  if [[ -z "${SPARK_MASTER_URL}" ]]
  then
    SPARK_MASTER_URL="spark://node21.bdcl:7077"
  fi

  if [[ -z "${HADOOP_DEFAULT_FS}" ]]
  then
    HADOOP_DEFAULT_FS="hdfs://node21.bdcl:9000"
  fi

  script_path=$1

  filename=$(echo ${script_path} | python -c 'import os; path = input(); print(os.path.splitext(os.path.basename(path))[0]);')

spark-submit \
  --master ${SPARK_MASTER_URL} \
  --conf 'spark.hadoop.fs.defaultFS='${HADOOP_DEFAULT_FS} \
  --conf 'spark.jars.packages=com.microsoft.azure:synapseml_2.12:0.9.5' \
  --conf 'spark.jars.repositories=https://mmlspark.azureedge.net/maven' \
  --conf 'spark.yarn.appMasterEnv.SCRIPT_ENV=cluster' \
  --conf 'spark.kryoserializer.buffer.max=512m' \
  --conf 'spark.driver.cores=4' \
  --conf 'spark.driver.memory=5g' \
  --conf 'spark.executor.instances=8' \
  --conf 'spark.executor.cores=8' \
  --conf 'spark.executor.memory=5g' \
  --conf 'spark.cores.max=8' \
  --conf 'spark.memory.fraction=0.6' \
  --conf 'spark.memory.storageFraction=0.5' \
  --conf 'spark.sql.autoBroadcastJoinThreshold=100MB' \
  --conf 'spark.sql.execution.arrow.pyspark.enabled=true' \
  --conf 'spark.scheduler.minRegisteredResourcesRatio=1.0' \
  --conf 'spark.scheduler.maxRegisteredResourcesWaitingTime=180s' \
  --jars jars/spark-lightautoml_2.12-0.1.jar \
  --py-files dist/LightAutoML-0.3.0.tar.gz ${script_path}
}

function help() {
  echo "
  Required env variables:
    KUBE_NAMESPACE - a kubernetes namespace to make actions in
    REPO - a private docker repository to push images to. It should be accessible by the cluster.

  List of commands.
    build-jars - Builds scala-based components of Slama and creates appropriate jar files in jar folder of the project
    build-pyspark-images - Builds and pushes base pyspark images required to start pyspark on cluster.
      Pushing requires remote docker repo address accessible from the cluster.
    build-lama-image - Builds and pushes a docker image to be used for running lama remotely on the cluster.
    build-dist - build_jars, build_pyspark_images, build_lama_image in a sequence
    submit-job - Submit a pyspark application with script that represent SLAMA automl app.
    submit-job-yarn - Submit a pyspark application to YARN cluster to execution.
    port-forward - Forwards port 4040 of the driver to 9040 port
    help - prints this message

  Examples:
  1. Start job
     KUBE_NAMESPACE=spark-lama-exps REPO=node2.bdcl:5000 ./bin/slamactl.sh submit-job ./examples/spark/tabular-preset-automl.py
  2. Forward Spark WebUI on local port
     KUBE_NAMESPACE=spark-lama-exps REPO=node2.bdcl:5000 ./bin/slamactl.sh port-forward ./examples/spark/tabular-preset-automl.py
  "
}


function main () {
    cmd="$1"

    if [ -z "${cmd}" ]
    then
      echo "No command is provided."
      help
      exit 1
    fi

    shift 1

    echo "Executing command: ${cmd}"

    case "${cmd}" in
    "build-jars")
        build_jars
        ;;

    "build-pyspark-images")
        build_pyspark_images
        ;;

    "build-lama-dist")
        build_lama_dist
        ;;

    "build-lama-image")
        build_lama_image
        ;;

    "build-dist")
        build_dist
        ;;

    "submit-job")
        submit_job "${@}"
        ;;

    "submit-job-yarn")
        submit_job_yarn "${@}"
        ;;

    "submit-job-spark")
        submit_job_spark "${@}"
        ;;

    "port-forward")
        port_forward "${@}"
        ;;

    "port-forward-by-expname")
        port_forward_by_expname "${@}"
        ;;

    "logs-by-expname")
        logs_by_expname "${@}"
        ;;

    "logs-ex-by-expname")
        logs_ex_by_expname "${@}"
        ;;

    "help")
        help
        ;;

    *)
        echo "Unknown command: ${cmd}"
        ;;

    esac
}

main "${@}"
