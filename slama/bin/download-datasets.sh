#!/usr/bin/env bash

set -ex

dataset_dir="/opt/spark_data/"

mkdir -p "${dataset_dir}"

wget https://www.openml.org/data/get_csv/53268/ailerons.arff -O ${dataset_dir}/ailerons.csv
wget https://www.openml.org/data/get_csv/1798106/phpV5QYya -O ${dataset_dir}/PhishingWebsites.csv
wget https://www.openml.org/data/get_csv/53515/kdd_internet_usage.arff -O ${dataset_dir}/kdd_internet_usage.csv
wget https://www.openml.org/data/get_csv/22045221/dataset -O ${dataset_dir}/nasa_phm2008.csv
wget https://www.openml.org/data/get_csv/1798816/php9VSzX6 -O ${dataset_dir}/Buzzinsocialmedia_Twitter.csv
wget https://www.openml.org/data/get_csv/52407/internet_usage.arff -O ${dataset_dir}/internet_usage.csv
wget https://www.openml.org/data/get_csv/1798765/phpYLeydd -O ${dataset_dir}/gesture_segmentation.csv
wget https://www.openml.org/data/get_csv/52422/ipums_la_97-small.arff -O ${dataset_dir}/ipums_97.csv

head -n 25001 ${dataset_dir}/Buzzinsocialmedia_Twitter.csv > ${dataset_dir}/Buzzinsocialmedia_Twitter_25k.csv

cp examples/data/sampled_app_train.csv ${dataset_dir}
unzip examples/data/small_used_cars_data.zip -d ${dataset_dir}
