import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pyspark.sql import SparkSession

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask

import pyspark.sql.functions as F


DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


def dump_data(path: str, ds: SparkDataset, **meta_kwargs):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    metadata = {
        "roles": ds.roles,
        "target": ds.target_column,
        "folds": ds.folds_column,
        "task_name": ds.task.name if ds.task else None
    }
    metadata.update(meta_kwargs)

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    cols_to_rename = [
        F.col(c).alias(c.replace("(", "[").replace(")", "]"))
        for c in ds.data.columns
    ]

    ds.data.select(*cols_to_rename).write.parquet(data_file)


def load_dump_if_exist(spark: SparkSession, path: Optional[str] = None) -> Optional[Tuple[SparkDataset, Dict]]:
    if path is None:
        return None

    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file)

    cols_to_rename = [
        F.col(c).alias(c.replace("[", "(").replace("]", ")"))
        for c in df.columns
    ]

    df = df.select(*cols_to_rename).repartition(16).cache()
    df.write.mode('overwrite').format('noop').save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"]
    )

    return ds, metadata


all_datastes = {
    "used_cars_dataset": {
        "path": "/opt/spark_data/small_used_cars_data.csv",
        "train_path": "/opt/spark_data/small_used_cars_data_train.csv",
        "test_path": "/opt/spark_data/small_used_cars_data_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "tiny_used_cars_dataset": {
        "path": "/opt/spark_data/tiny_used_cars_data_cleaned.csv",
        "train_path": "/opt/spark_data/tiny_used_cars_data_cleaned_train.csv",
        "test_path": "/opt/spark_data/tiny_used_cars_data_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head50k": {
        "path": "/opt/spark_data/head50k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head50k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head50k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head60k": {
        "path": "/opt/spark_data/head60k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head60k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head60k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head62_5k": {
        "path": "/opt/spark_data/head62_5k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head62_5k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head62_5k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head100k": {
        "path": "/opt/spark_data/head100k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head100k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head100k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head65k": {
        "path": "/opt/spark_data/head65k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head65k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head65k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },


    "used_cars_dataset_head70k": {
        "path": "/opt/spark_data/head70k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head70k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head70k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_head75k": {
        "path": "/opt/spark_data/head75k_0125x_cleaned.csv",
        "train_path": "/opt/spark_data/head75k_0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/head75k_0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_tmp": {
        "path": "/opt/spark_data/tmp_cleaned.csv",
        "train_path": "/opt/spark_data/tmp_cleaned_train.csv",
        "test_path": "/opt/spark_data/tmp_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_0125x": {
        "path": "/opt/spark_data/0125x_cleaned.csv",
        "train_path": "/opt/spark_data/0125x_cleaned_train.csv",
        "test_path": "/opt/spark_data/0125x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_025x": {
        "path": "/opt/spark_data/derivative_datasets/025x_cleaned.csv",
        "train_path": "/opt/spark_data/derivative_datasets/025x_cleaned_train.csv",
        "test_path": "/opt/spark_data/derivative_datasets/025x_cleaned_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_05x": {
        "path": "/opt/spark_data/derivative_datasets/05x_dataset.csv",
        "train_path": "/opt/spark_data/derivative_datasets/05x_dataset_train.csv",
        "test_path": "/opt/spark_data/derivative_datasets/05x_dataset_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_1x": {
        "path": "/opt/spark_data/derivative_datasets/1x_dataset.csv",
        "train_path": "/opt/spark_data/derivative_datasets/1x_dataset_train.csv",
        "test_path": "/opt/spark_data/derivative_datasets/1x_dataset_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ['longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    "used_cars_dataset_no_cols_limit": {
        "path": "/opt/spark_data/small_used_cars_data.csv",
        "train_path": "/opt/spark_data/small_used_cars_data_train.csv",
        "test_path": "/opt/spark_data/small_used_cars_data_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ['Unnamed: 0', '_c0'],
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    },

    # "used_cars_dataset_2x": {
    #     "path": "/opt/spark_data/derivative_datasets/2x_cleaned.csv",
    #     "task_type": "reg",
    #     "metric_name": "mse",
    #     "target_col": "price",
    #     "roles": {
    #         "target": "price",
    #         "drop": ["dealer_zip", "description", "listed_date",
    #                  "year", 'Unnamed: 0', '_c0',
    #                  'sp_id', 'sp_name', 'trimId',
    #                  'trim_name', 'major_options', 'main_picture_url',
    #                  'interior_color', 'exterior_color'],
    #         # "numeric": ['latitude', 'longitude', 'mileage']
    #         "numeric": ['longitude', 'mileage']
    #     },
    #     "dtype": {
    #         'fleet': 'str', 'frame_damaged': 'str',
    #         'has_accidents': 'str', 'isCab': 'str',
    #         'is_cpo': 'str', 'is_new': 'str',
    #         'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
    #     }
    # },

    "lama_test_dataset": {
        "path": "/opt/spark_data/sampled_app_train.csv",
        "train_path": "/opt/spark_data/sampled_app_train_train.csv",
        "test_path": "/opt/spark_data/sampled_app_train_test.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "TARGET",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    },

    # https://www.openml.org/d/734
    "ailerons_dataset": {
        "path": "/opt/spark_data/ailerons.csv",
        "train_path": "/opt/spark_data/ailerons_train.csv",
        "test_path": "/opt/spark_data/ailerons_test.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "binaryClass",
        "roles": {"target": "binaryClass"},
    },

    # https://www.openml.org/d/4534
    "phishing_websites_dataset": {
        "path": "/opt/spark_data/PhishingWebsites.csv",
        "train_path": "/opt/spark_data/PhishingWebsites_train.csv",
        "test_path": "/opt/spark_data/PhishingWebsites_test.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Result",
        "roles": {"target": "Result"},
    },

    # https://www.openml.org/d/981
    "kdd_internet_usage": {
        "path": "/opt/spark_data/kdd_internet_usage.csv",
        "train_path": "/opt/spark_data/kdd_internet_usage_train.csv",
        "test_path": "/opt/spark_data/kdd_internet_usage_test.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Who_Pays_for_Access_Work",
        "roles": {"target": "Who_Pays_for_Access_Work"},
    },

    # https://www.openml.org/d/42821
    "nasa_dataset": {
        "path": "/opt/spark_data/nasa_phm2008.csv",
        "train_path": "/opt/spark_data/nasa_phm2008_train.csv",
        "test_path": "/opt/spark_data/nasa_phm2008_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "class",
        "roles": {"target": "class"},
    },

    # https://www.openml.org/d/4549
    "buzz_dataset": {
        "path": "/opt/spark_data/Buzzinsocialmedia_Twitter_25k.csv",
        "train_path": "/opt/spark_data/Buzzinsocialmedia_Twitter_25k_train.csv",
        "test_path": "/opt/spark_data/Buzzinsocialmedia_Twitter_25k_test.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "Annotation",
        "roles": {"target": "Annotation"},
    },

    # https://www.openml.org/d/372
    "internet_usage": {
        "path": "/opt/spark_data/internet_usage.csv",
        "train_path": "/opt/spark_data/internet_usage_train.csv",
        "test_path": "/opt/spark_data/internet_usage_test.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "Actual_Time",
        "roles": {"target": "Actual_Time"},
    },

    # https://www.openml.org/d/4538
    "gesture_segmentation": {
        "path": "/opt/spark_data/gesture_segmentation.csv",
        "train_path": "/opt/spark_data/gesture_segmentation_train.csv",
        "test_path": "/opt/spark_data/gesture_segmentation_test.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "Phase",
        "roles": {"target": "Phase"},
    },

    # https://www.openml.org/d/382
    "ipums_97": {
        "path": "/opt/spark_data/ipums_97.csv",
        "train_path": "/opt/spark_data/ipums_97_train.csv",
        "test_path": "/opt/spark_data/ipums_97_test.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "movedin",
        "roles": {"target": "movedin"},
    }
}


def datasets() -> Dict[str, Any]:

    return all_datastes


def prepared_datasets(spark: SparkSession,
                      cv: int,
                      ds_configs: List[Dict[str, Any]],
                      checkpoint_dir: Optional[str] = None) -> List[Tuple[SparkDataset, SparkDataset]]:
    sds = []
    for config in ds_configs:
        path = config['path']
        train_path = config['train_path']
        test_path = config['test_path']

        task_type = config['task_type']
        roles = config['roles']

        ds_name = os.path.basename(os.path.splitext(path)[0])

        train_dump_path = os.path.join(checkpoint_dir, f"dump_{ds_name}_{cv}_train.dump") \
            if checkpoint_dir is not None else None
        test_dump_path = os.path.join(checkpoint_dir, f"dump_{ds_name}_{cv}_test.dump") \
            if checkpoint_dir is not None else None

        res_train = load_dump_if_exist(spark, train_dump_path)
        res_test = load_dump_if_exist(spark, test_dump_path)
        if res_train and res_test:
            dumped_train_ds, _ = res_train
            dumped_test_ds, _ = res_test

            sds.append((dumped_train_ds, dumped_test_ds))
            continue

        # df = spark.read.csv(path, header=True, escape="\"")
        # df = df.cache()
        # df.write.mode('overwrite').format('noop').save()
        #
        # train_df, test_df = df.randomSplit([0.8, 0.2], seed=100)

        train_df = spark.read.csv(train_path, header=True, escape="\"")
        test_df = spark.read.csv(test_path, header=True, escape="\"")

        sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv, advanced_roles=False)
        train_ds = sreader.fit_read(train_df, roles=roles)
        test_ds = sreader.read(test_df, add_array_attrs=True)

        if train_dump_path is not None:
            dump_data(train_dump_path, train_ds, cv=cv)
        if test_dump_path is not None:
            dump_data(test_dump_path, test_ds, cv=cv)

        sds.append((train_ds, test_ds))

    return sds


def get_test_datasets(dataset: Optional[str] = None,  setting: str = "all") -> List[Dict[str, Any]]:
    dss = datasets()

    if dataset is not None:
        return [dss[dataset]]

    if setting == "fast":
        return [dss['used_cars_dataset']]
    elif setting == "multiclass":
        return [dss['gesture_segmentation'], dss['ipums_97']]
    elif setting == "reg+binary":
        return [
            dss['used_cars_dataset'],
            dss["buzz_dataset"],
            dss['lama_test_dataset'],
            dss["ailerons_dataset"],
        ]
    elif setting == "binary":
        return [
            dss['lama_test_dataset'],
            dss["ailerons_dataset"],
        ]
    elif setting == "one_reg+one_binary":
        return [
            dss['used_cars_dataset'],
            dss['lama_test_dataset']
        ]
    elif setting == "all-tasks":
        return [
            dss['used_cars_dataset'],
            dss["buzz_dataset"],
            dss['lama_test_dataset'],
            dss["ailerons_dataset"],
            dss["gesture_segmentation"],
            dss['ipums_97']
        ]
    elif setting == "all":
        # exccluding all heavy datasets
        return list(cfg for ds_name, cfg in dss.items() if not ds_name.startswith('used_cars_dataset_'))
    else:
        raise ValueError(f"Unsupported setting {setting}")
