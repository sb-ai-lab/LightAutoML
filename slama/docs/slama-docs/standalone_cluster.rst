Running spark lama app on standalone cluster
============================================

Next, it will be shown how to run the ``examples/spark/tabular-preset-automl.py`` script for execution on Spark cluster.

1. First, let's go to the LightAutoML project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: imgs/LightAutoML_repo_files.png

Make sure that in the ``dist`` directory there is a wheel assembly and in the ``jars`` directory there is a jar file.
If the ``dist`` directory does not exist, or if there are no files in it, then you need to build lama dist files. ::

./bin/slamactl.sh build-lama-dist

If there are no jar file(s) in ``jars`` directory, then you need to build lama jar file(s). ::

./bin/slamactl.sh build-jars


2. Set Spark master URL via environment variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    export SPARK_MASTER_URL=spark://HOST:PORT

For example::

    export SPARK_MASTER_URL=spark://node21.bdcl:7077


3. Set Hadoop namenode address (fs.defaultFS) via environment variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    export HADOOP_DEFAULT_FS=hdfs://HOST:PORT

For example::

    export HADOOP_DEFAULT_FS=hdfs://node21.bdcl:9000

4. Submit job via ``slamactl.sh`` script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    ./bin/slamactl.sh submit-job-spark examples/spark/tabular-preset-automl.py
