from pyspark import keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.wrapper import JavaTransformer


@inherit_doc
class BalancedUnionPartitionsCoalescerTransformer(JavaTransformer):
    """
    Custom implementation of PySpark BalancedUnionPartitionsCoalescerTransformer wrapper
    """
    @keyword_only
    def __init__(self):
        super(BalancedUnionPartitionsCoalescerTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.lightautoml.utils.BalancedUnionPartitionsCoalescerTransformer",
            self.uid
        )