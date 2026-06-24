diff --git a/autowoe/lib/selectors/selector_first.py b/autowoe/lib/selectors/selector_first.py
index 123456..789012 100644
--- a/autowoe/lib/selectors/selector_first.py
+++ b/autowoe/lib/selectors/selector_first.py
@@ -59,7 +59,7 @@ class NanConstantSelector(Selector):
         data = data.drop(columns=features_to_drop)
     elif isinstance(features_to_drop, list):
         data = data.drop(features_to_drop)
-    else:
-        data = data.drop(columns=features_to_drop, axis=1)
+    else:
+        data = data.drop(features_to_drop, axis=1)
     return data
 