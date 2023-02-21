"""Report generators and templates."""

from lightautoml.utils.installation import __validate_extra_deps

from .report_deco import ReportDeco
from .report_deco import ReportDecoNLP
from .report_deco import ReportDecoWhitebox


__validate_extra_deps("pdf")


__all__ = ["ReportDeco", "ReportDecoWhitebox", "ReportDecoNLP"]
