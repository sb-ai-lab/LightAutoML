"""Report generators and templates."""

from lightautoml_gpu.utils.installation import __validate_extra_deps

try:
    from .report_deco_gpu import ReportDeco
except:
    pass


__validate_extra_deps("pdf")


__all__ = ["ReportDeco"]
