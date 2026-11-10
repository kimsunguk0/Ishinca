"""Ishinca: Distilling Alpamayo 1.5 (10B) into a 4B VLA."""

__version__ = "0.1.0"
__all__ = ["IshincaModel", "build_model_inputs", "load_ishinca"]


def __getattr__(name):  # lazy: keep `import ishinca.dart` free of the heavy model dependencies
    if name == "load_ishinca":
        from ishinca.loading import load_ishinca

        return load_ishinca
    if name == "IshincaModel":
        from ishinca.modeling import IshincaModel

        return IshincaModel
    if name == "build_model_inputs":
        from ishinca.inputs import build_model_inputs

        return build_model_inputs
    raise AttributeError(name)
