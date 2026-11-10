"""Pisco: Distilling Alpamayo 1.5 (10B) into a 2B VLA."""

__version__ = "0.1.0"
__all__ = ["PiscoModel", "build_model_inputs", "load_pisco"]


def __getattr__(name):  # lazy: keep `import pisco.dart` free of the heavy model dependencies
    if name == "load_pisco":
        from pisco.loading import load_pisco

        return load_pisco
    if name == "PiscoModel":
        from pisco.modeling import PiscoModel

        return PiscoModel
    if name == "build_model_inputs":
        from pisco.inputs import build_model_inputs

        return build_model_inputs
    raise AttributeError(name)
