#!/usr/bin/env python
import keras
import innvestigate
from ._architecture import _init_model


# def _make_analyzer(model, input_shape, n_classes, batch_size, add_softmax=False):
#   """Setup LRP analyzer for 3D-DeepLight."""
#   analyzer_model = _init_model(keras, input_shape, n_classes, batch_size, add_softmax=False)
#   return innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetBFlat(
#     model=analyzer_model,
#     neuron_selection_mode="index",
#     epsilon=1e-6)