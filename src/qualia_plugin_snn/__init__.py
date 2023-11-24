"""Qualia-Plugin-SNN plugin.

When Qualia's configuration file contains:

.. code-block:: toml

    [bench]
    plugin = ['qualia_plugin_snn']

The following subpackages are imported:

* :mod:`qualia_plugin_snn.preprocessing`
* :mod:`qualia_plugin_snn.learningframework`
* :mod:`qualia_plugin_snn.postprocessing`

Subpackage :mod:`qualia_plugin_snn.deployment` contains deployers for Qualia-CodeGen referenced in
:attr:`qualia_plugin_snn.postprocessing.QualiaCodeGen.QualiaCodeGen.deployers`.

Subpackage :mod:`qualia_plugin_snn.learningframework` contains the LearningFramework implementation for SpikingJelly.

Subpackage :mod:`qualia_plugin_snn.learningmodel` contains the spiking neural network templates referenced in
:attr:`qualia_plugin_snn.learningframework.SpikingJelly.SpikingJelly.learningmodels`, made available when the SpikingJelly
learningframework is used.

Subpackage :mod:`qualia_plugin_snn.preprocessing` contains preprocessing modules adapted for or dedicated to Spiking Neural
Networks.

Subpackage :mod:`qualia_plugin_snn.postprocessing` contains postprocessing modules adapted for or dedicated to
Spiking Neural Networks, in particular the Qualia-CodeGen interface relying on :mod:`qualia_codegen_plugin_snn`,
part of Qualia-CodeGen-Plugin-SNN.
"""
