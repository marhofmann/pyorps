API Reference
=============

Complete API documentation for all PYORPS packages, auto-generated from
docstrings. Signatures, parameters and return types are owned by this
reference; the hand-written pages under *API* explain purpose, limits and
examples. A machine-readable index of every documented name is published as
``api-index.json`` at the root of the built site.

Exported Names
--------------

Everything in ``pyorps.__all__``, grouped by topic.

.. currentmodule:: pyorps

Data input and rasterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   GeoDataset
   VectorDataset
   RasterDataset
   InMemoryVectorDataset
   LocalVectorDataset
   WFSVectorDataset
   LocalRasterDataset
   InMemoryRasterDataset
   initialize_geo_dataset
   GeoRasterizer
   CostAssumptions
   get_zero_cost_assumptions
   detect_feature_columns
   save_empty_cost_assumptions

Routing and results
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   PathFinder
   Path
   PathCollection
   Objective
   GradientOptions
   MetricStack
   RouteEnsemble

Cost fields, corridor graphs and tower fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   CostField
   CostFieldSet
   SavedCostField
   SearchSession
   Leg
   full_window_buffer_m
   fields_fit_in_memory
   corridor_graph_from_routes
   route_metrics
   cell_sharing_profile
   supercover_cells
   TowerField
   TowerFieldModel
   TowerFieldSolver
   TowerLattice
   AngleTables
   ClearanceModel
   solve_tower_field
   tower_field_from_raster
   tower_field_bounds
   check_bounds
   angle_tables_from_profile
   clearance_from_profile

Thin forbidden features
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   MIN_FORBIDDEN_WIDTH_CELLS
   safe_forbidden_width_m
   recommended_geometry_buffer_m
   is_thin
   min_feature_width
   thin_parts
   widen_thin_features
   detect_forbidden_burn_defects
   ForbiddenBurnReport
   ForbiddenBurnAssessment
   ForbiddenBurnSeverity
   ForbiddenBurnRoutingAssessment
   suggest_resolution
   ResolutionAdvice
   ThinForbiddenFeatureWarning
   ThinForbiddenFeatureError
   detect_repair_seals
   RepairSealReport
   SealedOpeningWarning
   SealedOpeningError

Exceptions
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   PyorpsError
   NoPathFoundError
   RasterShapeError
   AlgorithmNotImplementedError
   PairwiseError
   CostAssumptionsError
   WFSError

Packages
--------

Core Module
~~~~~~~~~~~

Essential data structures, types, and exceptions.

.. autosummary::
   :toctree: ../generated
   :recursive:
   :nosignatures:

   pyorps.core

I/O Module
~~~~~~~~~~

Geospatial data input via the GeoDataset hierarchy and the ``.gpur`` raw
raster format.

.. autosummary::
   :toctree: ../generated
   :recursive:
   :nosignatures:

   pyorps.io

Graph Module
~~~~~~~~~~~~

PathFinder, pluggable graph backends, cost fields, corridor graphs and tower
fields.

.. autosummary::
   :toctree: ../generated
   :recursive:
   :nosignatures:

   pyorps.graph

Raster Module
~~~~~~~~~~~~~

Raster data processing: rasterization, windowing and thin-feature tools.

.. autosummary::
   :toctree: ../generated
   :recursive:
   :nosignatures:

   pyorps.raster

Utils Module
~~~~~~~~~~~~

Performance-critical utilities and Cython extensions.

.. autosummary::
   :toctree: ../generated
   :recursive:
   :nosignatures:

   pyorps.utils

GUI Module
~~~~~~~~~~

Interactive workbench (needs the ``gui`` extra).

.. autosummary::
   :toctree: ../generated
   :nosignatures:

   pyorps.gui
