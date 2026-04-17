# Project Directory Overview

## .streamlit
**Configuration files**
- `config.toml`: Streamlit project configuration.

## resources
**External resources (may be consumed by external models)**
- `readme.txt`: Description of the external resources.

## CleanLogs
**Log files**
- Cleaning-log folder; records end-to-end cleaning logs.

## AnalyticsCache
**Analytics and caching — caches per-module inputs/outputs and acts as a staging area between modules**
- **ModuleTest**: Module-level test code
- `get_cleaner_excute_info.py`: Computes cleaner weights and execution order based on rules.
- `get_planuml_graph.py`: Visualizes the execution-order graph using PlantUML.
- `cleaner_associations_cycle.py`: Merges vertices on a cyclic graph and derives cleaner associations from the merged graph.
- `handle_rules.py`: Rule transformation — analyzes and instantiates mined rules for Spark execution.

## CoreSetSample
**Core-set sampling**
- **ModuleTest**: Module-level test code
- `get_patition_block.py`: Partitioning/blocking algorithm.
- `distri_samplify.py`: Samples data using the data distribution.
- `handle_data_distri.py`: Computes inter-sample distributions.
- `mapping_samplify.py`: Samples data based on mapping relationships.

## SampleScrubber
**Sample-scrubbing utilities**
- **ModuleTest**: Module-level test code
- **util**
    - `distance.py`: Distance computation.
    - `getNum.py`: Cleaning-accuracy metrics.
- `uniop_model.py`: Rule-mining model.
- `param_builder.py`: Rule-parameter construction.
- `param_selector.py`: Rule-parameter selection.
- **cleanOps**
    - `single.py`: Single-attribute operators.
    - `multiple.py`: Multi-attribute operators.
    - `soft.py`: Experimental operators.
    - `clean_penalty.py`: Cleaning-cost computation, including edit penalty, semantic-distance penalty, and Jaccard distance.

## SparkClean
**Spark-based cleaning module**
- **ModuleTest**: Module-level test code
- **util**
    - `distance.py`: Distance computation.
    - `get_types.py`: Data-type inference.
    - `cleanudf.py`: Custom Spark UDFs.
- `get_spark_rule.py`: Rule processing and parameter distribution.
- `spark_rule_model.py`: Model adapted for the Spark environment.
- `selector.py`: Parameter selection.
- `cleaner_model.py`: Cleaner data structures.
- `function_denpendency.py`: Cleaner dependency handling.

## TestDataset
**Test dataset**

## sysFlowVisualizer
**Visualization utilities**
- **cleanCache**: Cache for visualized data and charts.
- **PageHistory**: Deprecated visualization prototypes.

## pages
**Streamlit web pages**
- Frontend code for the Streamlit visualization pages.

## Main Scripts
- `Welcome.py`: Streamlit frontend entry point and home page.
- `main.py`: Terminal entry point for running a cleaning instance.
- `logsetting.py`: Logging configuration for the entry point.
- `Clean.py`: Core terminal cleaning code.
- `requirements.txt`: Entry-point dependencies.
- `Plantuml.svg`: Visualization of the cleaning pipeline.

# Roadmap
- [ ] More cleaning operators
- [ ] More detectors
- [ ] More varied datasets
