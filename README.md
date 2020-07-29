# pyKuLGaP

A Python package for statistical analysis and plotting of Patient Derived Xenographt (PDX) models of cancer. 

## Classes

The PyKuLGaP package provides three major classes associated with treatment response experiments. While the initial
implementation of this package was specifically designed for PDX models, we have attempted to make classes general
enough to accomodate other cancer models, such as cancer cell lines (CCLs) as well as to allow extension to other
kinds of treatment response experiments in Cancer or otherwise.

### TreatmentResponseExperiment

This class contains all CancerModel objects for a given treatment response experiment. It is the highest level class
in PyKuLGaP and stores the other two classes nested within it. This class provides a number of features to easily 
compute statistics aggregated over all CancerModels for each of the ExperimentalConditions, allowing a high level
interface for summarizing the results of a given treatment response experiment, be that in PDX models, CLLs or other
cancer model systems.

Attributes:
  - model_names: [list] The names of the `CancerModel` object contained within the object.
  - cancer_models: [list] The list of `CancerModel` object contained within the object.
    - Note: A `TreatmentResponseExperiment` (TRE) is iterable and returns a tuple of the model name and model object for 
      each `CancerModel` in the object.
  - summary_stats_df: [DataFrame] Table containing summary statistics computed for all `CancerModel`s in the TRE. 
  Computes the statistics if they don't exist already.

Methods:
  - experimental_condition_names: [list] Returns a list of names for all unique `TreatmentConditon` within the object.
  - to_dict: [dict] Returns the object as a dictionary
  - compute_all_statistics: [None] Computes all summary statistics and assigns them as a DataFrame to the 
  summary_stats_df attribute.
  
Features:
  - Single Index Subsetting:
    - treatment_response_experiment['\<cancer model name\>'] returns the named CancerModel
      - e.g., treatment_response_experiment["P1"] returns the CancerModel assocaited with Patient 1.
    - treatment_response_experiment[\<cancer model index\>] also returns the CancerModel at that index
      - e.g., treatment_response_experiment[1] returns the CancerModel in the first index, in this case still Patient 1.
  - Multiple Index Subsetting:
    - treatment_response_experiment[[<cancer model 1>, <cancer model 2>, ..., <cancer model N>]]
      - e.g., treatment_response_experiment[["P1", "P2", "P3"]] returns a list of CancerModel objects.
  - Chained Subsetting:
    - treatment_response_experiment[<cancer model name>][<experimental condition name>] returns the named
    ExperimentalCondition object from the name CancerModel.
    - treatment_response_experiment[<cancer_model_name>][<experiment condtion name>][<replicate number>] returns the 
    dose and response data for the specified replicate within the specified ExperimentalCondition and CancerModel

### CancerModel Class

A `CancerModel` represents one or more samples with the same source. For example, in PDX models it would represent
all tumour growth measurements for mice derived from a single patient. In cancer cell line (CCL) models it would 
represent all cellular viability measurements for cultures grown with a single cancer cell line.

### ExperimentalCondition Class

The `ExperimentalCondition` class stores treatment response data for an experimental condition within a `CancerModel`.
It stores all replicates for all variables of the experimental condition for a given cancer model system.

For example, in CancerModel Derived Xenograph (PDX) experiments it would store the tumour size measurements at each
exposure time for all mouse models derived from a single patient.

In cancer cell lines (CCLs) it would store all viability measurements for each dose level for all cultures derived
from a single cancer cell line and treated with a specific compound.

Thus the `ExperimentalCondition` class can be though of a storing data response data for a cancer model in two
dimensions: replicates (e.g., a specific mouse or culture) variable condition levels (e.g., a specific time or
dose).

Common experimental conditions:
  * Control, i.e. no treatment
  * Exposure to a specific drug or compound
  * Treatment with a specific type of ionizing radiation
  
It can have multiple replicates (ie. data for multiple growth curves)

## Additional Features

This documentation is a work in progress, we will expand it further over the coming months. 

In the mean time feel free to contact christopher.eeles@uhnrearch.ca for questions/troubleshooting.
