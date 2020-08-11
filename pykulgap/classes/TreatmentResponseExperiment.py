from .CancerModel import CancerModel
from ..plotting import create_measurement_dict

import pandas as pd
import numpy as np


# ---- TreatmentResponseExperimentObject
class TreatmentResponseExperiment:
    """
    This class contains all CancerModel objects for a given treatment response experiment.

    Attributes:
        - model_names: [list] The names of the `CancerModel` object contained within the object.
        - cancer_models: [list] The list of `CancerModel` object contained within the object.
            * Note: A `TreatmentResponseExperiment` is iterable and returns a tuple of the model name and model object
                for each `CancerModel` in the object.
        - summary_stats_df: [DataFrame] Table containing summary statistics computed for all `CancerModel`s in the
            object. Computes the statistics if they don't exist already.

    Methods:
        - experimental_condition_names: [list] Returns a list of names for all unique `TreatmentConditon` within
            the object.
        - to_dict: [dict] Returns the object as a dictionary
        - compute_all_statistics: [None] Computes all summary statistics and assigns them as a DataFrame to the
            summary_stats_df attribute.
    """

    # -- Constructor
    def __init__(self, cancer_model_list):
        """
        Create a container for storing `CancerModel` objects related to a a single treatment
        response experiment.

        :param cancer_model_list: [list] A list of `CancerModel` objects.
        """
        isCancerModel = [isinstance(item, CancerModel) for item in cancer_model_list]
        if False in isCancerModel:
            raise TypeError('One or more of the items in the supplied list are not `CancerModel` objects.')
        model_names = [model.name for model in cancer_model_list]
        self.__cancer_models = dict(zip(model_names, cancer_model_list))
        self.__summary_stats_df = None

    # -- Square Bracket Subsetting

    # Allows access to CancerModels within the object by name or index
    def __getitem__(self, item):
        # Wrap single items in list to allow single/multiple item subsetting with the same code
        if not isinstance(item, list):
            item = [item]
        # Model name indexing
        if all([isinstance(name, str) for name in item]):
            if all([name in self.model_names for name in item]):
                return [self.__cancer_models.get(name) for name in item] if len(item) > 1 else \
                    self.__cancer_models.get(item[0])
        # Numeric indexing
        elif all([isinstance(idx, int) for idx in item]):
            if max(item) > len(self.model_names) - 1 or min(item) < 0:
                raise IndexError(f"One of the specified indexes is out of bounds: valid indexes must be between"
                                 f"0 and {len(self.model_names) - 1}")
            else:
                return [self.cancer_models[idx] for idx in item] if len(item) > 1 else self.cancer_models[item[0]]
        # Invalid index
        else:
            raise ValueError(f"The value(s) {item} is/are not string(s) or integer(s), valid indexes are "
                             f"{self.model_names} or a value between {0} and {len(self.model_names) - 1}")

    # Set CancerModels within the object by name or index

    # -- Accessor methods
    @property
    def model_names(self):
        return list(self.__cancer_models.keys())

    @property
    def cancer_models(self):
        return list(self.__cancer_models.values())

    @cancer_models.setter
    def cancer_models(self, new_cancer_models):
        if not isinstance(new_cancer_models, list):
            raise TypeError("You must use a `list` of `CancerModel` objects with this setter!")
        if not all([isinstance(item, CancerModel) for item in new_cancer_models]):
            raise TypeError("An element in the list is not a `CancerModel` object!")
        self.__cancer_models = new_cancer_models

    @property
    def summary_stats_df(self):
        if self.__summary_stats_df is None:
            self.compute_all_statistics(fit_gps=True)
        return self.__summary_stats_df

    # -- Implementing built-in methods
    def __repr__(self):
        return f'Cancer Models: {self.model_names}\nTreatment Conditions: {self.experimental_condition_names()}\n'

    def __iter__(self):
        """Returns the iterator object when a `TreatmentResponseExperiment` is called for looping"""
        return TREIterator(TRE=self)

    # -- Class methods
    def experimental_condition_names(self):
        """
        Return the names of all unique treatment conditions in the `TreatmentResponseExperiment` object
        """
        experimental_conditions = [item.name for sublist in [model.experimental_conditions for model in
                                                             self.cancer_models]
                                   for item in sublist]
        return list(np.unique(np.array(experimental_conditions)))

    def to_dict(self, recursive=False):
        """
        Convert a TreatmentResponseExperiment into a dictionary, where attributes become keys and their values become
        items in the returned dictionary. If `recursive` is True, it will also convert all items into dictionaries such
        that only JSONizable Python types are contained in the nested dictonary.
        """
        if recursive:
            return dict(zip(self.model_names, [model.to_dict(recursive=True) for model in self.cancer_models]))
        else:
            return dict(zip(self.model_names, self.cancer_models))

    def compute_all_statistics(self, null_kl_filename='', fit_gps=True):
        for _, cancer_model in self:
            cancer_model.normalize_experimental_conditions()
            if fit_gps:
                cancer_model.fit_all_gps()
            cancer_model.compute_summary_statistics(fit_gp=fit_gps)
        if not null_kl_filename:
            null_kl_filename = 'https://raw.githubusercontent.com/bhklab/pyKuLGaP/master/data/kl_control_vs_control.csv'
        self.__summary_stats_df = pd.DataFrame.from_dict(create_measurement_dict(self, null_kl_filename)).transpose()


# -- Helper classes for TreatmentResponseExperiment

class TREIterator:
    """
    Iterator class for the `TreatmentResponseExperiment` class, allows looping over the object.

    :return: [tuple] A tuple with the model name as the first item and `CancerModel` object as the second.
    """

    def __init__(self, TRE):
        """Initialize the iterator object with the TreatmentResponseExperiment data and an iterator index"""
        self.TRE = TRE
        self.index = 0

    def __next__(self):
        """For each row return """
        if self.index <= len(self.TRE.model_names) - 1:
            result = (self.TRE.model_names[self.index], self.TRE.cancer_models[self.index])
        else:
            raise StopIteration
        self.index += 1
        return result
