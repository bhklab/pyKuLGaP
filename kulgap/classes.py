import itertools
# Logging
import logging
from collections import Counter

# plotting dependencies
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
# GPy dependencies
from GPy import plotting
from GPy.kern import RBF
from GPy.models import GPRegression
from scipy.integrate import quad
from scipy.stats import norm

import pandas as pd
import numpy as np
from .helpers import calculate_AUC, compute_response_angle, relativize, centre

plotting.change_plotting_library('matplotlib')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#### ---- TreatmentResponseExperiment Object

class TreatmentResponseExperiment:
    """
    This class contains all CancerModel objects for a given treatment response experiment.

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
        self.__model_names = model_names
        self.__cancer_models = cancer_model_list

    # -- Accessor methods
    @property
    def model_names(self):
        return self.__model_names

    @model_names.setter
    def model_names(self, new_model_names):
        self.__model_names = new_model_names

    @property
    def cancer_models(self):
        return self.__cancer_models

    @cancer_models.setter
    def cancer_models(self, new_cancer_models):
        if not isinstance(new_cancer_models, list):
            raise TypeError("You must use a `list` of `CancerModel` objects with this setter!")
        if not all([isinstance(item, CancerModel) for item in new_cancer_models]):
            raise TypeError("An element in the list is not a `CancerModel` object!")
        self.__cancer_models = new_cancer_models

    # -- Implementing built-in methods
    def __repr__(self):
        return f'Cancer Models: {self.model_names}\nTreatment Conditions: {self.all_treatment_conditions()}\n'

    def __iter__(self):
        """Returns the iterator object when a `TreatmentResponseExperiment` is called for looping"""
        return TREIterator(TRE=self)

    # -- Class methods
    def all_treatment_conditions(self):
        treatment_conditions = [item for sublist in [model.treatment_conditions for model in self.cancer_models]
                                for item in sublist]
        return np.unique(np.array(treatment_conditions))

    def to_dict(self, recursive=False):
        if recursive:
            return dict(zip(self.model_names, [model.to_dict(recursive=True) for model in self.cancer_models]))
        else:
            return dict(zip(self.model_names, self.cancer_models))


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


#### ---- CancerModel Object

class CancerModel:
    """
    A `CancerModel` represents one or more samples with the same source. For example, in PDX models it would represent
    all tumour growth measurements for mice derived from a single patient. In CCL models it would represent all
    cellular viability measurements for cultures grown with a single cancer cell line.
    """

    def __init__(self, name, source_id=None, tumour_type=None, level_start=None, treatment_level_start=None,
                 level_end=None, treatment_condition_dict={}, model_type='PDX'):
        """
        Initialize attributes.

        :param name: [string] Name of the patient or PHLC Donor ID
        :param source_id: [string] The source for this cancer model. (E.g., a patient id for PDX models, a specific
            cell line for CCL models.
        :param level_start: [float] A numeric representation of the starting date of the experiment in days.
        :param level_end: of monitoring
        """
        # -- Defining internal data representation
        self.__name = name
        self.__source_id = source_id
        self.__tumour_type = tumour_type
        self.__start_date = level_start
        self.__treatment_level_start = treatment_level_start
        self.__level_end = level_end
        self.__model_type = model_type
        self.__treatment_conditions = treatment_condition_dict

    ## ---- Defining object attributes to get and set attributes, allows type checking and other error handling
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name):
        self.__name = new_name

    @property
    def source_id(self):
        return self.__source_id

    @source_id.setter
    def source_id(self, new_source_id):
        self.__source_id = new_source_id

    @property
    def start_date(self):
        return self.__start_date

    @start_date.setter
    def start_date(self, new_start_date):
        self.__start_date = new_start_date

    @property
    def treatment_level_start(self):
        return self.__treatment_level_start

    @treatment_level_start.setter
    def treatment_level_start(self, new_treatment_level_start):
        self.__treatment_level_start = new_treatment_level_start

    @property
    def level_end(self):
        return self.__level_end

    @level_end.setter
    def level_end(self, new_level_end):
        self.__level_end = new_level_end

    @property
    def tumour_type(self):
        return self.__tumour_type

    @tumour_type.setter
    def tumour_type(self, new_tumour_type):
        self.__tumour_type = new_tumour_type

    @property
    def model_type(self):
        return self.__model_type

    @model_type.setter
    def model_type(self, new_model_type):
        self.__model_type = new_model_type

    @property
    def treatment_conditions(self):
        return self.__treatment_conditions

    @treatment_conditions.setter
    def treatment_conditions(self, new_treatment_conditions):
        if not isinstance(new_treatment_conditions, dict):
            raise TypeError("Please pass a dict with `TreatmentCondition` objects as values!")
        if any([not isinstance(val, TreatmentCondition) for val in new_treatment_conditions.values()]):
            raise TypeError("An item in your updated treatment conditions in not a `TreatmentCondition` object.")
        self.__treatment_conditions.update(new_treatment_conditions)

    ## ---- Implementing built in methods for `CancerModel` class
    def __repr__(self):
        return ('\n'.join([f"Cancer Model: {self.__name}",
                           f"Treatment Conditions: {list(self.__treatment_conditions.keys())}",
                           f"Source Id: {self.__source_id}",
                           f"Start Date: {self.__level_start}",
                           f"Treatment Start Date: {self.__treatment_level_start}",
                           f"End Date: {self.__level_end}"]))

    def __iter__(self):
        """Returns a dictionary object for iteration"""
        return CancerModelIterator(cancer_model=self)

    ## ---- Class methods
    def add_treatment_condition(self, treatment_condition):
        """
        Add a `TreatmentCondition` object to

        :param treatment_condition: a TreatmentCondition object
        """
        if not isinstance(treatment_condition, TreatmentCondition):
            raise TypeError("Only a `TreatmentCondition` object can be added with this method")
        if treatment_condition.name in list(self.treatment_conditions.keys()):
            raise TypeError(
                f"A treatment condition named {treatment_condition.name} already exists in the `CancerModel`")
        treatment_conds = self.treatment_conditions.copy()
        treatment_conds[treatment_condition.name] = treatment_condition
        self.treatment_conditions = treatment_conds

    def normalize_treatment_conditions(self):
        """
        Normalizes data for each TreatmentCondition in the CancerModel object and calculates the start and end
        parameters.
        Note: this requires the presence of a control!
        :return: [None]
        """
        control = self.treatment_conditions.get("Control")
        if not isinstance(control, TreatmentCondition):
            raise TypeError("The `control` variable is not a `TreatmentConditon`, please ensure a treatment condition"
                            "named 'Control' exists in this object before trying to normalize.")
        for treatment_condition_name, treatment_condition in self.__treatment_conditions.items():
            treatment_condition.normalize_data()
            if treatment_condition_name != "Control":
                treatment_condition.level_start = max(treatment_condition.find_level_start_index(), control.treatment_level_start)
                treatment_condition.end = min(control.level_end, treatment_condition.level_end)
                treatment_condition.create_full_data(control)
                assert (treatment_condition.full_data != [])

    def fit_all_gps(self):
        """
        Fits GPs to all `TreatmentCondition`s in the `CancerModel` object
        :return: [None]
        """
        control = self.treatment_conditions.get("Control")
        if not isinstance(control, TreatmentCondition):
            raise TypeError("The `control` variable is not a `TreatmentCondition`, please ensure a treatment condition"
                            "named 'Control' exists in this object.")
        control.fit_gaussian_processes()
        for condition_name, treatment_cond in self:
            if condition_name != "Control":
                treatment_cond.fit_gaussian_processes(control=control)
                treatment_cond.calculate_kl_divergence(control)

    def compute_other_measures(self, fit_gp, report_name=None):
        """
        Computes the other measures (MRECIST, angle, AUC, TGI) for all non-Control `TreatmentConditions` of the
        `CancerModel`.

        :fit_gp: whether a GP has been fit.
        :param report_name: Filename under which the error report will be saved
        :return: [None]
        """

        failed_mrecist = []
        failed_response_angle = []
        failed_AUC = []
        failed_tgi = []

        control = self.treatment_conditions["Control"]
        if not isinstance(control, TreatmentCondition):
            raise TypeError("The `control` variable is not a `TreatmentCondition`, please ensure a treatment condition"
                            "named 'Control' exists in this object.")
        for condition_name, treatment_condition in self:
            if condition_name != "Control":
                # MRECIST
                try:
                    treatment_condition.calculate_mrecist()
                    assert (treatment_condition.mrecist is not None)
                except ValueError as e:
                    failed_mrecist.append((treatment_condition.source_id, e))
                    print(e)
                    continue

                # angle
                try:
                    treatment_condition.calculate_response_angles(control)
                    assert (treatment_condition.response_angle is not None)
                    treatment_condition.response_angle_control = {}
                    for i in range(len(control.replicates)):

                        start = control.find_level_start_index() - control.treatment_level_start
                        if start is None:
                            raise TypeError("The 'start' parameter is None")
                        else:
                            treatment_condition.response_angle_control[control.replicates[i]] = compute_response_angle(
                                control.response[control.treatment_level_start:(control.level_end + 1)].ravel(),
                                centre(control.response[i, control.treatment_level_start:control.level_end + 1],
                                       start),
                                start)
                            treatment_condition.response_angle_rel_control[
                                control.replicates[i]] = compute_response_angle(
                                control.response[control.treatment_level_start:(control.level_end + 1)].ravel(),
                                relativize(control.response[i, control.treatment_level_start:control.level_end + 1],
                                           start), start)

                except ValueError as e:
                    failed_response_angle.append((treatment_condition.source_id, e))
                    print(e)
                    continue

                # compute AUC
                try:
                    treatment_condition.calculate_auc(control)
                    treatment_condition.calculate_auc_norm(control)
                    if fit_gp:
                        treatment_condition.calculate_gp_auc()
                        treatment_condition.auc_gp_control = \
                            calculate_AUC(control.response[control.treatment_level_start:(control.level_end + 1)],
                                          control.gp.predict(
                                              control.response[control.treatment_level_start:(control.level_end + 1)]
                                          )[0])
                    treatment_condition.auc_control = {}
                    start = max(treatment_condition.find_level_start_index(), control.treatment_level_start)
                    end = min(treatment_condition.level_end, control.level_end)
                    for i in range(len(control.replicates)):
                        treatment_condition.auc_control[control.replicates[i]] = calculate_AUC(control.level[start:end],
                                                                                               control.response[i, start:end])
                        treatment_condition.auc_control_norm[control.replicates[i]] = calculate_AUC(
                            control.level[start:end],
                            control.response_norm[i, start:end])
                except ValueError as e:
                    failed_AUC.append((treatment_condition.source_id, e))
                    print(e)
                    continue

                try:
                    treatment_condition.calculate_tgi(control)
                except ValueError as e:
                    failed_tgi.append((treatment_condition.source_id, e))
                    print(e)
                    continue

                    # PERCENT CREDIBLE INTERVALS
                if fit_gp:
                    treatment_condition.calculate_credible_intervals(control)
                    assert (treatment_condition.credible_intervals != [])
                    treatment_condition.calculate_credible_intervals_percentage()
                    assert (treatment_condition.percent_credible_intervals is not None)

                    # compute GP derivatives:
                    treatment_condition.compute_all_gp_derivatives(control)

        if report_name is not None:
            with open(report_name, 'w') as f:

                print("Errors calculating mRECIST:", file=f)
                print(failed_mrecist, file=f)
                print("\n\n\n", file=f)

                print("Errors calculating angles:", file=f)
                print(failed_response_angle, file=f)
                print("\n\n\n", file=f)

                print("Errors calculating angles:", file=f)
                print(failed_AUC, file=f)
                print("\n\n\n", file=f)

                print("Errors calculating angles:", file=f)
                print(failed_tgi, file=f)
                print("\n\n\n", file=f)

    def to_dict(self, recursive=False):
        return {
            'name': self.name,
            'source_id': self.source_id,
            'tumour_type': self.tumour_type,
            'level_start': self.level_start,
            'treatment_level_start': self.treatment_level_start,
            'level_end': self.level_end,
            'model_type': self.model_type,
            'treatment_conditions': dict([(name, condition.to_dict(json=True)) for name, condition in self]) if
            recursive else self.treatment_conditions
        }


# -- Helper classes for CancerModel

class CancerModelIterator:
    """
    Iterator to allow looping over `CancerModel` objects. Returns a set tuples where the first item is the treatment
    condition name and the second is the `TreatmentCondition` object.
    """

    def __init__(self, cancer_model):
        self.model = cancer_model
        self.index = 0

    def __next__(self):
        keys = list(self.model.treatment_conditions.keys())
        if self.index <= len(self.model.treatment_conditions) - 1:
            results = (keys[self.index], self.model.treatment_conditions.get(keys[self.index]))
        else:
            raise StopIteration
        self.index += 1
        return results


## ---- TreatmentCondition Object

class TreatmentCondition:
    """
    The `TreatmentCondition` class stores treatment response data for an experimental condition within a `CancerModel`.
    It stores all replicates for all levels of the experimental condition for a given cancer model system.

    For example, in CancerModel Derived Xenograph (PDX) experiments it would store the tumour size measurements at each
    exposure time for all mouse models derived from a single patient.

    In cancer cell lines (CCLs) it would store all viability measurements for each dose level for all cultures derived
    from a single cancer cell line and treated with a specific compound.

    Thus the `TreatmentCondition` class can be though of a storing data response data for a cancer model in two
    dimensions: replicates (e.g., a specific mouse or culture) x condition levels (e.g., a specific time or dose).

    Common experimental conditions:
        * Control, i.e. no treatment
        * Exposure to a specific drug or compound
        * Treatment with a specific type of ionizing radiation

    It can have multiple replicates (ie. data for multiple growth curves)
    """

    def __init__(self, name, source_id=None, level=None, response=None, replicates=None, treatment_level_start=None,
                 is_control=False):
        """
        Initialize a particular treatment condition within a cancer model. For example, exposure to a given compound
        in set of PDX models derived from a single patient.

        :param name: [string] Name of the experimental/treatment condition (e.g., Control, Erlotinib, Paclitaxel, etc.)
        :param source_id: [string] A unique identifier for the cancer model source. For PDX models this would be the
            name of id of the patient from which the models were derived. For CCLs this would be the strain from which
            all cell cultures were derived.
        :param level: [ndarray] The level of the experimental condition. For example, the treatment exposure time
            for each tumour size measurement or the dose level for each cell viability measurement.
        :param response: [ndarray] The response metric for the experimental condition. E.g., the tumour size in a PDX
            model after x days of treatment exposure or the cell viability measurements in a CCL at a specific compound
            dose.
        :param replicates: [ndarray] The
        :param is_control: [bool] Whether or not the treatment condition is a control.
        :return [None] Creates the TreatmentCondition object
        """

        self.name = name
        self.level = np.asarray([[lvl] for lvl in level])
        self.response = np.asarray(response.T).astype(float)
        self.response_norm = None
        self.level_end = max(level).item()
        self.level_start = min(level).item()
        self.treatment_level_start = treatment_level_start if treatment_level_start is not None else self.level_start


        self.start = None
        self.end = None

        self.source_id = source_id
        self.replicates = replicates if isinstance(replicates, list) else list(replicates)
        self.is_control = is_control
        self.kl_p_cvsc = None


        # GPs
        self.gp = None
        self.gp_kernel = None

        # all below are between the <treatment_condition> and the control
        self.empirical_kl = None

        # KL divergence stats
        self.kl_divergence = None
        self.kl_p_value = None

        # naive stats
        # {701: 'mCR', 711: 'mPR', ...}
        self.mrecist = {}
        self.mrecist_counts = None

        # {701: response angle, ...}
        self.response_angle = {}
        self.response_angle_rel = {}

        self.response_angle_control = {}
        self.response_angle_rel_control = {}

        # response angles based on average of curves
        self.average_angle = None
        self.average_angle_rel = None
        self.average_angle_control = None
        self.average_angle_rel_control = None

        # {701: AUC, ...}
        self.auc = {}
        self.auc_norm = {}

        self.auc_gp = None
        self.auc_gp_control = None
        self.auc_control = {}
        self.auc_control_norm = {}

        self.inverted = False

        # credible intervals stats
        self.credible_intervals = []
        self.percent_credible_intervals = None

        self.rates_list = []
        self.rates_list_control = []

        # Full Data is all of the data of the treatments and control
        self.full_data = np.array([])

        # gp_h0 and gp_h1 depend on the full_data
        self.gp_h0 = None
        self.gp_h0_kernel = None
        self.gp_h1 = None
        self.gp_h1_kernel = None

        self.delta_log_likelihood_h0_h1 = None

        self.tgi = None

    def to_dict(self, json=False):
        # Helper to convert any NumPy types into base types
        def _if_numpy_to_base(object):
            if isinstance(object, np.ndarray):
                return object.tolist()
            elif isinstance(object, np.generic):
                return object.item()
            else:
                return object

        if json:
            return dict(zip(list(self.__dict__.keys()),
                            [_if_numpy_to_base(item) for item in self.__dict__.values()]))
        else:
            return self.__dict__

    ## TODO:: Can we implement this in the constructor?
    def find_level_start_index(self):
        """

        Returns the index in the array of the location of the drug's
        start day, + or - 1.

        :return [int] The index:
        """
        start = None
        start_found = False

        for i in range(len(self.level.ravel())):
            if self.level[i] - 1 <= self.treatment_level_start <= self.level[i] + 1 and start_found is False:
                start = i
                start_found = True
        return start

    def normalize_data(self):
        """
        Normalizes all growths using normalize_first_day_and_log_transform helper function.

        :return [None] modifies self.response_norm
        """

        logger.info("Normalizing data for " + self.name)
        self.response_norm = self.__normalize_treatment_start_level_and_log_transform(self.response,
                                                                                    self.treatment_level_start)

    def __normalize_treatment_start_level_and_log_transform(self, y, treatment_start):
        """
        Normalize by dividing every y element-wise by the first day's median
        and then taking the log.

        :param y [array] the array of values to be normalised:
        :return [array] the normalised array:
        """

        # if y.ndim == 1:
        #     return np.log((y + 0.0001) / np.median(y[treatment_start]))
        # else:
        # print(self.treatment_level_start)
        # print(self.level)
        # print(y)
        #
        # print(np.log(np.asarray((y.T + 0.01) / y.T[int(treatment_start)], dtype=float).T) + 1)
        return np.log(np.asarray((y.T + 0.01) / y.T[int(treatment_start)], dtype=float).T) + 1

    def create_full_data(self, control):
        """
        Creates a 2d numpy array with columns time, treatment and tumour size
        :param control [Boolean] whether the treatment_condition is from the control group:
        :return [None] Creates the full_data array
        """

        # control
        for j, entry in enumerate(control.response_norm.T):
            for y in entry:
                self.full_data = np.append(self.full_data.copy(), [control.level[j][0], 0, y], axis=0)

        # case
        for j, entry in enumerate(self.response_norm.T):
            for y in entry:
                self.full_data = np.append(self.full_data.copy(), [self.level[j][0], 1, y], axis=0)

        self.full_data = np.asarray(self.full_data).astype(float)

    def calculate_tgi(self, control):
        """
        Calculates the Tumour Growth Index of a TreatmentCondition object
        :param control [Boolean] whether the treatment_condition is from the control group:
        :return [None] Writes the calculated value into self.tgi
        """

        def TGI(yt, yc, i, j):
            # calculates TGI between yt (Treatment) and yc (Control) during epoch i, to j
            return 1 - (yt[j] - yt[i]) / (yc[j] - yc[i])

        self.tgi = TGI(self.response_norm.mean(axis=0)[self.level_start:self.level_end + 1],
                       control.response_norm.mean(axis=0)[self.level_start:self.level_end + 1], 0, self.level_end - self.level_start)

    def fit_gaussian_processes(self, control=None, num_restarts=7):
        """
        This is the new version, which fits only on the `relevant' interval
        Fits a GP for both the control and case growth curves,
        H1 with time and treatment, and H0 with only time.

        :param control If None, then just fits one GP - else, fits 3 different GPs
                        (one for case, two for gp_h0 and gp_h1):
        :param num_restarts The number of restarts in the optimisation: 
        :return [None] creates the GP objects:
        """

        logger.info("Fitting Gaussian processes for " + self.name)

        # control for number of measurements per replicate if time not same length
        # self.response_norm.shape[0] is num replicates, [1] is num measurements
        ## TODO:: Can we remove this line?
        #obs_per_replicate = self.response_norm.shape[1]
        print("Now attempting to fit:")
        print("self.name:")
        print(self.name)
        print("Self.source_id:")
        print(self.source_id)

        self.gp_kernel = RBF(input_dim=1, variance=1., lengthscale=10.)

        y = self.response_norm[:, self.treatment_level_start:self.level_end]
        x = np.tile(self.level[self.treatment_level_start:self.level_end], (len(self.replicates), 1))
        x = np.resize(x, y.shape)
        ## FIXME:: Does the GPR model keep the sample size or do we need to record it here?
        self.gp = GPRegression(x, y, self.gp_kernel)
        self.gp.optimize_restarts(num_restarts=num_restarts, messages=False)

        if control is not None:
            # kernels
            self.gp_h0_kernel = RBF(input_dim=1, variance=1., lengthscale=10.)
            self.gp_h1_kernel = RBF(input_dim=2, variance=1., ARD=True)

            # GPs
            self.gp_h0 = GPRegression(self.full_data[:, 0:1], self.full_data[:, 2:3], self.gp_h0_kernel)
            self.gp_h1 = GPRegression(self.full_data[:, 0:2], self.full_data[:, 2:3], self.gp_h1_kernel)

            # optimize GPs
            self.gp_h0.optimize_restarts(num_restarts=num_restarts, messages=False)
            self.gp_h1.optimize_restarts(num_restarts=num_restarts, messages=False)

            self.delta_log_likelihood_h0_h1 = self.gp_h1.log_likelihood() - self.gp_h0.log_likelihood()

    def calculate_kl_divergence(self, control):
        """
        Calculates the KL divergence between the GPs fit for both the
        batched controls and batched cases.

        :param control: The corresponding control TreatmentCondition object
        :return: The KL divergence
        """

        logger.info("Calculating the KL Divergence for " + self.name)

        def kl_integrand(t):
            """
            Calculates the KL integrand
            :param t [the time index]:
            :return [float] the integrand:
            """
            mean_control, var_control = control.gp.predict(np.asarray([[t]]))
            mean_case, var_case = self.gp.predict(np.asarray([[t]]))

            return ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) + (
                    (var_case + (mean_case - mean_control) ** 2) / (2 * var_control)) - 1

        max_x_index = min(self.level_end, control.level_start)

        if control.response.shape[1] > self.response.shape[1]:
            self.kl_divergence = abs(1 / (self.level[max_x_index] - self.treatment_level_start) *
                                     quad(kl_integrand, self.treatment_level_start, self.level[max_x_index], limit=100)[
                                         0])[0]
        else:
            self.kl_divergence = abs(1 / (control.level[max_x_index] - self.treatment_level_start) *
                                     quad(kl_integrand, self.treatment_level_start, control.level[max_x_index],
                                          limit=100)[0])[0]

        logger.info(self.kl_divergence)

    @staticmethod
    def __fit_single_gaussian_process(x, y_norm, num_restarts=7):
        """
        GP fitting.

        Returns the GP and kernel.

        :param x: time
        :param y_norm: log-normalized target
        :return [tuple] a tuple:
            - the gp object
            - the kernel
        """

        obs_per_replicate = y_norm.shape[1]

        kernel = RBF(input_dim=1, variance=1., lengthscale=10.)
        x = np.tile(x, (y_norm.shape[0], 1))
        y = np.resize(y_norm, (y_norm.shape[0] * y_norm.shape[1], 1))
        gp = GPRegression(x, y, kernel)
        gp.optimize_restarts(num_restarts=num_restarts, messages=False)

        return gp, kernel

    @staticmethod
    def __relativize(y, start):
        """
        Normalises a numpy array to the start day
        :param y [ndarray] the array to be normalised:
        :param start [int] the start day:
        :return [ndarray] the normalised array:
        """
        return y / y[start] - 1

    @staticmethod
    def __centre(y, start):
        """
        Centres a numpy array to the start day
        :param y [ndarray] the array to be normalised:
        :param start [int] the start day:
        :return [ndarray] the normalised array:
        """
        return y - y[start]

    @staticmethod
    def __compute_response_angle(x, y, start):
        """
        Calculates the response angle for observations y, given time points x and start point start
        :param x [ndarray] the time points: 
        :param y [ndarray] the observations:
        :param start [umpy array] the start point for the angle computation:
        :return [float] the angle:
        """
        l = min(len(x), len(y))
        model = sm.OLS(y[start:l], x[start:l])
        results = model.fit()
        return np.arctan(results.params[0])

    def calculate_response_angles(self, control):

        """
        Builds the response angle dict.

        :param control [TreatmentCondition] the corresponding control object
        :return [None] writes to the angle parameters 
        """
        start = self.treatment_level_start
        for i in range(len(self.replicates)):

            if start is None:
                raise
            else:
                self.response_angle[self.replicates[i]] = self.__compute_response_angle(self.level.ravel(),
                                                                                        self.__centre(self.response[i],
                                                                                                      start),
                                                                                        start)
                self.response_angle_rel[self.replicates[i]] = self.__compute_response_angle(self.level.ravel(),
                                                                                            self.__relativize(
                                                                                                self.response[i],
                                                                                                start),
                                                                                            start)

        self.average_angle = self.__compute_response_angle(self.level.ravel(),
                                                           self.__centre(np.nanmean(self.response, axis=0), start),
                                                           start)
        self.average_angle_rel = self.__compute_response_angle(self.level.ravel(),
                                                               self.__relativize(np.nanmean(self.response, axis=0),
                                                                                 start),
                                                               start)
        self.average_angle_control = self.__compute_response_angle(control.level.ravel(),
                                                                   self.__centre(np.nanmean(control.response, axis=0),
                                                                                 start),
                                                                   start)
        self.average_angle_rel_control = self.__compute_response_angle(control.level.ravel(),
                                                                       self.__relativize(
                                                                           np.nanmean(control.response, axis=0),
                                                                           start), start)

    @staticmethod
    def __calculate_AUC(x, y):
        """
        Calculates the area under the curve of a set of observations 

        :param x [ndarray] the time points:
        :param y [ndarray] the observations:
        :return [float] The area under the curve:
        """
        AUC = 0
        l = min(len(x), len(y))
        for j in range(l - 1):
            AUC += (y[j + 1] - y[j]) / (x[j + 1] - x[j])
        return AUC

    def calculate_gp_auc(self):
        """
        Builds the AUC (Area under the curve) with respect to the GP fit.

        :return
        """
        #
        self.auc_gp = self.__calculate_AUC(self.level, self.gp.predict(self.level)[0])

    def calculate_auc(self, control):
        """
        Builds the AUC (Area under the curve) dict for y.
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_level_start_index(), control.treatment_level_start)
        end = min(self.level_end, control.level_end)
        for i in range(len(self.replicates)):
            self.auc[self.replicates[i]] = self.__calculate_AUC(self.level.ravel()[start:end],
                                                                self.response[i, start:end])

    def calculate_auc_norm(self, control):
        """
        Builds the AUC (Area under the curve) dict. for y_norm
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_level_start_index(), control.treatment_level_start)
        end = min(self.level_end, control.level_end)
        for i in range(len(self.replicates)):
            self.auc_norm[self.replicates[i]] = self.__calculate_AUC(self.level.ravel()[start:end],
                                                                     self.response_norm[i, start:end])

    def calculate_mrecist(self):
        """
        Builds the mRECIST dict.

        - **mCR**: BestResponse < -95% AND BestAverageResponse < -40%
        - **mPR**: BestResponse < -50% AND BestAverageResponse < -20%
        - **mSD**: BestResponse < 35% AND BestAverageResponse < 30%
        - **mPD**: everything else

        :return [None]
        """
        start = self.find_level_start_index()
        end = len(self.level) - 1
        for i in range(len(self.replicates) - 1):
            # days_volume = zip(self.level.ravel(), self.response[i])

            if start is None:
                raise
            else:
                initial_volume = self.response[i][start]

                # array of all responses for t >= 3
                responses = []
                average_responses = []

                for day, volume in zip(self.level.ravel(), self.response[i]):
                    if (day - self.treatment_level_start >= 3) and (day <= self.level[end]):
                        responses.append(((volume - initial_volume) / initial_volume) * 100)
                        average_responses.append(np.average(responses))

                if min(responses) < -95 and min(average_responses) < -40:
                    self.mrecist[self.replicates[i]] = 'mCR'
                elif min(responses) < -50 and min(average_responses) < -20:
                    self.mrecist[self.replicates[i]] = 'mPR'
                elif min(responses) < 35 and min(average_responses) < 30:
                    self.mrecist[self.replicates[i]] = 'mSD'
                else:
                    self.mrecist[self.replicates[i]] = 'mPD'

        for i in range(len(self.replicates)):
            days_volume = zip(self.level.ravel(), self.response[i])
            start = self.find_level_start_index()

            if start is None:
                raise
            else:
                initial_volume = self.response[i][start]

                # array of all responses for t >= 10
                responses = []
                average_responses = []

                day_diff = 0

                for day, volume in days_volume:
                    day_diff = day - self.treatment_level_start
                    if day >= self.treatment_level_start and day_diff >= 3:
                        responses.append(((volume - initial_volume) / initial_volume) * 100)
                        average_responses.append(np.average(responses))

                if min(responses) < -95 and min(average_responses) < -40:
                    self.mrecist[self.replicates[i]] = 'mCR'
                elif min(responses) < -50 and min(average_responses) < -20:
                    self.mrecist[self.replicates[i]] = 'mPR'
                elif min(responses) < 35 and min(average_responses) < 30:
                    self.mrecist[self.replicates[i]] = 'mSD'
                else:
                    self.mrecist[self.replicates[i]] = 'mPD'

    def enumerate_mrecist(self):
        """
        Builds up the mrecist_counts attribute with number of each occurrence of mRECIST treatment_condition.

        :return:
        """

        if self.mrecist is None:
            print("TreatmentCondition object does not have a mRECIST attribute.")
            raise

        self.mrecist_counts = Counter(mCR=0, mPR=0, mSD=0, mPD=0)
        for replicate in self.replicates:
            mrecist = self.mrecist[replicate]
            if mrecist == 'mCR':
                self.mrecist_counts['mCR'] += 1
            elif mrecist == 'mPR':
                self.mrecist_counts['mPR'] += 1
            elif mrecist == 'mSD':
                self.mrecist_counts['mSD'] += 1
            elif mrecist == 'mPD':
                self.mrecist_counts['mPD'] += 1

    def __credible_interval(self, threshold, t2, t1=0, control=None):
        """
        Credible interval function, for finding where the two GPs diverge.

        :param threshold [float] The level of confidence:
        :param t2 One time point:
        :param t1 The other time point:
        :param control: the corresponding control object:
        :return:
        """
        if control is not None:
            mu = 0
            sigma = 1

            a = np.array([1, -1, -1, 1])
            means = np.array([self.gp.predict(np.asarray([[t2]])),
                              self.gp.predict(np.asarray([[t1]])),
                              control.gp.predict(np.asarray([[t2]])),
                              control.gp.predict(np.asarray([[t1]]))])[:, 0, 0]

            controlp = [control.gp.predict(np.asarray([[t1]])), control.gp.predict(np.asarray([[t2]]))]
            variances = np.zeros((4, 4))

            variances[0:2, 0:2] = self.gp.predict(np.asarray([[t1], [t2]]), full_cov=True)[1]
            variances[2:4, 2:4] = control.gp.predict(np.asarray([[t1], [t2]]), full_cov=True)[1]

            mu = np.dot(a, means)
            sigma = np.dot(np.dot(a, variances), a.T)
            interval = norm.interval(threshold, mu, sigma)

            return (interval[0] < 0) and (interval[1] > 0)
        else:
            logger.error("The private function `__credible_interval` requires control.")

    def calculate_credible_intervals(self, control):
        """
c       :param control: control TreatmentCondition object
        :return:
        """

        logger.info("Calculating credible intervals for: " + self.name)

        if control is not None:
            largest_x_index = max(len(control.level), len(self.level))

            if len(control.level) > len(self.level):
                for i in self.level[1:]:
                    self.credible_intervals.append((self.__credible_interval(0.95, i[0], control=control)[0], i[0]))
            else:
                for i in control.level[1:]:
                    self.credible_intervals.append((self.__credible_interval(0.95, i[0], control=control)[0], i[0]))
        else:
            logger.error("The function `calculate_credible_intervals` requires control.")

    def calculate_credible_intervals_percentage(self):
        """
        :return [float] The credible intervals:
        """
        logger.info("Calculating percentage of credible intervals.")

        num_true = 0
        for i in self.credible_intervals:
            if i[0] == True:
                num_true += 1

        self.percent_credible_intervals = (num_true / len(self.credible_intervals)) * 100
        return self.percent_credible_intervals

    def __gp_derivative(self, x, gp):
        """
        Computes the derivative of the Gaussian Process gp
        (with respect to its 'time' variable) and
        returns the values of the derivative at time
        points x to deal with some weird stuff about
        :param x [float] The time point:
        :param gp [GP] The GP to be differentiated:
        :return [tuple] A tuple:
            - The mean
            - The covariance
        """

        if x.ndim == 1:
            x = x[:, np.newaxis]

        mu, ignore = gp.predictive_gradients(x)
        ignore, cov = gp.predict(x, full_cov=True)
        mult = [[((1. / gp.kern.lengthscale) * (1 - (1. / gp.kern.lengthscale) * (y - z) ** 2))[0] for y in x] for z in
                x]
        return mu, mult * cov

    def compute_all_gp_derivatives(self, control):
        """
        :param control: the corresponding control object:
        :return:
        """

        logger.info("Calculating the GP derivatives for: " + self.name + ' and control')
        for x in self.level:
            self.rates_list.append(self.__gp_derivative(x, self.gp)[0])
        for x in control.level:
            self.rates_list_control.append(self.__gp_derivative(x, control.gp)[0])
        self.rates_list = np.ravel(self.rates_list)
        self.rates_list_control = np.ravel(self.rates_list_control)
        logger.info("Done calcluating GP derivatives for: " + self.name + ' and control')

    def plot_with_control(self, control=None, output_path=None, show_kl_divergence=True, show_legend=True,
                          file_type=None, output_pdf=None):
        """
        Given all of the data and an output path, saves a PDF
        of the comparison with some statistics as well.


        :param control: The control TreatmentCondition object
        :param output_path: output filepath - if not specified, doesn't save
        :param show_kl_divergence: flag for displaying calculated kl_divergence
        :param show_legend: flag for displaying legend
        :param file_type: can be 'svg' or 'pdf', defaults to 'pdf'.
        :param output_pdf: an output_pdf object
        :return:
        """

        if control is None:
            logger.error("You need to plot with a control.")
        else:
            logger.info("Plotting with statistics for " + self.name)

            fig, ax = plt.subplots()
            plt.title(
                "Case (Blue) and Control (Red) Comparison of \n" + str(self.source_id) + " with " + str(self.name))

            # set xlim
            gp_x_limit = max(self.level) + 5

            # Control
            control.gp.plot_data(ax=ax, color='red')
            control.gp.plot_mean(ax=ax, color='red', plot_limits=[0, gp_x_limit])
            control.gp.plot_confidence(ax=ax, color='red', plot_limits=[0, gp_x_limit])

            # Case
            self.gp.plot_data(ax=ax, color='blue')
            self.gp.plot_mean(ax=ax, color='blue', plot_limits=[0, gp_x_limit])
            self.gp.plot_confidence(ax=ax, color='blue', plot_limits=[0, gp_x_limit])

            # Drug Start Line
            plt.plot([self.treatment_level_start, self.treatment_level_start], [-10, 15], 'k-', lw=1)

            plt.levellabel('Day')
            plt.ylabel('Normalized log tumor size')
            plt.ylim(-10, 15)

            # Always select the longest date + 5
            plt.levellim(0, max(self.level) + 5)

            if show_kl_divergence:
                plt.text(2, -8, 'KL Divergence: ' + str(self.kl_divergence))

            if show_legend is True:
                plt.legend(loc=0)

            if file_type == 'pdf':
                output_pdf.savefig(fig)
                plt.close(fig)
            elif file_type == 'svg':
                plt.savefig(output_path, format="svg")

    def __repr__(self):
        """
        Returns a string representation of the treatment_condition object.

        :return [string] The representation:
        """
        return ('\n'.join([f"Name: {self.name}",
                           f"Treatment Start Date: {self.treatment_level_start}",
                           f"Source Id: {self.source_id}",
                           f"K-L Divergence: {self.kl_divergence}",
                           f"K-L P-Value: {self.kl_p_value}",
                           f"mRecist: {self.mrecist}",
                           f"Percent Credible Interval: {self.percent_credible_intervals}",
                           f"Rates List: {self.rates_list}"]))
