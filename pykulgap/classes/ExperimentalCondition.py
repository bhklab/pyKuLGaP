import itertools
# Logging
import logging
from collections import Counter

# plotting dependencies
import matplotlib.pyplot as plt
import statsmodels.api as sm

# GPy dependencies
from GPy import plotting
from GPy.kern import RBF
from GPy.models import GPRegression
from scipy.integrate import quad
from scipy.stats import norm

import numpy as np
from pykulgap.helpers import calculate_AUC, compute_response_angle, relativize, centre
from pykulgap.plotting import create_measurement_dict
import pandas as pd

plotting.change_plotting_library('matplotlib')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## ---- ExperimentalCondition Object

class ExperimentalCondition:
    """
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
    """

    def __init__(self, name, source_id=None, variable=None, response=None, replicates=None,
                 variable_treatment_start=None, is_control=False):
        """
        Initialize a particular treatment condition within a cancer model. For example, exposure to a given compound
        in set of PDX models derived from a single patient.

        :param name: [string] Name of the experimental/treatment condition (e.g., Control, Erlotinib, Paclitaxel, etc.)
        :param source_id: [string] A unique identifier for the cancer model source. For PDX models this would be the
            name of id of the patient from which the models were derived. For CCLs this would be the strain from which
            all cell cultures were derived.
        :param variable: [ndarray] The independent variable of the experimental condition. For example, the treatment
            exposure time for each tumour size measurement or the dose variable for each cell viability measurement.
        :param response: [ndarray] The response metric for the experimental condition. E.g., the tumour size in a PDX
            model after variable days of treatment exposure or the cell viability measurements in a CCL at a specific
            compound dose.
        :param replicates: [ndarray] The indexes of replicate values in the response attribute.
        :param is_control: [bool] Whether or not the treatment condition is a control.
        :return [None] Creates the ExperimentalCondition object.
        """

        self.name = name
        self.variable = np.asarray([[var] for var in variable])
        self.response = np.asarray(response.T).astype(float)
        self.response_norm = None
        self.variable_end = self.variable[-1][0]
        # TODO:: Is there any situation where np.array indexing doesn't start at 0?
        self.variable_start = self.variable[0][0]
        self.variable_treatment_start = variable_treatment_start if variable_treatment_start is not None else \
            self.variable_start

        self.variable_start_index = np.where(self.variable.ravel() == self.variable_start)[0][0]
        self.variable_end_index = np.where(self.variable.ravel() == self.variable_end)[0][0]

        # Assume treatment start is the same as the start of the independent variable, unless the user assign
        self.variable_treatment_start_index = self.variable_start_index
        self.variable_treatment_end_index = self.variable_end_index

        self.source_id = source_id
        self.replicates = replicates if isinstance(replicates, list) else list(replicates)
        self.is_control = is_control
        self.kl_p_cvsc = None

        # GPs
        self.gp = None
        self.gp_kernel = None

        # all below are between the <experimental_condition> and the control
        self.empirical_kl = None

        # KL divergence stats
        self.kl_divergence = None
        self.kl_p_value = None

        # naive stats
        # {701: 'mCR', 711: 'mPR', ...}
        self.best_avg_response = np.array([], dtype=np.float64)
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

    # ---- Single Bracket Subsetting
    def __getitem__(self, item):
        """

        """
        # Deal with slices
        if isinstance(item, slice):
            if item.stop > max(self.replicates) or item.start > max(self.replicates):
                raise IndexError(f"Slice indexes out of bounds. Acceptable slice range is from "
                                 f"{min(self.replicates)} to {max(self.replicates) + 1}.")
            array = np.hstack([self.variable, self.response[item, :].T])
            return pd.DataFrame.from_records(array, columns=['variable', *['replicate_' + str(idx) for idx in
                                                                           range(item.start, item.stop,
                                                                                 item.step if item.step is not None
                                                                                 else 1)]])
        # Deal with numeric indexing
        if not isinstance(item, list):
            item = [item]
        if not all([isinstance(idx, int) for idx in item]):
            raise IndexError("Index must be an int or list of ints!")
        else:
            if max(item) > max(self.replicates) or min(item) < min(self.replicates):
                raise IndexError(f"One or more of {item} is an out of bounds index. Acceptable index range is from "
                                 f"{min(self.replicates)} to {max(self.replicates)}.")
            array = np.hstack([self.variable, self.response[item, :].T])
            return pd.DataFrame.from_records(array, columns=['variable', *['replicate_' + str(idx) for idx in item]])


    def to_dict(self, json=False):
        """
        Convert a ExperimentalCondition object into a dictionary with attributes as keys for their associated values. If
        `json` is True, all values will be coerced to JSONizable Python base types.
        """
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
    def find_variable_start_index(self):
        """
        Returns the index in the array of the location of the treatment start value, + or - 1. For a PDX model, this
        corresponds to the index of the day treatment was started.

        :return [int] The index.
        """
        start = None
        start_found = False

        for i in range(len(self.variable.ravel())):
            if self.variable[i] - 1 <= self.variable_treatment_start <= self.variable[i] + 1 and start_found is False:
                start = i
                start_found = True
        return start

    def normalize_data(self):
        """
        Normalizes all growths using normalize_first_day_and_log_transform helper function.

        :return [None] modifies self.response_norm
        """

        logger.info("Normalizing data for " + self.name)
        self.response_norm = self.__normalize_treatment_start_variable_and_log_transform(self.response,
                                                                                         self.find_variable_start_index())

    def __normalize_treatment_start_variable_and_log_transform(self, response, treatment_start_index):
        """
        Normalize by dividing every response element-wise by the first day's median
        and then taking the log.

        :param response [array] the array of values to be normalised:
        :return [array] the normalised array:
        """

        # if response.ndim == 1:
        #     return np.log((response + 0.0001) / np.median(response[treatment_start_index]))
        # else:
        # print(self.variable_treatment_start)
        # print(self.variable)
        # print(response)
        #
        # print(np.log(np.asarray((response.T + 0.01) / response.T[int(treatment_start_index)], dtype=float).T) + 1)
        return np.log(np.asarray((response.T + 0.01) / response.T[int(treatment_start_index)], dtype=float).T) + 1

    def create_full_data(self, control):
        """
        Creates a 2d numpy array with columns time, treatment and tumour size
        :param control [Boolean] whether the experimental_condition is from the control group:
        :return [None] Creates the full_data array
        """

        # control
        for j, entry in enumerate(control.response_norm.T):
            for y in entry:
                if self.full_data.size == 0:
                    self.full_data = np.array([control.variable[j][0], 0, y])
                else:
                    self.full_data = np.vstack((self.full_data, [control.variable[j][0], 0, y]))

        # case
        for j, entry in enumerate(self.response_norm.T):
            for y in entry:
                self.full_data = np.vstack((self.full_data, [self.variable[j][0], 1, y]))

    def calculate_tgi(self, control):
        """
        Calculates the Tumour Growth Index of a ExperimentalCondition object
        :param control [Boolean] whether the experimental_condition is from the control group
        :return [None] Writes the calculated value into self.tgi
        """

        def TGI(yt, yc, i, j):
            # calculates TGI between yt (Treatment) and yc (Control) during epoch i, to j
            return 1 - (yt[j] - yt[i]) / (yc[j] - yc[i])

        start = max(self.find_variable_start_index(), control.variable_treatment_start_index)
        end = min(self.variable_treatment_end_index, control.variable_treatment_end_index) + 1

        self.tgi = TGI(self.response_norm.mean(axis=0)[start:end],
                       control.response_norm.mean(axis=0)[start:end],
                       0, end - start - 1)

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
        obs_per_replicate = self.response_norm.shape[1]
        print("Now attempting to fit:")
        print("self.name:")
        print(self.name)
        print("Self.source_id:")
        print(self.source_id)

        self.gp_kernel = RBF(input_dim=1, variance=1., lengthscale=10.)

        response_norm_trunc = self.response_norm[
                                :, self.variable_treatment_start_index:self.variable_treatment_end_index
                              ]

        # # Determine index of first mouse death to remove all NaNs before fitting the model
        # first_death_idx = min(np.sum(~np.isnan(response_norm_trunc), axis=1))
        #
        # # Subset the independent variable and response data
        # response_norm_trunc = response_norm_trunc[:, 0:first_death_idx]
        # variable_trunc = self.variable[0:first_death_idx, :]

        # Reshape the data to pass into GPRegression (flatten into a single column)
        variable = np.tile(self.variable[self.variable_treatment_start_index:self.variable_treatment_end_index],
                           (len(self.replicates), 1))
        response = np.resize(response_norm_trunc, (response_norm_trunc.shape[0] * response_norm_trunc.shape[1], 1))

        self.gp = GPRegression(variable, response, self.gp_kernel)
        self.gp.optimize_restarts(num_restarts=num_restarts, messages=False)

        if control is not None:
            # Subset full data for control calculations
            # self.full_data = self.full_data[np.isin(self.full_data[:, 0], variable_trunc), :]

            # kernels
            self.gp_h0_kernel = RBF(input_dim=1, variance=1., lengthscale=10.)
            self.gp_h1_kernel = RBF(input_dim=2, variance=1., ARD=True)

            # GPs
            self.gp_h0 = GPRegression(self.full_data[:, 0:1], self.full_data[:, 2:3], self.gp_h0_kernel)
            self.gp_h1 = GPRegression(self.full_data[:, 0:2], self.full_data[:, 2:3], self.gp_h1_kernel)

            # optimize GPs
            self.gp_h0.optimize_restarts(num_restarts=num_restarts, messages=False, robust=True)  # silent exceptions
            self.gp_h1.optimize_restarts(num_restarts=num_restarts, messages=False, robust=True)

            self.delta_log_likelihood_h0_h1 = self.gp_h1.log_likelihood() - self.gp_h0.log_likelihood()

    def calculate_kl_divergence(self, control):
        """
        Calculates the KL divergence between the GPs fit for both the
        batched controls and batched cases.

        :param control: The corresponding control ExperimentalCondition object
        :return: The KL divergence
        """

        logger.info("Calculating the KL Divergence for " + self.name)

        def kl_integrand(variable):
            """
            Calculates the KL integrand
            :param variable [int?] The independent variable for the Gaussian Process Model (either time or dose).
            :return [float] The integrand
            """
            mean_control, var_control = control.gp.predict(np.asarray([[variable]]))
            mean_case, var_case = self.gp.predict(np.asarray([[variable]]))

            return ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) + (
                    (var_case + (mean_case - mean_control) ** 2) / (2 * var_control)) - 1

        max_x_index = min(self.variable_treatment_end_index, control.variable_treatment_end_index)

        if control.response.shape[1] > self.response.shape[1]:
            self.kl_divergence = abs(1 / (self.variable[max_x_index] - self.variable_treatment_start) *
                                     quad(kl_integrand, self.variable_treatment_start, self.variable[max_x_index],
                                          limit=100)[0])[0]
        else:
            self.kl_divergence = abs(1 / (control.variable[max_x_index] - self.variable_treatment_start) *
                                     quad(kl_integrand, self.variable_treatment_start, control.variable[max_x_index],
                                          limit=100)[0])[0]

        logger.info(self.kl_divergence)

    @staticmethod
    def __fit_single_gaussian_process(variable, response_norm, num_restarts=7):
        """
        GP fitting.

        Returns the GP and kernel.

        :param variable: time
        :param response_norm: log-normalized target
        :return [tuple] a tuple:
            - the gp object
            - the kernel
        """

        obs_per_replicate = response_norm.shape[1]

        kernel = RBF(input_dim=1, variance=1., lengthscale=10.)
        variable = np.tile(variable, (response_norm.shape[0], 1))
        response = np.resize(response_norm, (response_norm.shape[0] * response_norm.shape[1], 1))
        gp = GPRegression(variable, response, kernel)
        gp.optimize_restarts(num_restarts=num_restarts, messages=False)

        return gp, kernel

    @staticmethod
    def __relativize(y, start):
        """
        Normalises a numpy array to the start day
        :param response [ndarray] the array to be normalised:
        :param start [int] the start day:
        :return [ndarray] the normalised array:
        """
        return y / y[start] - 1

    @staticmethod
    def __centre(y, start):
        """
        Centres a numpy array to the start day
        :param response [ndarray] the array to be normalised:
        :param start [int] the start day:
        :return [ndarray] the normalised array:
        """
        return y - y[start]

    @staticmethod
    def __compute_response_angle(variable, response, start):
        """
        Calculates the response angle for observations response, given time points variable and start point start
        :param variable [ndarray] the time points
        :param response [ndarray] the observations
        :param start [numpy array] the start point for the angle computation
        :return [float] the angle:
        """
        min_length = min(len(variable), len(response))
        model = sm.OLS(response[start:min_length], variable[start:min_length], missing='drop')  # Drop NaNs
        results = model.fit()
        return np.arctan(results.params[0])

    def calculate_response_angles(self, control):

        """
        Builds the response angle dict.

        :param control [ExperimentalCondition] the corresponding control object
        :return [None] writes to the angle parameters 
        """
        start = self.find_variable_start_index()
        for i in range(len(self.replicates)):

            if start is None:
                raise ValueError("The `self.variable_start_index` parameter is missing, please initialize this value.")
            else:
                self.response_angle[self.replicates[i]] = self.__compute_response_angle(self.variable.ravel(),
                                                                                        self.__centre(self.response[i],
                                                                                                      start),
                                                                                        start)
                self.response_angle_rel[self.replicates[i]] = self.__compute_response_angle(self.variable.ravel(),
                                                                                            self.__relativize(
                                                                                                self.response[i],
                                                                                                start),
                                                                                            start)

        self.average_angle = self.__compute_response_angle(self.variable.ravel(),
                                                           self.__centre(np.nanmean(self.response, axis=0), start),
                                                           start)
        self.average_angle_rel = self.__compute_response_angle(self.variable.ravel(),
                                                               self.__relativize(np.nanmean(self.response, axis=0),
                                                                                 start),
                                                               start)
        self.average_angle_control = self.__compute_response_angle(control.variable.ravel(),
                                                                   self.__centre(np.nanmean(control.response, axis=0),
                                                                                 start),
                                                                   start)
        self.average_angle_rel_control = self.__compute_response_angle(control.variable.ravel(),
                                                                       self.__relativize(
                                                                           np.nanmean(control.response, axis=0),
                                                                           start), start)

    @staticmethod
    def __calculate_AUC(variable, response):
        """
        Calculates the area under the curve of a set of observations 

        :param variable [ndarray] the time points
        :param response [ndarray] the observations
        :return [float] The area under the curve
        """
        AUC = 0
        min_length = min(len(variable), len(response))
        for j in range(min_length - 1):
            AUC += (response[j + 1] - response[j]) / (variable[j + 1] - variable[j])
        return AUC

    def calculate_gp_auc(self):
        """
        Builds the AUC (Area under the curve) with respect to the GP fit.

        :return
        """
        #
        self.auc_gp = self.__calculate_AUC(self.variable, self.gp.predict(self.variable)[0])

    def calculate_auc(self, control):
        """
        Builds the AUC (Area under the curve) dict for response.
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_variable_start_index(), control.find_variable_start_index())
        end = min(self.variable_treatment_end_index, control.variable_treatment_end_index)
        for i in range(len(self.replicates)):
            self.auc[self.replicates[i]] = self.__calculate_AUC(self.variable.ravel()[start:end],
                                                                self.response[i, start:end])

    def calculate_auc_norm(self, control):
        """
        Builds the AUC (Area under the curve) dict. for response_norm
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_variable_start_index(), control.find_variable_start_index())
        end = min(self.variable_treatment_end_index, control.variable_treatment_end_index)
        for i in range(len(self.replicates)):
            self.auc_norm[self.replicates[i]] = self.__calculate_AUC(self.variable.ravel()[start:end],
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
        start = self.find_variable_start_index()
        end = self.variable_treatment_end_index
        for i in range(len(self.replicates) - 1):
            # days_volume = zip(self.variable.ravel(), self.response[i])

            if start is None:
                raise ValueError("The `start` attribute for this `ExperimentalCondition` object is set to None, "
                                 "please reset.")
            else:
                initial_volume = self.response[i][start]

                # array of all responses for t >= 3
                responses = []
                average_responses = []

                for day, volume in zip(self.variable.ravel(), self.response[i]):
                    if (day - self.variable_treatment_start >= 3) and (day <= self.variable[end]):
                        responses.append(((volume - initial_volume) / initial_volume) * 100)
                        average_responses.append(np.average(responses))

                if self.best_avg_response is not None:
                    self.best_avg_response = np.array([], dtype=np.float64)
                self.best_avg_response = np.append(self.best_avg_response, min(average_responses))
                if min(responses) < -95 and min(average_responses) < -40:
                    self.mrecist[self.replicates[i]] = 'mCR'
                elif min(responses) < -50 and min(average_responses) < -20:
                    self.mrecist[self.replicates[i]] = 'mPR'
                elif min(responses) < 35 and min(average_responses) < 30:
                    self.mrecist[self.replicates[i]] = 'mSD'
                else:
                    self.mrecist[self.replicates[i]] = 'mPD'

        for i in range(len(self.replicates)):
            days_volume = zip(self.variable.ravel(), self.response[i])
            start = self.find_variable_start_index()

            if start is None:
                raise ValueError("The `start` attribute for this `ExperimentalCondition` object is set to None, "
                                 "please reset.")
            else:
                initial_volume = self.response[i][start]

                # array of all responses for t >= 10
                responses = []
                average_responses = []

                day_diff = 0

                for day, volume in days_volume:
                    day_diff = day - self.variable_treatment_start
                    if day >= self.variable_treatment_start and day_diff >= 3:
                        responses.append(((volume - initial_volume) / initial_volume) * 100)
                        average_responses.append(np.average(responses))

                self.best_avg_response = np.append(self.best_avg_response, min(average_responses))
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
        Builds up the mrecist_counts attribute with number of each occurrence of mRECIST experimental_condition.

        :return:
        """

        # TODO:: Instead of error, we could just call method to calculate mrecist, then give the user a warning?
        if self.mrecist is None:
            raise ValueError("`ExperimentalCondition` object mrecist attribute is none, please calculate mrecist first!")

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

    def __credible_interval(self, threshold, variable_2, variable_1=0, control=None):
        """
        Credible interval function, for finding where the two GPs diverge.

        ## FIXME:: Is variable float or int?
        :param threshold [float] The variable of confidence
        :param variable_2 [int] The value of variable at the end of the range (i.e, time 2 or dose 2)
        :param variable_1 [int] The value of variable at the start of the range (i.e., time 1 or dose 1)
        :param control: the corresponding control object:
        :return:
        """
        if control is not None:
            mu = 0
            sigma = 1

            a = np.array([1, -1, -1, 1])
            means = np.array([self.gp.predict(np.asarray([[variable_2]])),
                              self.gp.predict(np.asarray([[variable_1]])),
                              control.gp.predict(np.asarray([[variable_2]])),
                              control.gp.predict(np.asarray([[variable_1]]))])[:, 0, 0]

            controlp = [control.gp.predict(np.asarray([[variable_1]])), control.gp.predict(np.asarray([[variable_2]]))]
            variances = np.zeros((4, 4))

            variances[0:2, 0:2] = self.gp.predict(np.asarray([[variable_1], [variable_2]]), full_cov=True)[1]
            variances[2:4, 2:4] = control.gp.predict(np.asarray([[variable_1], [variable_2]]), full_cov=True)[1]

            mu = np.dot(a, means)
            sigma = np.dot(np.dot(a, variances), a.T)
            interval = norm.interval(threshold, mu, sigma)

            return (interval[0] < 0) and (interval[1] > 0)
        else:
            logger.error("The private function `__credible_interval` requires control.")

    def calculate_credible_intervals(self, control):
        """
        :param control: control ExperimentalCondition object
        :return:
        """

        logger.info("Calculating credible intervals for: " + self.name)

        if control is not None:
            largest_x_index = max(len(control.variable), len(self.variable))

            if len(control.variable) > len(self.variable):
                for i in self.variable[1:]:  # Why starting at second value?
                    self.credible_intervals.append((self.__credible_interval(0.95, i[0], control=control)[0], i[0]))
            else:
                for i in control.variable[1:]:
                    self.credible_intervals.append((self.__credible_interval(0.95, i[0], control=control)[0], i[0]))
        else:
            logger.error("The function `calculate_credible_intervals` requires control.")

    def calculate_credible_intervals_percentage(self):
        """
        :return [float] The credible intervals; also has the side effect of setting the percent_credible_intervals
            attribute on the object.
        """
        logger.info("Calculating percentage of credible intervals.")

        num_true = 0
        for i in self.credible_intervals:
            if i[0] == True:
                num_true += 1

        self.percent_credible_intervals = (num_true / len(self.credible_intervals)) * 100
        return self.percent_credible_intervals

    def __gp_derivative(self, variable, gp):
        """
        Computes the derivative of the Gaussian Process gp
        (with respect to its 'time' variable) and
        returns the values of the derivative at time
        points variable to deal with some weird stuff about
        :param variable [float] The independent variable, either time for PDX models or dose for CCL models
        :param gp [GP] The GaussianProcess to be differentiated
        :return [tuple] A tuple:
            - The mean
            - The covariance
        """

        if variable.ndim == 1:
            variable = variable[:, np.newaxis]

        mu, ignore = gp.predictive_gradients(variable)
        ignore, cov = gp.predict(variable, full_cov=True)
        # FIXME:: How did this not divide by zero previously?
        mult = [[((1. / gp.kern.lengthscale) *
                  (1 - (1. / gp.kern.lengthscale) * (y - z) ** 2))[0] for y in variable if y != z]
                for z in variable]
        return mu, mult * cov

    def compute_all_gp_derivatives(self, control):
        """
        :param control [ExperimentalCondition] The control `ExperimentalCondition` for the current `CancerModel`
        :return: [None] Sets the `rates_list` attribute
        """

        if not isinstance(self.rates_list, list):
            self.rates_list = list(self.rates_list)
        if not isinstance(self.rates_list_control, list):
            self.rates_list_control = list(self.rates_list_control)

        logger.info("Calculating the GP derivatives for: " + self.name + ' and control')
        for var in self.variable:
            self.rates_list.append(self.__gp_derivative(var, self.gp)[0])
        for var in control.variable:
            self.rates_list_control.append(self.__gp_derivative(var, control.gp)[0])
        self.rates_list = np.ravel(self.rates_list)
        self.rates_list_control = np.ravel(self.rates_list_control)
        logger.info("Done calcluating GP derivatives for: " + self.name + ' and control')

    def plot_with_control(self, control=None, output_path=None, show_kl_divergence=True, show_legend=True,
                          file_type=None, output_pdf=None):
        """
        Given all of the data and an output path, saves a PDF
        of the comparison with some statistics as well.


        :param control: The control ExperimentalCondition object
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
                f"Case (Blue) and Control (Red) Comparison of \n {str(self.source_id)} with {str(self.name)}")

            # set xlim
            gp_x_limit = max(self.variable) + 5

            # Control
            control.gp.plot_data(ax=ax, color='red')
            control.gp.plot_mean(ax=ax, color='red', plot_limits=[0, gp_x_limit])
            control.gp.plot_confidence(ax=ax, color='red', plot_limits=[0, gp_x_limit])

            # Case
            self.gp.plot_data(ax=ax, color='blue')
            self.gp.plot_mean(ax=ax, color='blue', plot_limits=[0, gp_x_limit])
            self.gp.plot_confidence(ax=ax, color='blue', plot_limits=[0, gp_x_limit])

            # Drug Start Line
            plt.plot([self.variable_treatment_start, self.variable_treatment_start], [-10, 15], 'k-', lw=1)

            plt.xlabel('Day')
            plt.ylabel('Normalized log tumor size')
            plt.ylim(-10, 15)

            # Always select the longest date + 5
            plt.xlim(0, max(self.variable) + 5)

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
        Returns a string representation of the experimental_condition object.

        :return [string] The representation:
        """
        return ('\n'.join([f"Name: {self.name}",
                           f"Treatment Start Date: {self.variable_treatment_start}",
                           f"Source Id: {self.source_id}",
                           f"K-L Divergence: {self.kl_divergence}",
                           f"K-L P-Value: {self.kl_p_value}",
                           f"mRecist: {self.mrecist}",
                           f"Percent Credible Interval: {self.percent_credible_intervals}",
                           f"Rates List: {self.rates_list}"]))
