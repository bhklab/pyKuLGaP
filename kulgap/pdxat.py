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


plotting.change_plotting_library('matplotlib')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Category:
    """
    The category represents a specific category of the patient model, such as:

    - a control/vehicle, or
    - a particular drug administered.

    It can have multiple replicates (ie. multiple growth curves)
    """

    def __init__(self, name, phlc_id=None, x=None, y=None, replicates=None, drug_start_day=None, is_control=False):
        """
        Initialize a particular category of patient model.


        - x (preprocessed days)
        - y (growth data)
        - y_norm (normalized growth data)
        - gp (fit GP)
        - gp_kernel (kernel of fit GP)
        - full_data

        :param name: name of category ("Control", "Erlotinib", etc...)
        :param x: days of monitoring
        :param y: growth data
        :param replicates: IDs of all of the replicates
        :param is_control: whether or not the particular category is a control
        :return [None] Creates the Category object
        """

        self.name = name
        self.x = np.asarray([[day] for day in x])
        self.y = np.asarray(y.T).astype(float)
        self.y_norm = None
        self.drug_start_day = drug_start_day

        self.start = None
        self.end = None

        self.phlc_id = phlc_id
        self.replicates = replicates
        self.is_control = is_control
        self.kl_p_cvsc = None

        # GPs
        self.gp = None
        self.gp_kernel = None

        # all below are between the <treatment_category> and the control
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
        self.full_data = []

        # gp_h0 and gp_h1 depend on the full_data
        self.gp_h0 = None
        self.gp_h0_kernel = None
        self.gp_h1 = None
        self.gp_h1_kernel = None

        self.delta_log_likelihood_h0_h1 = None

        self.tgi = None

    def find_start_date_index(self):
        """

        Returns the index in the array of the location of the drug's
        start day, + or - 1.

        :return [int] The index:
        """
        start = None
        start_found = False

        for i in range(len(self.x.ravel())):
            if self.x[i] - 1 <= self.drug_start_day <= self.x[i] + 1 and start_found == False:
                start = i
                start_found = True
        return start

    def normalize_data(self):
        """
        Normalizes all growths using normalize_first_day_and_log_transform helper function.

        :return [None] modifies self.y_norm
        """

        logger.info("Normalizing data for " + self.name)
        self.y_norm = self.__normalize_treatment_start_day_and_log_transform(self.y,
                                                                             self.find_start_date_index())

    def __normalize_treatment_start_day_and_log_transform(self, y, treatment_start):
        """
        Normalize by dividing every y element-wise by the first day's median
        and then taking the log.

        :param y [array] the array of values to be normalised:
        :return [array] the normalised array:
        """

        # if y.ndim == 1:
        #     return np.log((y + 0.0001) / np.median(y[treatment_start]))
        # else:
        # print(self.drug_start_day)
        # print(self.x)
        # print(y)
        #
        # print(np.log(np.asarray((y.T + 0.01) / y.T[int(treatment_start)], dtype=float).T) + 1)
        return np.log(np.asarray((y.T + 0.01) / y.T[int(treatment_start)], dtype=float).T) + 1

    def create_full_data(self, control):
        """
        Creates a 2d numpy array with columns time, treatment and tumour size
        :param control [Boolean] whether the category is from the control group:
        :return [None] Creates the full_data array
        """

        # control
        for j, entry in enumerate(control.y_norm.T):
            for y in entry:
                self.full_data.append([control.x[j][0], 0, y])

        # case
        for j, entry in enumerate(self.y_norm.T):
            for y in entry:
                self.full_data.append([self.x[j][0], 1, y])

        self.full_data = np.asarray(self.full_data).astype(float)

    def calculate_tgi(self, control):
        """
        Calculates the Tumour Growth Index of a Category object
        :param control [Boolean] whether the category is from the control group:
        :return [None] Writes the calculated value into self.tgi
        """
        def TGI(yt, yc, i, j):
            # calculates TGI between yt (Treatment) and yc (Control) during epoch i, to j
            return 1 - (yt[j] - yt[i]) / (yc[j] - yc[i])

        self.tgi = TGI(self.y_norm.mean(axis=0)[self.start:self.end + 1],
                       control.y_norm.mean(axis=0)[self.start:self.end + 1], 0, self.end - self.start)

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
        # self.y_norm.shape[0] is num replicates, [1] is num measurements
        obs_per_replicate = self.y_norm.shape[1]
        print("Now attempting to fit:")
        print("self.name:")
        print(self.name)
        print("Self.phlc_id:")
        print(self.phlc_id)

        self.gp_kernel = RBF(input_dim=1, variance=1., lengthscale=10.)

        y_norm_trunc = self.y_norm[:, self.measurement_start:self.measurement_end]
        x = np.tile(self.x[self.measurement_start:self.measurement_end], (len(self.replicates), 1))
        y = np.resize(y_norm_trunc, (y_norm_trunc.shape[0] * y_norm_trunc.shape[1], 1))
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

        :param control: The corresponding control Category object
        :return: The KL divergence
        """

        logger.info("Calculating the KL Divergence for " + self.name)

        def kl_integrand(t):
            """
            Calculates the KL integrant
            :param t [the time index]:
            :return [float] the integrand:
            """
            mean_control, var_control = control.gp.predict(np.asarray([[t]]))
            mean_case, var_case = self.gp.predict(np.asarray([[t]]))

            return ((var_control + (mean_control - mean_case) ** 2) / (2 * var_case)) + (
                    (var_case + (mean_case - mean_control) ** 2) / (2 * var_control)) - 1

        max_x_index = min(self.measurement_end, control.measurement_end)

        if control.y.shape[1] > self.y.shape[1]:
            self.kl_divergence = abs(1 / (self.x[max_x_index] - self.drug_start_day) *
                                     quad(kl_integrand, self.drug_start_day, self.x[max_x_index], limit=100)[0])[0]
        #            abs(quad(kl_integrand, self.drug_start_day, self.x[max_x_index])[0]
        #                                     - max(self.x) / 2)[0]
        else:
            self.kl_divergence = abs(1 / (control.x[max_x_index] - self.drug_start_day) *
                                     quad(kl_integrand, self.drug_start_day, control.x[max_x_index], limit=100)[0])[0]
        #            abs(quad(kl_integrand, self.drug_start_day, control.x[max_x_index])[0]- max(control.x) / 2)[0]

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

        :param control [Category] the corresponding control object
        :return [None] writes to the angle parameters 
        """

        start = self.find_start_date_index()
        for i in range(len(self.replicates)):

            if start == None:
                raise
            else:
                self.response_angle[self.replicates[i]] = self.__compute_response_angle(self.x.ravel(),
                                                                                        self.__centre(self.y[i], start),
                                                                                        start)
                self.response_angle_rel[self.replicates[i]] = self.__compute_response_angle(self.x.ravel(),
                                                                                            self.__relativize(self.y[i],
                                                                                                              start),
                                                                                            start)

            #                np.arctan((self.y[i][-1] - self.y[i][start]) / (self.x.ravel()[len(self.y[i])-1] - self.drug_start_day) )
        self.average_angle = self.__compute_response_angle(self.x.ravel(),
                                                           self.__centre(np.nanmean(self.y, axis=0), start), start)
        self.average_angle_rel = self.__compute_response_angle(self.x.ravel(),
                                                               self.__relativize(np.nanmean(self.y, axis=0), start),
                                                               start)
        self.average_angle_control = self.__compute_response_angle(control.x.ravel(),
                                                                   self.__centre(np.nanmean(control.y, axis=0), start),
                                                                   start)
        self.average_angle_rel_control = self.__compute_response_angle(control.x.ravel(),
                                                                       self.__relativize(np.nanmean(control.y, axis=0),
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
        self.auc_gp = self.__calculate_AUC(self.x, self.gp.predict(self.x)[0])

    def calculate_auc(self, control):
        """
        Builds the AUC (Area under the curve) dict for y.
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_start_date_index(), control.measurement_start)
        end = min(self.measurement_end, control.measurement_end)
        for i in range(len(self.replicates)):
            self.auc[self.replicates[i]] = self.__calculate_AUC(self.x.ravel()[start:end], self.y[i, start:end])

    def calculate_auc_norm(self, control):
        """
        Builds the AUC (Area under the curve) dict. for y_norm
        :param control: the corresponding control object:
        :return [None]:
        """
        start = max(self.find_start_date_index(), control.measurement_start)
        end = min(self.measurement_end, control.measurement_end)
        for i in range(len(self.replicates)):
            self.auc_norm[self.replicates[i]] = self.__calculate_AUC(self.x.ravel()[start:end],
                                                                     self.y_norm[i, start:end])

    def calculate_mrecist(self):
        """
        Builds the mRECIST dict.

        - **mCR**: BestResponse < -95% AND BestAverageResponse < -40%
        - **mPR**: BestResponse < -50% AND BestAverageResponse < -20%
        - **mSD**: BestResponse < 35% AND BestAverageResponse < 30%
        - **mPD**: everything else

        :return [None]
        """
        start = self.find_start_date_index()
        end = self.measurement_end
        for i in range(len(self.replicates)):
            # days_volume = zip(self.x.ravel(), self.y[i])

            if start is None:
                raise
            else:
                initial_volume = self.y[i][start]

                # array of all responses for t >= 3
                responses = []
                average_responses = []

                for day, volume in zip(self.x.ravel(), self.y[i]):
                    if (day - self.drug_start_day >= 3) and (day <= self.x[end]):
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
            days_volume = zip(self.x.ravel(), self.y[i])
            start = self.find_start_date_index()

            if start is None:
                raise
            else:
                initial_volume = self.y[i][start]

                # array of all responses for t >= 10
                responses = []
                average_responses = []

                day_diff = 0

                for day, volume in days_volume:
                    day_diff = day - self.drug_start_day
                    if day >= self.drug_start_day and day_diff >= 3:
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
        Builds up the mrecist_counts attribute with number of each occurrence of mRECIST category.

        :return:
        """

        if self.mrecist is None:
            print("Category object does not have a mRECIST attribute.")
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
c       :param control: control Category object
        :return:
        """

        logger.info("Calculating credible intervals for: " + self.name)

        if control is not None:
            largest_x_index = max(len(control.x), len(self.x))

            if len(control.x) > len(self.x):
                for i in self.x[1:]:
                    self.credible_intervals.append((self.__credible_interval(0.95, i[0], control=control)[0], i[0]))
            else:
                for i in control.x[1:]:
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
        for x in self.x:
            self.rates_list.append(self.__gp_derivative(x, self.gp)[0])
        for x in control.x:
            self.rates_list_control.append(self.__gp_derivative(x, control.gp)[0])
        self.rates_list = np.ravel(self.rates_list)
        self.rates_list_control = np.ravel(self.rates_list_control)
        logger.info("Done calcluating GP derivatives for: " + self.name + ' and control')

    def plot_with_control(self, control=None, output_path=None, show_kl_divergence=True, show_legend=True,
                          file_type=None, output_pdf=None):
        """
        Given all of the data and an output path, saves a PDF
        of the comparison with some statistics as well.


        :param control: The control Category object
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
            plt.title("Case (Blue) and Control (Red) Comparison of \n" + str(self.phlc_id) + " with " + str(self.name))

            # set xlim
            gp_x_limit = max(self.x) + 5

            # Control
            control.gp.plot_data(ax=ax, color='red')
            control.gp.plot_mean(ax=ax, color='red', plot_limits=[0, gp_x_limit])
            control.gp.plot_confidence(ax=ax, color='red', plot_limits=[0, gp_x_limit])

            # Case
            self.gp.plot_data(ax=ax, color='blue')
            self.gp.plot_mean(ax=ax, color='blue', plot_limits=[0, gp_x_limit])
            self.gp.plot_confidence(ax=ax, color='blue', plot_limits=[0, gp_x_limit])

            # Drug Start Line
            plt.plot([self.drug_start_day, self.drug_start_day], [-10, 15], 'k-', lw=1)

            plt.xlabel('Day')
            plt.ylabel('Normalized log tumor size')
            plt.ylim(-10, 15)

            # Always select the longest date + 5
            plt.xlim(0, max(self.x) + 5)

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
        Returns a string representation of the category object.

        :return [string] The representation:
        """
        
        return ("\nName: %s\n"
                "drug_start_day: %s\n"
                "phlc_id: %s\n"
                # "replicates: %s\n"
                "kl_divergence: %s\n"
                "kl_p_value: %s\n"
                "mrecist: %s\n"
                "percent_credible_intervals: %s\n"
                "rates_list: %s \n"
                % (self.name, str(self.drug_start_day), self.phlc_id,# [r for r in self.replicates], 
                   self.kl_divergence, self.kl_p_value, self.mrecist, self.percent_credible_intervals, self.rates_list))


class Patient:
    """
    The patient represents where the cancer sample comes from.
    """

    def __init__(self, name, phlc_sample=None, tumour_type=None,
                 start_date=None, drug_start_day=None,
                 end_date=None):
        """
        Initialize attributes.

        :param name: name of the patient or PHLC Donor ID
        :param phlc_sample: may not exist for all
        :param start_date: of monitoring
        :param drug_start_day: of drug administration
        :param end_date: of monitoring
        :param is_rdata: if legacy
        """

        self.name = name  # also phlc_id
        self.categories = {}

        self.phlc_sample = phlc_sample
        self.start_date = start_date
        self.drug_start_day = drug_start_day
        self.end_date = end_date

        self.tumour_type = tumour_type

        

    def add_category(self, category):
        """
        Add a category to the Patient.

        :param category: a Category object
        """
        self.categories[category.name] = category

    def __repr__(self):
        return ("\nPatient: %s\n"
                "categories: %s \n"
                "phlc sample: %s\n"
                "start date: %s\n"
                "drug start day: %s\n"
                "end date: %s\n"
                % (self.name, [key for key in self.categories], self.phlc_sample,
                   self.start_date, self.drug_start_day, self.end_date))
