from .ExperimentalCondition import ExperimentalCondition  # , ExpCondIterator
from ..helpers import compute_response_angle, centre, relativize, calculate_AUC


class CancerModel:
    """
    A `CancerModel` represents one or more samples with the same source. For example, in PDX models it would represent
    all tumour growth measurements for mice derived from a single patient. In CCL models it would represent all
    cellular viability measurements for cultures grown with a single cancer cell line.
    """

    def __init__(self, name, source_id=None, tumour_type=None, variable_start=None, variable_treatment_start=None,
                 variable_end=None, treatment_condition_dict={}, model_type='PDX'):
        """
        Initialize attributes.

        :param name: [string] Name of the patient or PHLC Donor ID
        :param source_id: [string] The source for this cancer model. (E.g., a patient id for PDX models, a specific
            cell line for CCL models).
        :param variable_start: [float] A numeric representation of the starting date of the experiment in days.
        :param variable_end: of monitoring
        """
        # -- Defining internal data representation
        self.__name = name
        self.__source_id = source_id
        self.__tumour_type = tumour_type
        self.__variable_start = variable_start
        self.__variable_end = variable_end
        self.__variable_treatment_start = variable_treatment_start
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
    def variable_start(self):
        return self.__variable_start

    @variable_start.setter
    def variable_start(self, new_variable_start):
        self.__variable_start = new_variable_start

    @property
    def variable_treatment_start(self):
        return self.__variable_treatment_start

    @variable_treatment_start.setter
    def variable_treatment_start(self, new_variable_treatment_start):
        self.__variable_treatment_start = new_variable_treatment_start

    @property
    def variable_end(self):
        return self.__variable_end

    @variable_end.setter
    def variable_end(self, new_variable_end):
        self.__variable_end = new_variable_end

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
            raise TypeError("Please pass a dict with `ExperimentalCondition` objects as values!")
        if any([not isinstance(val, ExperimentalCondition) for val in new_treatment_conditions.values()]):
            raise TypeError("An item in your updated treatment conditions in not a `ExperimentalCondition` object.")
        self.__treatment_conditions.update(new_treatment_conditions)

    ## ---- Implementing built in methods for `CancerModel` class
    def __repr__(self):
        return ('\n'.join([f"<Cancer Model: {self.__name}",
                           f"Treatment Conditions: {list(self.__treatment_conditions.keys())}",
                           f"Source Id: {self.__source_id}",
                           f"Start Date: {self.__variable_start}",
                           f"Treatment Start Date: {self.__variable_treatment_start}",
                           f"End Date: {self.__variable_end}>"]))

    def __iter__(self):
        """Returns a dictionary object for iteration"""
        return CancerModelIterator(cancer_model=self)

    ## ---- Class methods
    def add_treatment_condition(self, treatment_condition):
        """
        Add a `ExperimentalCondition` object to

        :param treatment_condition: a ExperimentalCondition object
        """
        if not isinstance(treatment_condition, ExperimentalCondition):
            raise TypeError("Only a `ExperimentalCondition` object can be added with this method")
        if treatment_condition.name in list(self.treatment_conditions.keys()):
            raise TypeError(
                f"A treatment condition named {treatment_condition.name} already exists in the `CancerModel`")
        treatment_conds = self.treatment_conditions.copy()
        treatment_conds[treatment_condition.name] = treatment_condition
        self.treatment_conditions = treatment_conds

    def normalize_treatment_conditions(self):
        """
        Normalizes data for each ExperimentalCondition in the CancerModel object and calculates the start and end
        parameters.
             - Note: this requires the presence of a control!
        :return: [None]
        """
        control = self.treatment_conditions.get("Control")
        if not isinstance(control, ExperimentalCondition):
            raise TypeError("The `control` variable is not a `TreatmentConditon`, please ensure a treatment condition"
                            "named 'Control' exists in this object before trying to normalize.")
        for treatment_condition_name, treatment_condition in self:
            treatment_condition.normalize_data()
            if treatment_condition_name != "Control":
                treatment_condition.variable_start = max(treatment_condition.find_variable_start_index(),
                                                         control.variable_treatment_start)
                treatment_condition.end = min(control.variable_treatment_end_index,
                                              treatment_condition.variable_treatment_end_index)
                treatment_condition.create_full_data(control)
                assert treatment_condition.full_data.size != 0

    def fit_all_gps(self):
        """
        Fits Gaussian Process models to all `ExperimentalCondition`s in the `CancerModel` object
        :return: [None] Modifies the `CancerModel` object by reference.
        """
        control = self.treatment_conditions.get("Control")
        if not isinstance(control, ExperimentalCondition):
            raise TypeError(
                "The `control` variable is not a `ExperimentalCondition`, please ensure a treatment condition"
                "named 'Control' exists in this object.")
        control.fit_gaussian_processes()
        for condition_name, treatment_cond in self:
            if condition_name != "Control":
                treatment_cond.fit_gaussian_processes(control=control)
                treatment_cond.calculate_kl_divergence(control)

    def compute_summary_statistics(self, fit_gp, report_name=None):
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
        if not isinstance(control, ExperimentalCondition):
            raise TypeError(
                "The `control` variable is not a `ExperimentalCondition`, please ensure a treatment condition"
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
                    for i in control.replicates:

                        start = control.find_variable_start_index() - control.variable_treatment_start_index
                        if start is None:
                            raise TypeError("The 'start' parameter is None")
                        else:
                            treatment_condition.response_angle_control[control.replicates[i]] = \
                                compute_response_angle(
                                    variable=control.variable[
                                             control.variable_treatment_start_index:
                                             (control.variable_treatment_end_index + 1)
                                             ].ravel(),
                                    response=
                                    centre(control.response[i,
                                           control.variable_treatment_start_index:
                                           control.variable_treatment_end_index + 1],
                                           start),
                                    start=start)
                            treatment_condition.response_angle_rel_control[control.replicates[i]] = \
                                compute_response_angle(
                                    variable=control.variable[
                                             control.variable_treatment_start_index:
                                             (control.variable_treatment_end_index + 1)
                                             ].ravel(),
                                    response=
                                    relativize(control.response[i,
                                               control.variable_treatment_start_index:
                                               control.variable_treatment_end_index + 1],
                                               start),
                                    start=start)

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
                        # FIXME:: May need to swap for treatment index
                        treatment_condition.auc_gp_control = \
                            calculate_AUC(
                                control.variable[control.variable_treatment_start_index:
                                                 (control.variable_treatment_end_index + 1)],
                                control.gp.predict(
                                    control.variable[control.variable_treatment_start_index:
                                                     (control.variable_treatment_end_index + 1)])[0])
                    treatment_condition.auc_control = {}
                    start = max(treatment_condition.find_variable_start_index(), control.variable_treatment_start_index)
                    end = min(treatment_condition.variable_treatment_end_index, control.variable_treatment_end_index)
                    for i in control.replicates:
                        treatment_condition.auc_control[control.replicates[i]] = calculate_AUC(
                            control.variable[start:end],
                            control.response[i, start:end])
                        treatment_condition.auc_control_norm[control.replicates[i]] = calculate_AUC(
                            control.variable[start:end],
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
        """
        Convert a `CancerModel` object to a dictionary, with attribute names as keys for their respective values. If
        `recrusive` is True, also converts all `TreatmentCondtion` objects in the CancerModel to dictionaries such
        that only JSONizable Python types remain in the nested dictionary.
        """
        return {
            'name': self.name,
            'source_id': self.source_id,
            'tumour_type': self.tumour_type,
            'variable_start': self.variable_start,
            'variable_treatment_start': self.variable_treatment_start,
            'variable_end': self.variable_end,
            'model_type': self.model_type,
            'treatment_conditions': dict([(name, condition.to_dict(json=True)) for name, condition in self]) if
            recursive else self.treatment_conditions
        }


# -- Helper classes for CancerModel

class CancerModelIterator:
    """
    Iterator to allow looping over `CancerModel` objects. Returns a set tuples where the first item is the treatment
    condition name and the second is the `ExperimentalCondition` object.
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
