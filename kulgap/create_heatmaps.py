import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score


def conservative_score(l1, l2, n, y):
    """

    :param l1:
    :param l2:
    :param n:
    :param y:
    :return:
    """
    assert len(l1) == len(l2)

    def convert(x, n, y):  # Convert what to what? Variable names shadow enclosing scope
        """
        Add brief description of function here.

        :param x:
        :param n:
        :param y:
        :return:
        """
        if x == n:
            return -1
        if x == y:
            return 1
        return 0

    return (l1.map(lambda x: convert(x, n, y)).sum() - l2.map(lambda x: convert(x, n, y)).sum()) / 2 / len(l1)


## TODO:: Is the input to this function the agreements DF or is it finding the aggreements? Consider better parameter
##     name. Want to maximize the information provided to the user with function/paramter names. That way we minimize
##     the use of comments (which are hard to maintain) and shorten documentation.
def create_agreements(agreements_df):
    """

    :param agreements_df: [DataFrame] ...
    :return: [<return type>]
    """
    agreements = pd.DataFrame([[accuracy_score(agreements_df[x], agreements_df[y]) for x in
                                agreements_df.columns] for y in agreements_df.columns])
    agreements_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    agreements.columns = agreements_df.columns
    agreements.index = agreements_df.columns
    return agreements

## TODO:: Consider more informative function name. Suggestion - find_conservative_aggreements
def create_conservative(agreements_df):
    """

    :param agreements_df: [DataFrame]
    :return: []
    """
    conservative_agreements = pd.DataFrame([[conservative_score(agreements_df[x], agreements_df[y], -1, 1) for x in
                                             agreements_df.columns] for y in agreements_df.columns])
    agreements_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    conservative_agreements.columns = agreements_df.columns
    conservative_agreements.index = agreements_df.columns
    return conservative_agreements


def create_FDR(aggreements_df):
    """

    :param aggreements_df:
    :return:
    """
    n = aggreements_df.shape[1]
    FDR_df = pd.DataFrame(np.zeros((n, n)))
    for row in range(n):
        for col in range(n):
            FDR_df.iloc[row, col] = aggreements_df[(aggreements_df.iloc[:, row] == -1) & (aggreements_df.iloc[:, col] == 1)].shape[0] / \
                                    aggreements_df[aggreements_df.iloc[:, col] == 1].shape[0]

    FDR_df = FDR_df.T  # transpose
    FDR_df.columns = aggreements_df.columns
    FDR_df.index = aggreements_df.columns
    return FDR_df


def create_KT(ag_df):
    """

    :param ag_df:
    :return:
    """
    kts_df = pd.DataFrame([[stats.kendalltau(ag_df[x], ag_df[y])[0] for x in ag_df.columns] for y in ag_df.columns])
    kts_df.columns = ag_df.columns
    kts_df.index = ag_df.columns
    return kts_df
