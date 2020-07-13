import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats


def conservative_score(l1, l2, n, y):
    """

    @param l1:
    @param l2:
    @param n:
    @param y:
    @return:
    """
    assert len(l1) == len(l2)

    def convert(x, n, y):  # Convert what to what? Variable names shadow enclosing scope
        """
        Add brief description of function here.

        @param x:
        @param n:
        @param y:
        @return:
        """
        if x == n:
            return -1
        if x == y:
            return 1
        return 0

    return (l1.map(lambda x: convert(x, n, y)).sum() - l2.map(lambda x: convert(x, n, y)).sum()) / 2 / len(l1)


def create_agreements(ag_df):
    """

    @param ag_df:
    @return:
    """
    ag = pd.DataFrame([[accuracy_score(ag_df[x], ag_df[y]) for x in ag_df.columns] for y in ag_df.columns])
    ag_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    ag.columns = ag_df.columns
    ag.index = ag_df.columns
    return ag


def create_conservative(ag_df):
    """

    @param ag_df:
    @return:
    """
    cons = pd.DataFrame([[conservative_score(ag_df[x], ag_df[y], -1, 1) for x in ag_df.columns] for y in ag_df.columns])
    ag_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    cons.columns = ag_df.columns
    cons.index = ag_df.columns
    return cons


def create_FDR(ag_df):
    """

    @param ag_df:
    @return:
    """
    n = ag_df.shape[1]
    FDR_df = pd.DataFrame(np.zeros((n, n)))
    for row in range(n):
        for col in range(n):
            FDR_df.iloc[row, col] = ag_df[(ag_df.iloc[:, row] == -1) & (ag_df.iloc[:, col] == 1)].shape[0] / \
                                    ag_df[ag_df.iloc[:, col] == 1].shape[0]

    FDR_df = FDR_df.T  # transpose
    FDR_df.columns = ag_df.columns
    FDR_df.index = ag_df.columns
    return FDR_df


def create_KT(ag_df):
    """

    @param ag_df:
    @return:
    """
    kts_df = pd.DataFrame([[stats.kendalltau(ag_df[x], ag_df[y])[0] for x in ag_df.columns] for y in ag_df.columns])
    kts_df.columns = ag_df.columns
    kts_df.index = ag_df.columns
    return kts_df
