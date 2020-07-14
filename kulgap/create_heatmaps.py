import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score

#TODO THIS function is to be removed
# def conservative_score(l1, l2, n, y):
#     """

#     :param l1:
#     :param l2:
#     :param n:
#     :param y:
#     :return:
#     """
#     assert len(l1) == len(l2)

#     def convert(x, n, y):  # Convert what to what? Variable names shadow enclosing scope
#         """
#         Add brief description of function here.

#         :param x:
#         :param n:
#         :param y:
#         :return:
#         """
#         if x == n:
#             return -1
#         if x == y:
#             return 1
#         return 0

#     return (l1.map(lambda x: convert(x, n, y)).sum() - l2.map(lambda x: convert(x, n, y)).sum()) / 2 / len(l1)



def create_agreements(responders_df):
    """
    Creates the agreement matrix (percentage of same calls) between the different measures.
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return: [DataFrame] agreements the agreement matrix
    """
    agreements = pd.DataFrame([[accuracy_score(responders_df[x], responders_df[y]) for x in
                                responders_df.columns] for y in responders_df.columns])
    responders_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
    agreements.columns = responders_df.columns
    agreements.index = responders_df.columns
    return agreements


#TODO ThIS function is to be removed
## TODO:: Consider more informative function name. Suggestion - find_conservative_aggreements

# def create_conservative(agreements_df):
#     """

#     :param agreements_df: [DataFrame]
#     :return: []
#     """
#     conservative_agreements = pd.DataFrame([[conservative_score(agreements_df[x], agreements_df[y], -1, 1) for x in
#                                              agreements_df.columns] for y in agreements_df.columns])
#     agreements_df.rename(columns={"mRECIST-Novartis": "mRECIST"}, inplace=True)
#     conservative_agreements.columns = agreements_df.columns
#     conservative_agreements.index = agreements_df.columns
#     return conservative_agreements


def create_FDR(responders_df):
    """
    Creates the false discovery rate (FDR) matrix from the responders
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return [DataFrame]: The FDR matrix
    """
    n = responders_df.shape[1]
    FDR_df = pd.DataFrame(np.zeros((n, n)))
    for row in range(n):
        for col in range(n):
            if responders_df[responders_df.iloc[:, col] == 1].shape[0]!=0:
                FDR_df.iloc[row, col] = responders_df[(responders_df.iloc[:, row] == -1) & (responders_df.iloc[:, col] == 1)].shape[0] / \
                                    responders_df[responders_df.iloc[:, col] == 1].shape[0]
            else:
                FDR_df.iloc[row, col] =np.nan

    FDR_df = FDR_df.T  # transpose
    FDR_df.columns = responders_df.columns
    FDR_df.index = responders_df.columns
    return FDR_df


def create_KT(responders_df):
    """
    Creates the matrix of Kendall tau tests between the different responders
    :param responders_df: [DataFrame] The dataframe of responders: one column per measure, one row per experiment
    :return [DataFrame]: The matrix of Kendall tau results
    """
    kts_df = pd.DataFrame([[stats.kendalltau(responders_df[x], responders_df[y])[0] for x in responders_df.columns] for y in responders_df.columns])
    kts_df.columns = responders_df.columns
    kts_df.index = responders_df.columns
    return kts_df
