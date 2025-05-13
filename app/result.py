"""
Result Data Class used to hold the output of the models
Note: Each result class is an independent frame that has been analyzed
"""

### Author: Edric Lay
### Date Created: 09/04/2025
############################################################################
import numpy as np
import camera

class Result():
    """Result object used to hold the output of analysis for a certain frame 
    """
    def __init__(self, time,result, status = True, other = None):
        """Create Result object

        Args:
            time (float): Time for when a frame occurred
            frame (np.array): Frame which got analysed
            result (variable): Result of analysis, dependent on which model it is
            status (bool, optional): status of frame. If a hand has been removed, it will return False. Defaults to True.
        """
        self.time = time
        self.result = result
        self.status = status
        self.other = other # Will be used differently depending on if hand or object

    def getTime(self):
        return self.time

def splitResult(resultList: list, timeStamp: float):
    """Will split a list into two sections based off a provided timeStamp using numpy's inbuilt binary search

    Args:
        resultList (list(Result)): List of Results
        timeStamp (float): Upper limit of time, such that the first list ends at a time less than timeStamp

    Returns:
        outputList (tuple(Lists)): Two lists, which are split at the timestamp
    """
    listTimeStamps = np.array(analysis.time for analysis in resultList) # Create a numpy array of all the timestamps in a list of results
    splitIndex = np.searchsorted(listTimeStamps, timeStamp, side="right")
    return resultList[:splitIndex], resultList[splitIndex:]

def mergeResults(*args: list):
    """Will merge multiple lists of results together. Note: you must add them in the correct order or else your video will be out of order.

    Returns:
        finalResult (list(Result)): Concatenated list of Results
    """
    finalResult = []
    for list in args:
        finalResult.extend(list)
    return finalResult