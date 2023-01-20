from function import IdealFunction
from pandas import pandas as pd

def minimise_loss(training_fn, error_fn, candidate_fn_list):
    """ 
    Returns the best fitting function from a list of candidate functions based on a training function and an error function.
    :param training_fn: The function used to train the model
    :param error_fn: The function used to calculate the error between the candidate functions and the training function
    :param candidate_fn_list: A list of functions to be evaluated
    :return: An IdealFunction object representing the function with the least error.
    """
    best_fit = None
    lowest_error = None
    for fn in candidate_fn_list:
        current_error = error_fn(training_fn, fn)
        if (lowest_error == None) or current_error < lowest_error:
            lowest_error = current_error
            best_fit = fn

    ideal_function = IdealFunction(function=best_fit, training_function=training_fn, error=lowest_error)
    return ideal_function

def classify_point(point, ideal_fn_list):
    """
    Determines the closest classification for a given point based on a list of IdealFunction objects and their tolerances.
    :param point: A dict containing "x" and "y" values
    :param ideal_fn_list: A list of IdealFunction objects
    :return: A tuple containing the closest classification and the distance to the point
    """
    closest_classification = None
    closest_distance = None

    for fn in ideal_fn_list:
        try:
            y_value = fn.locate_y_based_on_x(point["x"])
        except IndexError:
            print("Classification function does not represent this point")
            raise IndexError

        distance = abs(y_value - point["y"])

        if abs(distance) < fn.tolerance:
            if closest_classification == None or distance < closest_distance:
                closest_classification = fn
                closest_distance = distance

    return closest_classification, closest_distance