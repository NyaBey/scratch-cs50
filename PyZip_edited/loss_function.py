def calculate_squared_error(first_function, second_function):
    """
    Computes the squared error between two functions, represented by dataframes.
    :param first_function: The first function dataframe
    :param second_function: The second function dataframe
    :return: The complete deviation from squared error.
    """
    diff = second_function - first_function
    diff["y"] = diff["y"] ** 2
    deviation = diff["y"].sum()
    return deviation
