import pandas as pd
from sqlalchemy import create_engine

class FunctionManager:
    def __init__(self, csv_path):
        """
        Reads a CSV file and creates a list of Function objects.
        :param csv_path: local path of the CSV file
        """
        self._functions = []
        try:
            self._function_data = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error reading file {csv_path}")
            raise

        x_values = self._function_data["x"]
        for column_name, column_data in self._function_data.items():
            if column_name == "x":
                continue
            subset = pd.concat([x_values, column_data], axis=1)
            function = Function.from_dataframe(column_name, subset)
            self._functions.append(function)

    @property
    def functions(self):
        """
        Returns a list of all functions.
        """
        return self._functions

    def __iter__(self):
        """
        Returns an iterator for the object.
        """
        return FunctionManagerIterator(self)

    def __repr__(self):
        return f"Holds {len(self.functions)} different functions"


class FunctionManagerIterator:
    def __init__(self, function_manager):
        """
        Iterates over a FunctionManager.
        :param function_manager: FunctionManager object
        """
        self._index = 0
        self._function_manager = function_manager

    def __next__(self):
        """
        Returns the next function in the list.
        """
        if self._index < len(self._function_manager.functions):
            value_requested = self._function_manager.functions[self._index]
            self._index += 1
            return value_requested
        raise StopIteration


class Function:
    def __init__(self, name):
        """
        Initialize a new Function object with the given name.
        :param name: the name of the function.
        """
        self._name = name
        self.dataframe = pd.DataFrame()

    def locate_y_based_on_x(self, x):
        """
        Find the y-value of the function at the given x-value.
        :param x: the x-value to look up.
        :returns: the corresponding y-value.
        :raises: IndexError if the x-value is not found.
        """
        search_key = self.dataframe["x"] == x
        try:
            return self.dataframe.loc[search_key].iat[0, 1]
        except IndexError:
            raise IndexError

    @property
    def name(self):
        """
        Get the name of the function.
        :returns: the name of the function.
        """
        return self._name

    def __iter__(self):
        return FunctionIterator(self)

    def __sub__(self, other):
        """
        Subtract the other function from this function and return the result as a new dataframe.

        :param other: the other function to subtract.
        :returns: a new dataframe with the difference between the two functions.
        """
        diff = self.dataframe - other.dataframe
        return diff

    @classmethod
    def from_dataframe(cls, name, dataframe):
        """
        Create a new Function object from the given dataframe.

        :param name: the name of the function.
        :param dataframe: the dataframe containing the function data.
        :returns: a new Function object.
        """
        function = cls(name)
        function.dataframe = dataframe
        function.dataframe.columns = ["x", "y"]
        return function

    def __repr__(self):
        return "Function for {}".format(self.name)



class FunctionIterator:

    def __init__(self, function):
        """
        Initialize a new FunctionIterator object for the given function.

        :param function: the function to iterate over.
        """
        self._function = function
        self._index = 0

    def __next__(self):
        """
        Get the next point of the function.

        :returns: a dictionary containing the x and y values of the current point.
        :raises: StopIteration if the end of the function has been reached.
        """
        if self._index < len(self._function.dataframe):
            point_series = self._function.dataframe.iloc[self._index]
            point = {"x": point_series.x, "y": point_series.y}
            self._index += 1
            return point
        raise StopIteration

class IdealFunction(Function):
    def __init__(self, function, training_function, error):
        """
        An ideal function stores the predicting function, training data and the regression.
        Make sure to provide a tolerance_factor if for classification purpose tolerance is allowed
        Otherwise it will default to the maximum deviation between ideal and train function
        :param function: the ideal function
        :param training_function: the training data the classifying data is based upon
        :param squared_error: the beforehand calculated regression
        """
        super().__init__(function.name)
        self.dataframe = function.dataframe

        self.training_function = training_function
        self.error = error
        self._tolerance_value = 1
        self._tolerance = 1

    def _determine_largest_deviation(self, ideal_function, train_function):
        # Accepts an two functions and substracts them
        # From the resulting dataframe, it finds the one which is largest
        distances = train_function - ideal_function
        distances["y"] = distances["y"].abs()
        largest_deviation = max(distances["y"])
        return largest_deviation

    @property
    def tolerance(self):
        """
        This property describes the accepted tolerance towards the regression in order to still count as classification.
        Although you can set a tolerance directly (good for unit testing) this is not recommended. Instead provide
        a tolerance_factor
        :return: the tolerance
        """
        self._tolerance = self.tolerance_factor * self.largest_deviation
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):

        self._tolerance = value

    @property
    def tolerance_factor(self):
        """
        Set the factor of the largest_deviation to determine the tolerance
        :return:
        """
        return self._tolerance_value

    @tolerance_factor.setter
    def tolerance_factor(self, value):
        self._tolerance_value = value

    @property
    def largest_deviation(self):
        """
        Retrieves the largest deviation between classifying function and the training function it is based upon
        :return: the largest deviation
        """
        largest_deviation = self._determine_largest_deviation(self, self.training_function)
        return largest_deviation

