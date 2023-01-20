from utils import Utils,DBUtils
from models import TrainingData,IdeaFunction
from function import FunctionManager,FunctionManagerIterator
from regression import minimise_loss,classify_point
from loss_function import calculate_squared_error
from plotting import create_ideal_function_plots,create_plot_points_with_their_ideal_function
import math

train_data_path = 'data/train_datasetsPyy.csv'
ideal_funcs_path ='data/ideal_datasetsPy.csv'


# This constant is the factor for the criterion. It is specific to the assignment
ACCEPTED_FACTOR = math.sqrt(2)

if __name__ == "__main__":
    # The FunctionManager accepts a path to a csv and parses Function objects from the data.
    # A Function  stores X and Y points of a function. It uses Pandas to do this efficiently.
    
    candidate_ideal_function_manager = FunctionManager(ideal_funcs_path)
    train_function_manager = FunctionManager(csv_path=train_data_path)

    #this line creates te SQLite database to store our data
    #creates all the tables as described by our database models
    DBUtils.create_db()

    #load the train data CSV and also populate it into the SQLite db
    DBUtils.populate_db(train_data_path,TrainingData)

    #load the ideal functions data CSV and also populate it into the SQLite db
    DBUtils.populate_db(ideal_funcs_path,IdeaFunction)

    # As Recap:
    # Within train_function_manager 4 functions are stored.
    # Withing ideal_function_manager 50 functions are stored.
    # In the next step we can use this data to compute an IdealFunction.
    # An IdealFunction amongst others stores best fitting function, the train data and is able to compute the tolerance.
    # All we now need to do is iterate over all train_functions
    # Matching ideal functions are stored in a list.
    ideal_functions = []

    for train_function in train_function_manager:
        # minimise_loss is able to compute the best fitting function given the train function
        ideal_function = minimise_loss(training_fn=train_function,
                                       candidate_fn_list=candidate_ideal_function_manager.functions,
                                       error_fn=calculate_squared_error)
        ideal_function.tolerance_factor = ACCEPTED_FACTOR
        ideal_functions.append(ideal_function)

    # We can use the classification to do some plotting
    create_ideal_function_plots(ideal_functions, "plots/train_and_ideal")

    # Now it is time to look at all points within the test data
    # The FunctionManager provides all the necessary to load a CSV, so it will be reused.
    # Instead of multiple Functions like before, it will now contain a single "Function" at location [0]
    # The benefit is that we can iterate over each point with the Function object
    test_path = "data/datasetsPy.csv"
    test_function_manager = FunctionManager(csv_path=test_path)
    test_function = test_function_manager.functions[0]

    points_with_ideal_function = []
    for point in test_function:
        ideal_function, delta_y = classify_point(point=point, ideal_fn_list=ideal_functions)
        result = {"point": point, "classification": ideal_function, "delta_y": delta_y}
        points_with_ideal_function.append(result)


    # Recap: within points_with_ideal_functions a list of dictionaries is stored.
    # These dictionaries represent the classification result of each point.

    # We can plot all the points with the corresponding classification function
    create_plot_points_with_their_ideal_function(points_with_ideal_function, "plots/point_and_ideal")

    # Finally the dict object is used to write it to a sqlite
    # In this method a pure SQLAlchamy approach has been choosen with a MetaData object to save myself from SQL-Language
    Utils.write_deviation_results_to_db(points_with_ideal_function)
    print("main.db: SQLite file with all tables")
    print("train_and_ideal.html: View the train data as scatter and the best fitting ideal function as curve")
    print("points_and_ideal.html: View for those point with a matching ideal function the distance between them in a figure")

    print("Script completed successfully")