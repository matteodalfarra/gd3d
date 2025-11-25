def cost_function(w: float, b: float, points: list[tuple[int, int]]):
    """
    Calculate the Mean Squared Error (MSE) for a linear model.

    Parameters:
    w (float): weight of the model
    b (float): bias of the model
    points (list of tuple[int, int]): list of (x, y) data points

    Returns:
    float: the mean squared error over all points
    """

    # total number of points
    m = len(points)
    error = 0

    for x, y in points:
        # compute the predicted value using current weight and bias
        y_hat = w * x + b
        # add the squared difference between actual and predicted value to the total error
        error += (y - y_hat)**2

    # compute the mean of the squared errors (MSE)
    j = (1/m)*(error)
    return j
