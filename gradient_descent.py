def gradient_descent(w: float, b: float, a: float, points: list[tuple[int, int]]):
    """
    Perform one step of gradient descent to update the parameters of a linear model y = w*x + b.

    Parameters:
    w (float): weight of the model
    b (float): bias of the model
    a (float): learning rate
    points (list of tuple[int, int]): list of (x, y) data points

    Returns:
    tuple: Updated weight and bias (w, b)
    """

    # total number of points
    m = len(points)

    # initialize gradients for w and b
    dw_i = 0
    db_i = 0

    for x, y in points:
        # predicted value using current weights
        y_hat = w * x + b

        # compute partial derivates for w and b
        dw_i += -2*(y - y_hat)*x
        db_i += -2*(y - y_hat)

    # average the gradients over all data points
    dw = (1/m)*dw_i
    db = (1/m)*db_i

    # update weights using gradient descent
    w = w - (a * dw)
    b = b - (a * db)

    # return the updated weights
    return w, b
