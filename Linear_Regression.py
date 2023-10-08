import numpy as np
import matplotlib.pyplot as plt

# Simple AI to predict fuel efficiency of a car based on its price using linear regression and gradient descent

# The price of the car in thousands of dollars
price = np.array([25, 22, 19, 28, 24, 21, 18, 27, 23.5, 20.5,
                  17.5, 26.5, 25.5, 22.5, 19.5, 29])
# Fuel efficiency of a car defined by Miles Per Gallon
mpg = np.array([28, 34, 40, 22, 30, 36, 42, 25, 32, 38, 44, 27, 29, 35, 41, 24])
# Number of tests in training set
n = len(mpg)


def update_w_and_b(price, mpg, w, b, alpha):
    """updates w and b with the gradient"""
    prediction = w * price + b
    dl_dw = (1 / float(n)) * np.sum(-2*price*(mpg - prediction))
    dl_db = (1 / float(n)) * np.sum((-2*(mpg - prediction)))

    w = w - (1 / float(n)) * dl_dw * alpha
    b = b - (1 / float(n)) * dl_db * alpha

    return w, b


def train(price, mpg, w, b, alpha, epochs):
    """updates w and b for a set number of epochs"""
    for e in range(epochs):
        w, b = update_w_and_b(price, mpg, w, b, alpha)

        if e % 5000 == 0:
            print("epoch:", e, "loss: ", avg_loss(price, mpg, w, b))

    return w, b


def avg_loss(price, mpg, w, b):
    """calculates the cost function for a certain w and b"""
    prediction = w * price + b
    return (1 / (2*float(n))) * np.sum((mpg - prediction) ** 2)


def predict(x, w, b):
    """predicts the fuel efficiency y for a car that costs x dollars"""
    return w*x + b


def plot_graph(price, mpg, w, b, x_new, y_new):
    plt.scatter(price, mpg, color='blue', label='Data Points', zorder=1)
    plt.plot(price, w * price + b, color='red', label='Best-fit Line', zorder=1)
    plt.scatter(x_new, y_new, color='lime', label='AI Prediction', zorder=2, s=100)
    plt.xlabel('Price (in thousands of dollars)')
    plt.ylabel('MPG')
    plt.title('Price vs MPG')
    plt.legend()
    plt.show()


def main():
    w, b = train(price, mpg, 0.0, 0.0, 0.01, 100000)
    x_new = np.array([22.7, 27.8, 24.3, 26.1, 18.7, 21.3])
    y_new = w * x_new + b
    print(y_new)
    print(w, b)
    plot_graph(price, mpg, w, b, x_new, y_new)


if __name__ == "__main__":
    main()






