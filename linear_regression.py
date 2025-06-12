# Python script for simple linear regression 

import numpy as np
import matplotlib.pyplot as plt

# define feature vector
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# define response vector
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 11, 13, 15, 15, 14, 16])

def get_coef(x, y):
    
    # get number of data points
    n = np.size(x)

    # calculate means of x and y 
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculate cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculate regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1)
    
def plot_line(x, y, b_0, b_1):
    
    # plot the data points as scatter plot
    plt.scatter(x, y, color = "m", marker = "o", s = 30)

    # create regression line
    y_pred = b_0 + b_1*x

    # plot the regression line
    plt.plot(x, y_pred, color = "g")

    # add axis labels
    plt.xlabel('x')
    plt.ylabel('y')

    # show the plot    
    plt.show()
    
# estimate coefficients
(b_0, b_1) = get_coef(x, y)
print("Coefficients:\nb_0 = {} \nb_1 = {}".format(b_0, b_1))

# plot data and regression line    
plot_line(x, y, b_0, b_1)

exit()