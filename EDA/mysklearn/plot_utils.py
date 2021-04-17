# TODO: your reusable plotting functions here
from matplotlib import pyplot as plt


def frequency_diagram(x, y, x_range, y_range, title, x_label, y_label):
    """Generates bar plot showing frequency of given attributed"""
    plt.figure(figsize=(14, 8))
    plt.bar(x_range, y, 0.45, align="center")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_range, x)
    plt.yticks(y_range)
    plt.show()
    
    
def histogram(x, bins, title, x_label, y_label):
    """Draw simple histogram given the values of x"""
    plt.figure(figsize=(14, 6))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
    
def scatter_plot(x, y, title, x_label, y_label, regression_line, corr, cov, text_x, text_y):
    plt.figure(figsize=(14, 6))
    plt.scatter(x, y, c='red')
    plt.plot(x, regression_line)
    ax = plt.gca()
    ax.annotate("Corr=%.2f, Cov=%.2f" %(corr, cov),
        xy=(text_x, text_y), xycoords='data', color="r",
        bbox=dict(boxstyle="round", fc="1", color="r"))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()