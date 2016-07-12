from math import sqrt

# Mean of given values
def mean(array, size):
    arraySum = 0.0
    for i in range(size):
        arraySum += array[i]
    return arraySum / size


# Variance of given values
def variance(array, size):
    avg = mean(array, size)
    dev = 0.0
    for i in range(size):
        dev += (array[i] - avg) * (array[i] - avg)
    return dev / size


# Standard deviation of given values
def stddev(array, size):
    return sqrt(variance(array, size))


# Root mean square of given values
def rms(array, size):
    sqrSum = 0.0
    for i in range(size):
        sqrSum += array[i] * array[i]
    return sqrt(sqrSum / size)
