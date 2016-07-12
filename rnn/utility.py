from math import sqrt

def mean(array, size):
    arraySum = 0.0
    for i in range(size):
        arraySum += array[i]
    return arraySum / size

def stddev(array, size):
    avg = mean(array, size)
    dev = 0.0
    for i in range(size):
        dev += (array[i] - avg) * (array[i] - avg)
    return sqrt(dev / size)
