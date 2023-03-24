# Import Packages
import math
import random
import sys
import time

import numpy as np
from pyspark import SparkConf, SparkContext


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x: strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end - start) * 1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end - start) * 1000), " ms")


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def weightSummation(wt):
    wtSum = 0

    for i in range(0, len(wt)):
        wtSum = wtSum + wt[i]

    return wtSum


def bZ(z, centerPoint, radius):
    arr = []

    for i in range(0, len(z)):
        distance = euclidean(centerPoint, z[i])

        if distance <= radius:
            arr.append(1)
        else:
            arr.append(0)

    return arr


def sumBZWeights(bz, wt):
    _sum = 0

    for i in range(0, len(wt)):
        if bz[i]:
            _sum = _sum + wt[i]

    return _sum


def radiusMin(points, k, z):
    noOfPoints = k + z + 1
    minDistance = 1000000000000

    # start calculating distance between each points uptill noOfPoints
    for i in range(0, noOfPoints):
        for j in range(i + 1, noOfPoints):
            d = euclidean(points[i], points[j])
            if d < minDistance and d > 0:
                minDistance = d

    return minDistance / 2


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return res


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return math.sqrt(res)


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):
    startRound1 = time.time()
    # ------------- ROUND 1 ---------------------------

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k + z + 1))

    # END OF ROUND 1

    # ------------- ROUND 2 ---------------------------

    elems = coreset.collect()
    endRound1 = time.time()
    print("Time run round 1: ", str((endRound1 - startRound1) * 1000), " ms")

    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])

    startRound2 = time.time()
    _solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    endRound2 = time.time()
    print("Time run round 2: ", str((endRound2 - startRound2) * 1000), " ms")

    return _solution

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points) - 1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k - 1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0]  # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point, centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point, centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(points, weights, k, z, alpha):
    _z = []  # uncovered points
    s = []  # list of cluster centers
    wZ = 0
    r = 0
    bMax = 0
    newCenter = []
    _weights = weights
    guesses = 1

    wZ = weightSummation(_weights)

    rMin = radiusMin(points, k, z)
    print('Initial guess = ' + str(rMin))
    r = rMin
    while True:
        _z = points
        s = []
        # wZ = 0 already calculated

        _weights = weights
        wZ = weightSummation(_weights)
        while ((len(s) < k) and (wZ > 0)):
            bMax = 0
            newCenter = []

            for i in range(0, len(points)):
                ballWeigth = sumBZWeights(bZ(_z, points[i], (1 + 2 * alpha) * r), _weights)
                if ballWeigth > bMax:
                    bMax = ballWeigth
                    newCenter = points[i]

            s.append(newCenter)
            newTf = bZ(_z, newCenter, (3 + 4 * alpha) * r)

            newZ = []
            newWt = []
            # print("newTf", len(newTf))
            for i in range(0, len(newTf)):

                if newTf[i] == 0:
                    newZ.append(_z[i])
                    newWt.append(_weights[i])
                    # wZ = wZ - _weights[i]

            _z = newZ
            _weights = newWt  # subtracting points means remove their weights as well.
            wZ = weightSummation(_weights)

        if wZ <= z:
            return s, r, guesses
        else:
            r = 2 * r
            guesses = guesses + 1


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
    return 0


# Just start the main program
if __name__ == "__main__":
    main()
