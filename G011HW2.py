import time
import sys
import math
import operator

def readVectorsSeq(filename):
	with open(filename) as f:
		result = [tuple(map(float, i.split(','))) for i in f]
	return result;

def euclidean(point1,point2):
	res = 0
	for i in range(len(point1)):
		diff = (point1[i]-point2[i])
		res +=  diff*diff
	return math.sqrt(res);

def weightSummation (wt):
    wtSum = 0;

    for i in range (0, len (wt)):
        wtSum = wtSum + wt[i];

    return wtSum;

def bZ (z, centerPoint, radius):
    arr = [];

    for i in range (0, len (z)):
        distance = euclidean (centerPoint, z[i]);

        if distance <= radius:
            arr.append (1);
        else:
            arr.append (0);
            

    return arr;

def sumBZWeights (bz, wt):
    _sum = 0;

    for i in range (0, len (wt)):
        if bz[i]:
            _sum = _sum + wt[i];

    return _sum;

def seqWeightedOutliers (p, k, z, alpha, weights):
    _z        = []; # uncovered points
    s         = []; # list of cluster centers
    wZ        = 0;
    r         = 0;
    bMax      = 0;
    newCenter = [];
    _weights  = weights;
    guesses   = 1;

    wZ   = weightSummation (_weights);

    rMin = radiusMin (p, k, z);
    print ('Initial guess = ' + str (rMin));
    r   = rMin;
    while True:
        _z  = p;
        s   = [];
        #wZ = 0; already calculated
        
        _weights  = weights;
        wZ   = weightSummation (_weights);
        while ((len (s) < k) and (wZ > 0)):
            bMax      = 0;
            newCenter = [];

            for i in range (0, len (p)):
                ballWeigth = sumBZWeights (bZ (_z, p[i],(1+2*alpha)*r ), _weights);
                if ballWeigth > bMax:
                    bMax      = ballWeigth;
                    newCenter = p[i];

            s.append (newCenter);
            newTf = bZ (_z, newCenter,(3+4*alpha)*r );

            newZ  = [];
            newWt = [];
            #print("newTf", len(newTf))
            for i in range (0, len (newTf)):
                
                if newTf[i] == 0:
                    newZ.append (_z[i]);
                    newWt.append (_weights[i]);
                    #wZ = wZ - _weights[i];
                    
                    

            _z       = newZ;
            _weights = newWt;# subtracting points means remove their weights as well.
            wZ   = weightSummation (_weights);

        if wZ <= z:     
            return s, r, guesses;
        else:
            r       = 2 * r;
            guesses = guesses + 1;

def radiusMin (points, k, z):
    noOfPoints  = k + z + 1
    minDistance = 1000000000000;

    # start calculating distance between each points uptill noOfPoints
    for i in range (0, noOfPoints):
        for j in range (i+1, noOfPoints):
            d = euclidean (points[i], points[j]);
            if d < minDistance and d > 0:
                minDistance = d

    return minDistance / 2;

def initializeUnitWeights (p):
    arr = [];

    # assign unit weights as many as points
    for i in range (0, len (p)):
        arr.append (1);

    return arr;

def computeObjective (p, s, z):
    obFuncMin = 0;
    
    dist_matrix = {};
    for _s in s:
        dist_matrix[_s]={}
        for _p in p:
            d = euclidean (_s,_p);
            dist_matrix[_s][_p] = d                  
    
    sum_distance={}
    for c in dist_matrix:
        for point in dist_matrix[c]:
            if point in sum_distance:
                sum_distance[point] += dist_matrix[c][point]
            else:
                sum_distance[point] = dist_matrix[c][point]


    # find the min dist. from each cluster and do not include the zLargest points  
    clusters={}
    for l in p:
        min_dist=0
        for c in dist_matrix:
            if not (c in clusters.keys()):
                clusters[c]={}
            if (l in s):
                continue
            elif (min_dist==0 or min_dist>dist_matrix[c][l] ):
                min_dist = dist_matrix[c][l] 
                min_dist_key = l
                min_dist_c_key = c
        if not(l in s):
            clusters[min_dist_c_key][min_dist_key]=min_dist
            
    for i in range(z):
        prev =0
        for c in clusters:
            max_key = max(clusters[c].items(), key=operator.itemgetter(1))[0]
            max_value = clusters[c][max_key]
            if(max_value>prev):
                for _s in dist_matrix :
                    if max_value>dist_matrix[_s][max_key]:
                        break
                    else:            
                        z_key = max_key
                        z_cluster_key=c
            prev=max_value
            
        clusters[z_cluster_key].pop(z_key)
                
                
            
    maxDists = []
    for c in clusters:
        for v in clusters[c]:
           max_key = max(clusters[c].items(), key=operator.itemgetter(1))[0]
        maxDists.append(clusters[c][max_key])
        
    obFuncMin = max(maxDists)
    return obFuncMin;

if __name__ == '__main__':
    startTime  = time.time ();
    
    # constants
    fileName    = sys.argv[1];
    points      = readVectorsSeq (fileName);
    k           = 3;
    z           = 1;
    alpha       = 0;
    unitWeights = [];

    print ('Input size n = ' +         str (len (points)));
    print ('Number of centers k = ' +  str (k));
    print ('Number of outliers z = ' + str (z));
    
    # assign unit weights
    unitWeights = initializeUnitWeights (points);

    res         = seqWeightedOutliers (points, k, z, alpha, unitWeights);

    print ('Final guess = ' +       str (res[1]));
    print ('Number of guesses = ' + str (res[2]));

    # minimise the max distance between centres and associated points within its radius
    objectiveMin = computeObjective (points, res[0], z);

    print ('Objective function = ' + str (objectiveMin));
    print ('Time of seqWeightedOutliers = ' + str (time.time () - startTime));
