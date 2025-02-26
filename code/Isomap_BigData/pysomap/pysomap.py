#!/bin/env python
# Pysomape - python implementation of epsilon-isomap

import sys, math, time
MAX = 9999999999.

# Try to import numpy
try:
  import numpy.linalg
except:
  print "numpy must be installed"
  sys.exit(2)

# Try to import floyd
try:
  from floyd import *
except:
  print "Error: Can't load Floyd's algorithm library"
  sys.exit(2)

class isodata:
  # loading input data
  def load_isodata(self, indata):
    try:
      self.indata = numpy.array(indata)
    except:
      print "Error: wrong format of input data"
      sys.exit(2)
    self.N, self.M = numpy.shape(self.indata)

  # optionally you can load directly distance matrix
  def load_dismat(self, dismat):
    try:
      self.dismat = numpy.array(dismat)
    except:
      print "Error: wrong format of distance matrix"
      sys.exit(2)
    N1, N2 = numpy.shape(self.dismat)
    if N1 != N2:
      print "Error: distance matrix must be square"
      sys.exit(2)
    self.N = N1

  # Calculating distance matrix
  def distance_isodata(self):
    current_time = time.clock()
    if self.verbose=="v":
      print "Calculating distance matrix ............ ",
    dismat = numpy.zeros((self.N, self.N), "float")
    for i in range(self.N):
      for j in range(i+1, self.N):
        difference = self.indata[i]-self.indata[j]
        dismat[i][j] = math.sqrt(numpy.dot(difference, difference))
        dismat[j][i] = dismat[i][j]
    self.dismat=dismat
    if self.verbose=="v":
      print "%6.2f s" % (time.clock()-current_time)

    # Calculating graph matrix
  def graph_isodata(self):
    current_time = time.clock()
    if self.verbose=="v":
      print "Calculating graph matrix ............... ",
    graph = numpy.zeros((self.N, self.N), "float")
    for i in range(self.N):
      for j in range(self.N):
        graph[i][j] = MAX

    if self.isomap_type == "e":
      contr = numpy.zeros(self.N, "int")
      for i in range(self.N):
        for j in range(i, self.N):
          if (self.dismat[i][j] < self.e):
            graph[i][j] = self.dismat[i][j]
            contr[i] = contr[i] + 1
            contr[j] = contr[j] + 1
          graph[j][i] = graph[i][j]
      if numpy.min(contr) < 3:
        print "Error: epsilon is too low"
        sys.exit(2)

    if self.isomap_type == "K":
      for i in range(self.N):
        indexes = []
        for j in range(self.N):
          indexes.append([self.dismat[i][j], j])
        for Kindex in range(self.K+1):
          minlen = MAX
          nindexes = []
          for line in indexes:
            if (line[0] <= minlen):
              minlen = line[0]
              ki = line[1]
          for line in indexes:
            if (line[1] == ki):
              graph[i][ki] = self.dismat[i][ki]
            else:
              nindexes.append(line)
          indexes = nindexes
      for i in range(self.N):
        for j in range(self.N):
          if (graph[i][j] < graph[j][i]):
            graph[j][i] = graph[i][j]
    self.graph = graph
    if self.verbose=="v":
      print "%6.2f s" % (time.clock()-current_time)

    # Calculating the shortest path using Floyd's algorithm
  def path_isodata(self):
    current_time = time.clock()
    if self.verbose=="v":
      print "Calculating the shortest path matrix ... ",
    path_to_floyd = new_array()
    for i in range(self.N):
      for j in range(self.N):
        array_set(path_to_floyd, self.graph[i][j], i, j)
    array_floyd(path_to_floyd, self.N)
    spath=[]
    for i in range(self.N):
      spathline=[]
      for j in range(self.N):
        spathline.append(array_get(path_to_floyd, i, j))
      spath.append(spathline)
    self.path=numpy.array(spath)
    if self.verbose=="v":
      print "%6.2f s" % (time.clock()-current_time)

    # Calculating matrix A and B
  def mds_isodata(self):
    current_time = time.clock()
    if self.verbose=="v":
      print "Multidimensionally scalling ............ ",
    matrixA = numpy.zeros((self.N, self.N), "float")
    for i in range(self.N):
      for j in range(self.N):
        matrixA[i][j] = - self.path[i][j] * self.path[i][j] / 2
    a1 = numpy.zeros(self.N, "float")
    for i in range(self.N):
      a1[i] = 0.0
      for j in range(self.N):
        a1[i] = a1[i] + matrixA[i][j]/self.N
    a2 = 0.0
    for i in range(self.N):
      for j in range(self.N):
        a2 = a2 + matrixA[i][j]/(self.N * self.N)
    matrixB = numpy.zeros((self.N, self.N), "float")
    for i in range(self.N):
      for j in range(self.N):
        matrixB[i][j] = matrixA[i][j] - a1[i] - a1[j] + a2
    eigenvals, eigenvecs = numpy.linalg.eig(matrixB)
    self.outdata = numpy.zeros((self.N, self.O), "float")
    for i in range(self.N):
      for j in range(self.O):
        self.outdata[i][j] = eigenvecs[i][j] * math.sqrt(eigenvals[j])
    if self.verbose=="v":
      print "%6.2f s" % (time.clock()-current_time)

  # dimensionality reduction of input data
  def reduce_isodata(self, isomap_type="K", K=5, e=0.5, O=2, verbose="v"):
    begining = time.clock()
    self.verbose = verbose
    self.isomap_type = isomap_type
    self.K = K
    self.e = e
    self.O = O
    self.distance_isodata()
    if type(O) != int:
      print "Error: O must be integer"
      sys.exit(2)
    if O > self.N:
      print "Error: O must be smaller or equal to N"
      sys.exit(2)
    if isomap_type == "K":
      if type(K) != int:
        print "Error: K must be integer"
        sys.exit(2)
      if verbose=="v":
        print "Using K-isomap, K = %i, calculating %i-dimensional embedding" % (K, O)
    elif isomap_type == "e":
      if type(e) != float:
        print "Error: epsilon must be float"
        sys.exit(2)
      if verbose=="v":
        print "Using epsilon-isomap, epsilon = %f, calculating %i-dimensional embedding" % (e, O)
    else:
      print "Error: you can use either K-isomap (isomap_type=\"K\") or epsilon-isomap (isomap_type=\"e\")"
      sys.exit(2)
    self.graph_isodata()
    self.path_isodata()
    self.mds_isodata()
    if verbose=="v":
      print "--------------------------------------------------"
      print "Total procedure ........................ ",
      print "%6.2f s" % (time.clock()-begining)

  # dimensionality reduction of distance matrix
  def reduce_isodata2(self, isomap_type="K", K=5, e=0.5, O=2, verbose="v"):
    begining = time.clock()
    self.verbose = verbose
    self.isomap_type = isomap_type
    self.K = K
    self.e = e
    self.O = O
    if type(O) != int:
      print "Error: O must be integer"
      sys.exit(2)
    if O > self.N:
      print "Error: O must be smaller or equal to N"
      sys.exit(2)
    if isomap_type == "K":
      if type(K) != int:
        print "Error: K must be integer"
        sys.exit(2)
      if verbose=="v":
        print "Using K-isomap, K = %i, calculating %i-dimensional embedding" % (K, O)
    elif isomap_type == "e":
      if type(e) != float:
        print "Error: epsilon must be float"
        sys.exit(2)
      if verbose=="v":
        print "Using epsilon-isomap, epsilon = %f, calculating %i-dimensional embedding" % (e, O)
    else:
      print "Error: you can use either K-isomap (isomap_type=\"K\") or epsilon-isomap (isomap_type=\"e\")"
      sys.exit(2)
    self.graph_isodata()
    self.path_isodata()
    self.mds_isodata()
    if verbose=="v":
      print "--------------------------------------------------"
      print "Total procedure ........................ ",
      print "%6.2f s" % (time.clock()-begining)
