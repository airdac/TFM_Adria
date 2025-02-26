# pysomap
import pysomap
import numpy

data = open("pysomap_example_data", "r").readlines()
smatrix = []
for line in data:
  sline = str.split(line)
  aline = []
  for item in sline:
    aline.append(float(item))
  smatrix.append(aline)
M = numpy.array(smatrix)

A = pysomap.isodata()                                 # creates a new object A
# loads python array X (N x M) into object A
A.load_isodata(M)
# performs dimensionality reduction of A
A.reduce_isodata(isomap_type="e", e=0.5, O=2)
#   isomap_type is "e" or "K" for ε-
#      or K-isomap, respectively
#   epsilon or K value must be set
#   O is output dimensionality

results = open("results", "w")
for line in A.outdata():
  for item in line:
    results.write("  %f" % item)
  results.write("\n")
results.close()
