#!/bin/env python

from pysomap import *

data = open("data", "r").readlines()

smatrix = []
for line in data:
  sline = str.split(line)
  aline=[]
  for item in sline:
    aline.append(float(item))
  smatrix.append(aline)

M = numpy.array(smatrix)
A = isodata()
A.load_isodata(M)
A.reduce_isodata(isomap_type="e", e=0.5, O=2)

results = open("results", "w")
for line in A.outdata:
  for item in line:
    results.write(" %f" % item)
  results.write("\n")
results.close()

