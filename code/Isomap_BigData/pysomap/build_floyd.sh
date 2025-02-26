#!/bin/sh
# run this script to generate library for Floyd's
# algorithm library

echo "generating input files using SWIG ..."
swig -python floyd.i

echo "compiling ..."
# change compiler if you use other than gcc
gcc -c floyd.c floyd_wrap.c -I/usr/include/python2.4 

echo "linking ..."
ld -shared floyd.o floyd_wrap.o -o _floyd.so

echo "for SELinux you must run chcon ..."
chcon -t textrel_shlib_t _floyd.so 

