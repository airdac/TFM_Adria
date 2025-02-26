/* Floyd's algorithm */
int floyd(double (*ar)[10000], int asize) {
  int i;
  int j;
  int k;
  for (k=0; k<asize; k++) {
    for (i=0; i<asize; i++) {
      for (j=0; j<asize; j++) {
        if (ar[i][j] > ar[i][k] + ar[k][j]) {
          ar[i][j] = ar[i][k] + ar[k][j];
        }
      }
    }
  }
  return 0;
}
