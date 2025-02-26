/* Folyd's algorithm */
%module floyd
%inline %{
double (*new_array())[10000] {
    return (double (*)[10000]) malloc(10000*10000*sizeof(double));
}
double array_get(double (*ar)[10000], int index1, int index2) {
    return ar[index1][index2];
}
double array_floyd(double (*ar)[10000], int asize) {
    return floyd(ar, asize);
}
int floyd(double (*ar)[10000], int asize);
void array_set(double (*ar)[10000], double value, int index1, int index2) {
    ar[index1][index2]=value;
}
%}
%rename(delete_array) free(void *);
int floyd(double (*ar)[10000], int asize);

