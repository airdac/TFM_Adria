## do.idmap Interactive Document Map documentation example

library(Rdimtools)

## load iris data
data(iris)
set.seed(100)
subid <- sample(1:150, 50)
x <- as.matrix(iris[subid, 1:4])
lab <- as.factor(iris[subid, 5])
## let's compare with other methods
out1 <- do.pca(x, ndim = 2)
out2 <- do.lda(x, ndim = 2, label = lab)
out3 <- do.idmap(x, ndim = 2, engine = "NNP")
## visualize
opar <- par(no.readonly = TRUE)
par(mfrow = c(1, 3))
plot(out1$Y, pch = 19, col = lab, main = "PCA")
plot(out2$Y, pch = 19, col = lab, main = "LDA")
plot(out3$Y, pch = 19, col = lab, main = "IDMAP")
par(opar)