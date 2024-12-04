lasso <- scan('lasso.txt', character(), quote="")
ridge <- scan('ridge.txt', character(), quote="")
poly <- scan('poly.txt', character(), quote="")
true <- read.csv2('test.csv', header=TRUE, sep=',')$Exam_Score
lasso <- as.numeric(lasso) - true
ridge <- as.numeric(ridge) - true
poly <- as.numeric(poly) - true

print('lasso vs ridge')
print('difference in means')
print(mean(lasso) - mean(ridge))
print('t test')
print(t.test(lasso, ridge, alternative = "two.sided", conf.level = 0.95)$conf.int)
print(t.test(lasso, ridge, alternative = "two.sided", conf.level = 0.95)$p.value)

print('lasso vs poly')
print('difference in means')
print(mean(lasso) - mean(poly))
print('t test')
print(t.test(lasso, poly, alternative = "two.sided", conf.level = 0.95)$conf.int)
print(t.test(lasso, poly, alternative = "two.sided", conf.level = 0.95)$p.value)


print('ridge vs poly')
print('difference in means')
print(mean(ridge) - mean(poly))
print('t test')
print(t.test(ridge, poly, alternative = "two.sided", conf.level = 0.95)$conf.int)
print(t.test(ridge, poly, alternative = "two.sided", conf.level = 0.95)$p.value)