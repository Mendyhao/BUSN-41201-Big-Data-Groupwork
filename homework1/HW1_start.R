# ***** AMAZON REVIEWS 

# READ REVIEWS

data<-read.table("Review_subset.csv",header=TRUE)
dim(data)

# 13319 reviews
# ProductID: Amazon ASIN product code
# UserID:  id of the reviewer
# Score: numeric from 1 to 5
# Time: date of the review
# Summary: text review
# nrev: number of reviews by this user
# Length: length of the review (number of words)

# READ WORDS

words<-read.table("words.csv")
words<-words[,1]
length(words)
#1125 unique words

# READ text-word pairings file

doc_word<-read.table("word_freq.csv")
names(doc_word)<-c("Review ID","Word ID","Times Word" )
# Review ID: row of the file  Review_subset
# Word ID: index of the word
# Times Word: number of times this word occurred in the text


# We'll do 1125 univariate regressions of 
# star rating on word presence, one for each word.
# Each regression will return a p-value, and we can
# use this as an initial screen for useful words.

# Don't worry if you do not understand the code now.
# We will go over similar code in  the class in a few weeks.

# Create a sparse matrix of word presence


library(gamlr)

spm<-sparseMatrix(i=doc_word[,1],
                  j=doc_word[,2],
                  x=doc_word[,3],
                  dimnames=list(id=1:nrow(data),words=words))

dim(spm)
# 13319 reviews using 1125 words

# Create a dense matrix of word presence

P <- as.data.frame(as.matrix(spm>0))

library(parallel)

margreg <- function(p){
	fit <- lm(stars~p)
	sf <- summary(fit)
	return(sf$coef[2,4]) 
}

# The code below is an example of parallel computing
# No need to understand details now, we will discuss more later

cl <- makeCluster(detectCores())

# Pull out stars and export to cores

stars <- data$Score

clusterExport(cl,"stars") 

# Run the regressions in parallel

mrgpvals <- unlist(parLapply(cl,P,margreg))

# If parallel stuff is not working, 
# you can also just do (in serial):
# mrgpvals <- c()
# for(j in 1:1125){
# 	print(j)
# 	mrgpvals <- c(mrgpvals,margreg(P[,j]))
# }
# make sure we have names

names(mrgpvals) <- colnames(P)

# The p-values are stored in mrgpvals 

# Q1
# Plot density histogram of p-values
hist(mrgpvals, freq = FALSE, col = 'lightblue', breaks=50)

# Q2
sig_0.05 <- sum(mrgpvals < 0.05)
sig_0.01 <- sum(mrgpvals < 0.01)
print(paste('The number of significant tests at 0.05:', sig_0.05))
print(paste('The number of significant tests at 0.01:', sig_0.01))

# Q3
fdr_cut <- function(pvals, q){
  pvals <- pvals[!is.na(pvals)]
  N <- length(pvals)
  
  k <- rank(pvals, ties.method="min")
  alpha <- max(pvals[ pvals<= (q*k/N) ])
  
  return(alpha)
}

# @ 1% FDR

cutoff1 <- fdr_cut(mrgpvals,q=.01)

print(cutoff1)

print(sum(mrgpvals<=cutoff1))

## visualize the B+H FDR algorithm 
# warning: the plot can take a bit of time/memory

sig <- factor(mrgpvals<=cutoff1)

o <- order(mrgpvals)
N <- length(mrgpvals)
plot(mrgpvals[o], col=c("grey60","red")[sig[o]], pch=20, 
     ylab="p-values", xlab="tests ordered by p-value", main = 'FDR = 1%')
lines(1:N,  0.01*(1:N)/N)

# Q4
# @ 1% FDR
cutoff1 <- fdr_cut(mrgpvals,q=.01)
print(cutoff1)
print(sum(mrgpvals<=cutoff1))

# Q5
# plot the p-value distribution; notice the tiny spike near zero
hist(as.numeric(mrgpvals), main='', xlab='p-values',freq=FALSE,col="lightblue",breaks=40) # freq=FALSE gives you a density plot

# top 10 locations to investigate
names(mrgpvals)[order(mrgpvals)[1:10]]

