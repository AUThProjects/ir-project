# ir-project

This is a repository for the project in the course of Information Retrieval, of 7th semester.

## How to run

### IRProjectApp

```
spark-submit --class gr.auth.csd.IRProject.IRProjectApp ./target/scala-2.11/ir-project-assembly-1.0.jar
```

### Supervised Algorithms

#### Logistic Regression

```
spark-submit --class gr.auth.csd.IRProject.supervised.LRAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```

### Unsupervised Algorithms

#### K-means

```
spark-submit --class gr.auth.csd.IRProject.unsupervised.KMeansAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```
