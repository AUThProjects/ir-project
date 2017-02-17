# ir-project

This is a repository for the project in the course of Information Retrieval, of 7th semester.

## How to run

There are multiple different paths to take:

1. With/Without dictionary
2. With/Without Word2Vec dimensionality reduction

### Running the dictionary alterations

The DictPreprocessing class is responsible for appending values in the raw input files. For this reason:

```
1. Copy the data folder into src/main/resources directory
2. Run the DictPreprocessing class
    java -cp src/main/java/gr/auth/csd/IRProject/DictPreprocessing.java DictPreprocessing
3. Run the preprocessing python script
    python src/preprocessing_{,un}labelled.py
4. Run the TF-IDF analysis from IRProjectApp
```

### IRProjectApp

In any case, before running the IRProjectApp, 
you must already have run the preprocessing python scripts to produce the json wrapped input file from the raw files distributed in the assignment.
(step 3 from above)

```
spark-submit --class gr.auth.csd.IRProject.IRProjectApp ./target/scala-2.11/ir-project-assembly-1.0.jar
```

### Running the Word2Vec dimensionality Reduction

After the vector model (TF-IDF) analysis, you can run the Word2Vec class in order to reduce the available features.

```
spark-submit --class gr.auth.csd.IRProject.transform.Word2Vector ./target/scala-2.11/ir-project-assembly-1.0.jar
```

Wherever you need the data with the reduced dimensions, you must change the input from `src/main/resources/tfIdfData.json` to `src/main/resources/w2vData` and the set the input 
column from features to `w2vRes`.


### Supervised Algorithms

#### Logistic Regression

```
spark-submit --class gr.auth.csd.IRProject.supervised.LRAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```

#### SVM

```
spark-submit --class gr.auth.csd.IRProject.supervised.SVMAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```


### Unsupervised Algorithms

#### K-means

```
spark-submit --class gr.auth.csd.IRProject.unsupervised.KMeansAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```


#### Latent Dirichlet Allocation

```
spark-submit --class gr.auth.csd.IRProject.unsupervised.LDAAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```

#### Gaussian Mixture

```
spark-submit --class gr.auth.csd.IRProject.unsupervised.GaussianMixtureAlgorithm ./target/scala-2.11/ir-project-assembly-1.0.jar
```

## Results

The results of our predictions are located under `results/predictions.txt`.
