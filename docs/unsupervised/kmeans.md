# K-means

## Algorithm

* Use 10-fold cross validation. (split manually)

```
for(i=0;i<10;++i)
    1. Split dataset in 90%-10% randomly
    2. Train 90%
    3. Test 10%
        3.1 Identify clusters
            Take one cluster and find majority of labels
        3.2 Evaluate accuracy
        
Pick model with highest accuracy.
```
