# Tomato Juice
## What is **Tomato Juice**?
A simple and efficient Machine Learning library for training Logistic Regression models on sparse data with rare events. The library uses Coordinate Descent for model training based on the Stanford University paper [Regularization Paths for Generalized Linear Models
via Coordinate Descent](http://web.stanford.edu/%7Ehastie/Papers/glmnet.pdf).

## Usage
The algorithm uses a data-structure called `SparseObservation` to hold the sparse feature vectors. The features themselves are stored in a `SparseArray` for minimizing the memory footprint
```java
SparseArray x = new SparseArray(); // Feature array. Independent Variables
x.set(0, 1.5); // Set value 1.5 at index 0
x.set(10, 2.0); // Set value 2.0 at index 10
x.set(50, 1.234); // Set value 1.234 at index 50

double y = 0.56 // Observation or Dependent Variable
int weight = 19 // Number of occurrences of this feature vector in the data set 

SparseObservation featureVector = new SparseObservation(x, y, weight); // Sparse feature vector
```

Once the data set is read into the `SparseObservation`s we can begin training models
```java
SparseObservation[] featureVectorsForTraining = ...; // Training dataset
int numOfFeatures = 100; // Total Number of distinct features in your data set
double[] initialBetas = null; // Initial model to start training with. It can be set to null or one can provide a starting point model to begin training
double alpha = 1; // Elastic-net Parameter. Use 1 for L1 Regularization, 0 for L2 Regularization. A value between 0 and 1 corresponds to a combination of L1 and L2
double[] lambdaGrid = ...; // Array of Regularization Parameters to train models. This can be generated using LRUtil.getLambdaGrid(int size, double start, double end);
double[] lambdaScaleFactors = ... // Array of scale factors to scale the regularization parameter per feature based on its frequency in the dataset. This can be generated using LRUtil.generateLambdaScaleFactors(SparseObservation[] featureVectorsForTraining, int featureVectorLen)
double tolerance = 1.0E-6; // Max tolerable convergence error between 2 iterations to stop the training process
int maxIterations = 400; // Max number of training iterations

LR lr = new LR(featureVectorsForTraining, numOfFeatures, initialBetas, alpha, lambdaGrid, lambdaScaleFactors, tolerance, maxIterations, new CoordinateDescentTrainer()); // Initialize the LR algorithm with a CoordinateDescentTrainer

boolean warmStart = false; // The training algorithm will train a model for each lambda in the lambdaGrid. Setting this to true will use the model generated for the previous lambda to warm-start training for the next lambda
List<LRResult> lrResults = lr.calculateBetas(warmStart); // Train models for each lambda in the lambda grid
```

## Examples
For a guided walk-through of how to use the library, you can check out the examples package. Within the examples package, you can find two examples (`LogisticRegressionWithGeneratedData.java` and `LogisticRegressionWithDataFromFile.java`). 

### `LogisticRegressionWithGeneratedData.java`
This example contains a full-run through of our logistic regression algorithm including
* setting up sample parameters for training including input data
* training (logistic regression)
* evaluating the performance of the training results on test data

### `LogisticRegressionWithDataFromFile.java`
This example is similar to the one above only it reads data from a file (`observations.tsv`) rather than generating data. 

## License
This project is licensed under the Apache License, Version 2.0 - see LICENSE for details.