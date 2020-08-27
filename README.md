# ElasticNet4j: A Performant Elastic Net Logistic Regression Solver in Java
## What is **ElasticNet4j**?
A simple and efficient Machine Learning library for training Logistic Regression models on sparse data with rare events. The library uses Coordinate Descent for model training based on the Stanford University paper [Regularization Paths for Generalized Linear Models
via Coordinate Descent](http://web.stanford.edu/%7Ehastie/Papers/glmnet.pdf).

## Getting Started
We support building with [Maven](https://maven.apache.org/)

## Building Locally
```
$ mvn clean install
```

## Running Unit Tests
```
$ mvn test
```

## Usage
The algorithm uses a data-structure called `SparseObservation` to hold the sparse feature vectors. The features themselves are stored in a `SparseArray` for minimizing the memory footprint
```java
SparseArray x = new SparseArray(); // Feature array. Independent Variables
x.set(0, 1.5); // Set value 1.5 at index 0
x.set(10, 2.0); // Set value 2.0 at index 10
x.set(50, 1.234); // Set value 1.234 at index 50

double y = 2.0 // Number of success events of this feature vector in the data set
int weight = 1023 // Number of trials or observations of this feature vector in the data set

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
double tolerance = 1.0E-6; // Convergence criteria. If all the beta parameters change less than this value on the current iteration, stop further training iterations
int maxIterations = 400; // Max number of training iterations

LR lr = new LR(featureVectorsForTraining, numOfFeatures, initialBetas, alpha, lambdaGrid, lambdaScaleFactors, tolerance, maxIterations, new CoordinateDescentTrainer()); // Initialize the LR algorithm with a CoordinateDescentTrainer

boolean warmStart = true; // The training algorithm will train a model for each lambda in the lambdaGrid. Setting this to true will use the model generated for the previous lambda to warm-start training for the next lambda
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

## Contributors
* [Chinmay Nerurkar](https://github.com/nchinmay) [:e-mail:](mailto:nchinmay@hotmail.com)
* [Noah Stebbins](https://github.com/nstebbins) [:e-mail:](mailto:nstebbins1@gmail.com)
* [Tian Yu](https://github.com/ty277) [:e-mail:](mailto:ty277@cornell.edu)
* [Lei Hu](https://github.com/interboys11) [:e-mail:](mailto:lei.stone.hu@gmail.com)
* [Yana Volkovich](https://github.com/volkovich) [:link:](https://www.yanavolkovich.com)
* [Abraham Greenstein](https://github.com/agreens) [:e-mail:](mailto:abraham.greenstein@gmail.com)
* [Moussa Taifi](https://github.com/moutai) [:e-mail:](mailto:moussa.taifi@outlook.com) [:link:](https://www.moussataifi.com)

## License
This project is licensed under the Apache License, Version 2.0 - see [LICENSE](LICENSE) for details.
