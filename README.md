# Tomato Juice
## What is **Tomato Juice**?
A simple and efficient Machine Learning library for training Logistic Regression models on sparse data with rare events. The library uses Coordinate Descent for model training based on the GLMNET paper.

## Usage
The algorithm uses a data-structure called `SparseObservation` to hold the sparse feature vectors. The features themselves are stored in a `SparseArray` for minimizing the memory footprint
> SparseArray x = new SparseArray(); // Feature array. Independent Variables
> x.set(0, 1.5); // Set value 1.5 at index 0
> x.set(10, 2.0); // Set value 2.0 at index 10
> x.set(50, 1.234); // Set value 1.234 at index 50
>
> double y = 0.56 // Observation or Dependent Variable
> int weight = 19 // Number of occurrences of this feature vector in the data set 
>
> SparseObservation featureVector = new SparseObservation(x, y, weight); // Sparse feature vector

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