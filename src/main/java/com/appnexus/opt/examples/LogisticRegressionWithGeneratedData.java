/*
 *    Copyright 2018 APPNEXUS INC
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.appnexus.opt.examples;

import com.appnexus.opt.ml.*;

import java.util.Arrays;
import java.util.List;

/*
 * This file demonstrates an example use of the library by generating random training and testing data,
 * training a Logistic Regression model using the training set and evaluating its performance on the testing set.
 */
public class LogisticRegressionWithGeneratedData {

    private static final long COL_SEED = 8;         // random seed for column (feature) generation
    private static final long BETA_SEED = 16;       // random seed for beta generation
    private static final long DATA_SEED = 32;       // random seed for data (feature value) generation
    private static final long WEIGHT_SEED = 64;     // random seed for observation weight generation
    private static final double SPARSE_PCT = 0.1;   // desired percentage of nonzero features
    private static final double TOLERANCE = 1e-6;   // tolerance of training algorithm
    private static final double TRAINING_PCT = 0.9; // desired percentage of data for training algorithm

    /**
     * Example of how to use our logistic regression library for training and testing data
     */
    public static void runLogisticRegressionExample() {
        /* Below is metadata about the data going into the algorithm as well as metadata for the training algorithm itself. */
        int numOfObservations = 1000;   // total desired number of data observations
        int numOfFeatures = 200;        // total desired number of features
        double alpha = 1.0;             // elastic net parameter for training
        int maxIterations = 1000;       // maximum number of iterations for training
        int lambdaSize = 1;             // number of lambda tuning parameters for training
        int lambdaStart = 1;            // start for lambda tuning parameters for training
        int lambdaEnd = 17;             // end for lambda tuning parameters for training
        /* Generate observations using the metadata above. */
        SparseObservation[] observations = ExampleUtils
            .createTestData(numOfObservations, numOfFeatures, SPARSE_PCT, COL_SEED, BETA_SEED, DATA_SEED, WEIGHT_SEED);
        /* Generate a grid of lambda tuning parameters using the metadata above. */
        double[] lambdaGrid = LRUtil.getLambdaGrid(lambdaSize, lambdaStart, lambdaEnd);
        /* Use TRAINING_PCT percentage of data as training data for our algorithm. The remainder is data to test our algorithm. */
        int splitIdx = (int) (TRAINING_PCT * observations.length);
        SparseObservation[] trainObservations = Arrays.copyOfRange(observations, 0, splitIdx);
        SparseObservation[] testObservations = Arrays.copyOfRange(observations, splitIdx, observations.length);
        /* Generate lambda scale factors for the training algorithm. */
        double[] lambdaScaleFactors = LRUtil.generateLambdaScaleFactors(trainObservations, numOfFeatures);
        /* Generate an initial beta weight vector that will be updated by the training algorithm per iteration. */
        double[] initialBetas = ExampleUtils.createBetas(numOfFeatures + 1, BETA_SEED);
        /* Train! */
        LR lr = new LR(trainObservations, numOfFeatures, initialBetas, alpha, lambdaGrid, lambdaScaleFactors, TOLERANCE,
            maxIterations, new CoordinateDescentTrainer());
        /* Visualize the results of the training algorithm and calculate entropy for the test data. */
        List<LRResult> lrResults = lr.calculateBetas(false);
        for (LRResult lrResult : lrResults) {
            System.out.println("LR result: " + lrResult);
            System.out.println(
                "entropy for test data: " + LREvalUtil.getEntropy(testObservations, lrResult.getBetasWithBeta0()));
            System.out.println("normalized entropy for test data: " + LREvalUtil
                .getEntropyNormalized(testObservations, lrResult.getBetasWithBeta0()));
        }
    }

    /*
        TEST CODE
     */

    public static void main(String[] args) {
        runLogisticRegressionExample();
    }

}
