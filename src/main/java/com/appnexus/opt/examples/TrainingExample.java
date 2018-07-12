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

public class TrainingExample {

    private static final long COL_SEED = 8;
    private static final long BETA_SEED = 16;
    private static final long DATA_SEED = 32;
    private static final long WEIGHT_SEED = 64;
    private static final double SPARSE_PCT = 0.1;
    private static final double TOLERANCE = 1e-6;
    private static final double TRAINING_PCT = 0.9;

    /**
     * runs through an example of logistic regression
     */
    public static void runLR() {
        // sample parameters
        double alpha = 1.0; // sample elastic net parameter
        int maxIterations = 1000;
        int numOfObservations = 1000;
        int numOfFeatures = 200;
        // generate lambda penalties and test data
        double[] lambdaGrid = LRUtil.getLambdaGrid(1, 1, 17);
        SparseObservation[] observations = Utils
            .createTestData(numOfObservations, numOfFeatures, SPARSE_PCT, COL_SEED, BETA_SEED, DATA_SEED, WEIGHT_SEED);
        // split into train and test
        int splitIdx = (int) (TRAINING_PCT * observations.length);
        SparseObservation[] trainObservations = Arrays.copyOfRange(observations, 0, splitIdx);
        SparseObservation[] testObservations = Arrays.copyOfRange(observations, splitIdx, observations.length);
        double[] lambdaScaleFactors = LRUtil.generateLambdaScaleFactors(trainObservations, numOfFeatures);
        double[] initialBetas = Utils.createBetas(numOfFeatures + 1, BETA_SEED);
        // train
        LR lr = new LR(trainObservations, numOfFeatures, initialBetas, alpha, lambdaGrid, lambdaScaleFactors, TOLERANCE,
            maxIterations, new CoordinateDescentTrainer());
        // training results and entropy on test data
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
        runLR();
    }

}
