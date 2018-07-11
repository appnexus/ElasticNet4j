package com.appnexus.opt.examples;

import com.appnexus.opt.ml.*;

import java.util.List;

public class TrainingExample {

    private static final long COL_SEED = 8;
    private static final long BETA_SEED = 16;
    private static final long DATA_SEED = 32;
    private static final long WEIGHT_SEED = 64;
    private static final double SPARSE_PCT = 0.1;
    private static final double TOLERANCE = 1e-6;

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
        double[] lambdaScaleFactors = LRUtil.generateLambdaScaleFactors(observations, numOfFeatures);
        double[] initialBetas = Utils.createBetas(numOfFeatures + 1, BETA_SEED);
        // train
        LR lr = new LR(observations, numOfFeatures, initialBetas, alpha, lambdaGrid, lambdaScaleFactors, TOLERANCE,
            maxIterations, new CoordinateDescentTrainer());
        // results
        List<LRResult> lrResults = lr.calculateBetas(false);
        for (LRResult lrResult : lrResults) {
            System.out.println(lrResult);
        }
    }

    /*
        TEST CODE
     */

    public static void main(String[] args) {
        runLR();
    }

}
