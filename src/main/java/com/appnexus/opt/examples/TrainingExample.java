package com.appnexus.opt.examples;

import com.appnexus.opt.ml.SparseObservation;

public class TrainingExample {

    private static final long COL_SEED = 8;
    private static final long BETA_SEED = 16;
    private static final long DATA_SEED = 32;
    private static final long WEIGHT_SEED = 64;

    // TODO, VVAL-219: doc string
    public static void runCoordinateDescentTrainer() {
        int numOfObservations = 1000;
        int numOfFeatures = 200;
        double sparsePct = 0.1;
        SparseObservation[] observations = Utils.createTestData(numOfObservations, numOfFeatures, sparsePct, COL_SEED, BETA_SEED, DATA_SEED, WEIGHT_SEED);
    }

}
