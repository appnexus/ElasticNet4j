package com.appnexus.opt.ml;

import java.util.Random;

public class LRTestUtils {
    // private static final double SPARCE_PCT = 0.2;
    // private static final double TOLERANCE = 1e-6;
    // private static final long COL_SEED = 8;
    // private static final long BETA_SEED = 16;
    // private static final long DATA_SEED = 32;
    // private static final long WEIGHT_SEED = 64;
    private static final double DEFAULT_BETA_0 = -3;
    private static final double BETA_MAX = 1.5; // -1.5 to 1.5

    public static SparseObservation[] createTestData(int numOfObservations, int numOfFeatures, double sparsePct,
        long colSeed, long betaSeed, long dataSeed, long weightSeed) {
        int numOfNonZeroFeatures = (int) (sparsePct * numOfFeatures);

        // Create feature vectors
        SparseArray[] featureVecs = new SparseArray[numOfObservations];
        for (int i = 0; i < numOfObservations; ++i) {
            featureVecs[i] = new SparseArray();
        }
        Random colRn = new Random(colSeed);
        Random dataRn = new Random(dataSeed);
        for (int i = 0; i < numOfObservations; ++i) {
            for (int j = 0; j < numOfNonZeroFeatures; ++j) {
                featureVecs[i].set(colRn.nextInt(numOfFeatures), dataRn.nextGaussian());
            }
        }

        // Make betas
        double[] betasWithBeta0 = makeBetas(numOfFeatures + 1, betaSeed);

        // Make observations
        Random weightRn = new Random(weightSeed);
        SparseObservation[] obs = new SparseObservation[numOfObservations];
        for (int i = 0; i < numOfObservations; ++i) {
            int weight = 50 + weightRn.nextInt(50);
            double y = weight * LRUtil.expit(LRUtil.betasDotXi(featureVecs[i], betasWithBeta0));
            obs[i] = new SparseObservation(featureVecs[i], y, weight);
        }
        return obs;
    }

    public static double[] makeBetas(int numOfBetasWithBeta0, long betaSeed) {
        Random rn = new Random(betaSeed);
        double[] betas = new double[numOfBetasWithBeta0];
        for (int i = 1; i < numOfBetasWithBeta0; ++i) {
            betas[i] = (rn.nextDouble() - 0.5D) + BETA_MAX;
        }
        betas[0] = DEFAULT_BETA_0;
        return betas;
    }
}
