package com.appnexus.opt.examples;

import com.appnexus.opt.ml.LRUtil;
import com.appnexus.opt.ml.SparseArray;
import com.appnexus.opt.ml.SparseObservation;

import java.util.Random;

public class Utils {

    private static final double DEFAULT_BETA_0 = -3;
    private static final double BETA_MAX = 1.5;

    /**
     * @param numOfBetasWithBeta0 number of betas with beta 0
     * @param betaSeed            beta seed
     * @return beta array
     */
    public static double[] createBetas(int numOfBetasWithBeta0, long betaSeed) {
        Random rn = new Random(betaSeed);
        double[] betas = new double[numOfBetasWithBeta0];
        betas[0] = DEFAULT_BETA_0;
        for (int i = 1; i < numOfBetasWithBeta0; ++i) {
            betas[i] = (rn.nextDouble() - 0.5D) + BETA_MAX;
        }
        return betas;
    }

    /**
     * @param numOfObservations number of observations
     * @param numOfFeatures     number of features
     * @param sparsePct         sparse percentage (determines number of nonzero features)
     * @param colSeed           column seed
     * @param betaSeed          beta seed
     * @param dataSeed          data seed
     * @param weightSeed        weight seed
     * @return generated test data
     */
    public static SparseObservation[] createTestData(int numOfObservations, int numOfFeatures, double sparsePct,
        long colSeed, long betaSeed, long dataSeed, long weightSeed) {

        // feature vectors
        SparseArray[] featureVectors = new SparseArray[numOfObservations];
        for (int i = 0; i < numOfObservations; ++i) {
            featureVectors[i] = new SparseArray();
        }
        Random colRn = new Random(colSeed);
        Random dataRn = new Random(dataSeed);
        // set number of nonzero features on each sparse observation based on desired sparse percentage
        int numOfNonZeroFeatures = (int) (sparsePct * numOfFeatures);
        for (int i = 0; i < numOfObservations; ++i) {
            for (int j = 0; j < numOfNonZeroFeatures; ++j) {
                featureVectors[i].set(colRn.nextInt(numOfFeatures), dataRn.nextGaussian());
            }
        }

        // make betas
        double[] betasWithBeta0 = createBetas(numOfFeatures + 1, betaSeed);

        // make observation
        Random weightRn = new Random(weightSeed);
        SparseObservation[] obs = new SparseObservation[numOfObservations];
        for (int i = 0; i < numOfObservations; ++i) {
            int weight = 50 + weightRn.nextInt(50);
            double y = weight * LRUtil.expit(LRUtil.betasDotXi(featureVectors[i], betasWithBeta0));
            obs[i] = new SparseObservation(featureVectors[i], y, weight);
        }
        return obs;
    }

    /*
        TEST CODE
     */

    public static void main(String[] args) {
        SparseObservation[] testObservations = createTestData(10, 5, 0.5, 99, 99, 99, 99);
        for (SparseObservation observation : testObservations) {
            System.out.println(observation);
        }
    }

}
