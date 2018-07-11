package com.appnexus.opt.examples;

import com.appnexus.opt.ml.LREvalUtil;
import com.appnexus.opt.ml.LRUtil;
import com.appnexus.opt.ml.SparseArray;
import com.appnexus.opt.ml.SparseObservation;

import java.util.Random;

public class Utils {

    public static double[] createBetas(int numOfBetasWithBeta0, long betaSeed) {
        // TODO, VVAL-219: fill in
        Random rn = new Random(betaSeed);
        double[] betas = new double[numOfBetasWithBeta0];
        return betas;
    }

    // TODO, VVAL-219: complete
    public static SparseObservation[] createTestData(int numOfObservations, int numOfFeatures, double sparsePct, long colSeed,
        long betaSeed, long dataSeed, long weightSeed) {

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

        Random weightRn = new Random(weightSeed);
        SparseObservation[] obs = new SparseObservation[numOfObservations];
        for (int i = 0; i < numOfObservations; ++i) {
            int weight = 50 + weightRn.nextInt(50);
            double y = weight * LRUtil.expit(LRUtil.betasDotXi(featureVectors[i], betasWithBeta0));
            obs[i] = new SparseObservation(featureVectors[i], y, weight);
            // TODO, VVAL-219: remove eventually
            System.out.println(obs[i]);
        }
        return obs;
    }

    // TODO, VVAL-219: remove eventually
    public static void main(String[] args) {
        createTestData(10, 5, 0.5, 99, 99, 99, 99);
    }

}
