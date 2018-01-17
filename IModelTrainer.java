package com.appnexus.opt.ml;

public interface IModelTrainer {

    /**
     * Iterate through the feature vectors once and train a set of Betas
     * 
     * @param observations Contain feature vectors without intercept term for beta0, weight of the feature vector and dependent variable
     * @param totalWeights Sum of all weights / total number of trials
     * @param oldBetasWithBeta0 Betas from the last iteration
     * @param alpha elastic-net parameter 1 -> L1, 0 -> L2
     * @param lambda regularization parameter
     * @param lambdaScaleFactors scale factors for different regularization on different predictors
     * @param tolerance max error between successive iterations
     * @param maxIterations max iterations
     * @return new Betas trained for this iteration
     */
    public LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights, double[] oldBetasWithBeta0, double alpha, double lambda, double[] lambdaScaleFactors, double tolerance, int maxIterations);
}
