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

package com.appnexus.opt.ml;

/**
 * This interface declares the functions that need to be defined to create a new ModelTrainer class
 */
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
    LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights, double[] oldBetasWithBeta0,
        double alpha, double lambda, double[] lambdaScaleFactors, double tolerance, int maxIterations);
}
