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

import java.util.Arrays;
import java.util.LinkedList;

/**
 * This class is the entry point into using the LR training functionality
 */
public class LR {
    private final SparseObservation[] observations; // contains x, y and weight but does not contain entries for beta0
    private final int numOfFeatures;
    private final double totalSuccesses;
    private final double totalWeights;
    private final double[] initialBetasWithBeta0; // contains beta0 in first index
    private final double alpha;
    private final double[] lambdaGrid;
    private final double[] lambdaScaleFactors;
    private final double tolerance;
    private final int maxIterations;
    private final IModelTrainer modelTrainer;

    public LR(SparseObservation[] observations, int numOfFeatures, double[] initialBetasWithBeta0, double alpha,
        double[] lambdaGrid, double[] lambdaScaleFactors, double tolerance, int maxIterations,
        IModelTrainer modelTrainer) {
        this.observations = observations;
        this.numOfFeatures = numOfFeatures;
        this.totalSuccesses = getTotalSuccesses(this.observations);
        this.totalWeights = getTotalWeights(this.observations);
        this.initialBetasWithBeta0 = getInitialBetasWithBeta0(initialBetasWithBeta0);
        this.alpha = alpha;
        this.lambdaGrid = lambdaGrid;
        this.lambdaScaleFactors = lambdaScaleFactors;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.modelTrainer = modelTrainer;
    }

    /**
     * @param observations observations
     * @return total success across observations
     */
    static double getTotalSuccesses(SparseObservation[] observations) {
        double totalSuccesses = 0;
        for (SparseObservation obs : observations) {
            totalSuccesses += obs.getY();
        }
        return totalSuccesses;
    }

    /**
     * @param observations observations
     * @return total weights across observations
     */
    static double getTotalWeights(SparseObservation[] observations) {
        double totalWeights = 0;
        for (SparseObservation obs : observations) {
            totalWeights += obs.getWeight();
        }
        return totalWeights;
    }

    /**
     * @param initialBetasWithBeta0 initial betas with beta0
     * @return initial betas with beta0 (either guessed or unchanged)
     */
    double[] getInitialBetasWithBeta0(double[] initialBetasWithBeta0) {
        double betasWithBeta0[] = initialBetasWithBeta0;
        if (betasWithBeta0 == null) {
            betasWithBeta0 = new double[this.numOfFeatures + 1];
            betasWithBeta0[0] = guessInitialBetaZero();
        }
        return betasWithBeta0;
    }

    /**
     * @return guessed initial beta0
     */
    double guessInitialBetaZero() {
        double globalCtr = this.totalSuccesses / this.totalWeights;
        return Math.log(globalCtr / (1 - globalCtr));
    }

    /*
     * helper methods
     */

    /**
     * @param warmStart warm start flag
     * @return beta results across lambda grid
     */
    public LinkedList<LRResult> calculateBetas(boolean warmStart) {
        LinkedList<LRResult> lrResultList = new LinkedList<>();
        LRResult lrResult = null;
        for (double lambda : this.lambdaGrid) {
            double[] startBetasWithBeta0 = ((warmStart && lrResult != null) ?
                Arrays.copyOf(lrResult.getBetasWithBeta0(), lrResult.getBetasWithBeta0().length) :
                Arrays.copyOf(this.initialBetasWithBeta0, this.initialBetasWithBeta0.length));
            lrResult = calculateBetas(startBetasWithBeta0, lambda);
            lrResultList.add(lrResult);
        }
        return lrResultList;
    }

    /**
     * @param startBetasWithBeta0 initial betas
     * @param lambda              lambda
     * @return trained betas
     */
    public LRResult calculateBetas(double[] startBetasWithBeta0, double lambda) {
        return this.modelTrainer
            .trainNewBetasWithBeta0(this.observations, this.totalWeights, startBetasWithBeta0, this.alpha, lambda,
                this.lambdaScaleFactors, tolerance, maxIterations);
    }

}
