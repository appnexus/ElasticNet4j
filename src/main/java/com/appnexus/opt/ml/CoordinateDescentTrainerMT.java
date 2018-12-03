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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;

/**
 * This class implements a {@link IModelTrainer} that uses Coordinate Descent method to train Logistic Regression models
 */
public class CoordinateDescentTrainerMT implements IModelTrainer {
    private static final double PROB_EPSILON = 1e-15;

    private CompletionService<Boolean> completionService;
    private int numTrainingThreads;

    public CoordinateDescentTrainerMT(CompletionService<Boolean> completionService, int numTrainingThreads) {
        this.completionService = completionService;
        this.numTrainingThreads = numTrainingThreads;
    }

    @Override
    public LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights,
        double[] oldBetasWithBeta0, double alpha, double lambda, double[] lambdaScaleFactors, double tolerance,
        int maxIterations) {
        LRResult lrResult = new LRResult();
        ArrayList<LRIterationMetadata> metadataList = new ArrayList<>();
        lrResult.setMetaDataList(metadataList);

        // retrieve datasetRanges
        List<DatasetRange> datasetRanges = MultiThreadingUtil
            .splitDatasetIntoRanges(observations, this.numTrainingThreads);

        /*
          Calculate mi (current weight), zi (current target) terms
         */
        long start = System.currentTimeMillis();
        long miZiCalcStartMillis = start; // split train-time metrics

        double[] mi = new double[observations.length];
        double[] zi = new double[observations.length];

        for (DatasetRange datasetRange : datasetRanges) {
            MiZiTask miZiTask = new MiZiTask(datasetRange, oldBetasWithBeta0, mi, zi);
            MultiThreadingUtil.submitTask(this.completionService, miZiTask);
        }

        MultiThreadingUtil.waitForThreadCompletion(this.completionService, this.numTrainingThreads);

        long miZiCalcEndMillis = System.currentTimeMillis(); // split train-time metrics
        lrResult.setMiZiCalcMillis(miZiCalcEndMillis - miZiCalcStartMillis); // split train-time metrics

        /*
          Calculate a-term and c-term parts
         */
        long ajCj1CalcStartMillis = miZiCalcEndMillis; // split train-time metrics
        // aj and cj
        List<double[]> ajList = new LinkedList<>();
        List<double[]> cj_1List = new LinkedList<>();

        for (int i = 0; i < datasetRanges.size(); i++) {
            ajList.add(new double[oldBetasWithBeta0.length]);
            cj_1List.add(new double[oldBetasWithBeta0.length]);
            AjCjTask ajCjTask = new AjCjTask(datasetRanges.get(i), mi, zi, ajList.get(i), cj_1List.get(i),
                totalWeights);
            MultiThreadingUtil.submitTask(this.completionService, ajCjTask);
        }

        MultiThreadingUtil.waitForThreadCompletion(this.completionService, this.numTrainingThreads);

        // combine results

        double[] aj = new double[oldBetasWithBeta0.length];
        double[] cj_1 = new double[oldBetasWithBeta0.length];
        for (int i = 0; i < ajList.size(); i++) {
            // aj (add all results)
            double[] ajI = ajList.get(i);
            Arrays.setAll(aj, idx -> ajI[idx] + aj[idx]);
            // cj_1 (add all results)
            double[] cj_1I = cj_1List.get(i);
            Arrays.setAll(cj_1, idx -> cj_1I[idx] + cj_1[idx]);
        }

        long ajCj1CalcEndMillis = System.currentTimeMillis(); // split train-time metrics
        lrResult.setAjCj1CalcMillis(ajCj1CalcEndMillis - ajCj1CalcStartMillis); // split train-time metrics

        /*
          update and refine betas
         */
        double[] scaledLambdaMulAlpha = new double[oldBetasWithBeta0.length - 1];
        for (int i = 0; i < scaledLambdaMulAlpha.length; ++i) {
            scaledLambdaMulAlpha[i] = lambda * alpha * lambdaScaleFactors[i];
        }
        double[] scaledLambdaMulOneMinusAlpha = new double[oldBetasWithBeta0.length - 1];
        for (int i = 0; i < scaledLambdaMulOneMinusAlpha.length; ++i) {
            scaledLambdaMulOneMinusAlpha[i] = lambda * (1 - alpha) * lambdaScaleFactors[i];
        }
        double[] newBetasWithBeta0 = null;
        double maxAbsDifferencePct = 0;
        double trainingEntropy = 0;
        int iters = 0;

        // Pre-processing for cj_2
        long weightedCovarCalcStartMillis = System.currentTimeMillis(); // split train-time metrics
        double[][] weightedCovar = this.getWeightedCovarianceMatrix(oldBetasWithBeta0.length, observations, mi);
        long weightedCovarCalcEndMillis = System.currentTimeMillis(); // split train-time metrics
        lrResult.setWeightedCovarCalcMillis(
            weightedCovarCalcEndMillis - weightedCovarCalcStartMillis); // split train-time metrics

        long betasUpdateStartMillis = weightedCovarCalcEndMillis; // split train-time metrics
        do {
            long startLoop = System.currentTimeMillis();
            newBetasWithBeta0 = Arrays.copyOf(oldBetasWithBeta0, oldBetasWithBeta0.length);
            for (int j = 0; j < newBetasWithBeta0.length; ++j) {
                if (aj[j] == 0) {
                    newBetasWithBeta0[j] = 0;
                } else {
                    double denominator = j == 0 ? aj[0] : aj[j] + scaledLambdaMulOneMinusAlpha[j - 1];
                    if (denominator != 0) {
                        double cj = calculateCj2(j, weightedCovar, newBetasWithBeta0, cj_1[j], totalWeights);
                        if (j == 0) {
                            newBetasWithBeta0[0] = cj / denominator;
                        } else if (cj < -scaledLambdaMulAlpha[j - 1]) {
                            newBetasWithBeta0[j] =
                                denominator == 0 ? 0 : (cj + scaledLambdaMulAlpha[j - 1]) / denominator;
                        } else if (cj > scaledLambdaMulAlpha[j - 1]) {
                            newBetasWithBeta0[j] =
                                denominator == 0 ? 0 : (cj - scaledLambdaMulAlpha[j - 1]) / denominator;
                        } else {
                            newBetasWithBeta0[j] = 0;
                        }
                    }
                }
            }
            ++iters;
            /*
              Calculate convergence error
             */
            maxAbsDifferencePct = LRUtil.getMaxAbsDifferencePct(oldBetasWithBeta0, newBetasWithBeta0);
            trainingEntropy = LREvalUtil.getEntropy(observations, newBetasWithBeta0);
            long endLoop = System.currentTimeMillis();

            /*
              Record Metrics
             */
            // create metadata
            LRIterationMetadata iterationMetadata = new LRIterationMetadata();
            iterationMetadata.setAlpha(alpha);
            iterationMetadata.setLambda(lambda);
            iterationMetadata.setIteration(iters);
            iterationMetadata.setMaxAbsDifferencePct(maxAbsDifferencePct);
            iterationMetadata.setTrainingEntropy(trainingEntropy);
            iterationMetadata.setBetas(newBetasWithBeta0);
            iterationMetadata.setTrainingTimeMillis(endLoop - startLoop);

            metadataList.add(iterationMetadata);
            /*
              Set New betas to old for next Iteration
             */
            oldBetasWithBeta0 = Arrays.copyOf(newBetasWithBeta0, newBetasWithBeta0.length);
        } while (!LRUtil.hasConverged(maxAbsDifferencePct, tolerance) && iters < maxIterations);
        long betasUpdateEndMillis = System.currentTimeMillis(); // split train-time metrics
        lrResult.setBetasUpdateMillis(betasUpdateEndMillis - betasUpdateStartMillis); // split train-time metrics

        long trainingTimeMillis = System.currentTimeMillis() - start;
        /**
         * Create LRResult
         */

        lrResult.setAlpha(alpha);
        lrResult.setLambda(lambda);
        lrResult.setIteration(iters);
        lrResult.setMaxAbsDifferencePct(maxAbsDifferencePct);
        lrResult.setTrainingEntropy(trainingEntropy);
        lrResult.setBetasWithBeta0(newBetasWithBeta0);
        lrResult.setTrainingTimeMillis(trainingTimeMillis);

        return lrResult;
    }

    /**
     * Calculate the Cj term. This is re-computed after calculating every 'j'th beta
     *
     * @param j                        index of the 'j'th beta starting from beta0
     * @param weightedCovarianceMatrix mi weighted covariance matrix with diagonal terms zeroed out
     * @param currentBetasWithBeta0    current betas with beta0
     * @param cjStaticTerm             cjStaticTerm[0] += mi[i] * zi[i] / W; AND cjStaticTerm[j + 1] += mi[i] * xij * zi[i] / W; // c-terms first part
     * @param totalWeights             sum of all weights / total trials
     */
    static double calculateCj2(int j, double[][] weightedCovarianceMatrix, double[] currentBetasWithBeta0,
        double cjStaticTerm, double totalWeights) {
        double residual = 0;
        for (int k = 0; k < currentBetasWithBeta0.length; ++k) {
            residual += weightedCovarianceMatrix[j][k] * currentBetasWithBeta0[k];
        }
        return cjStaticTerm - residual / totalWeights;
    }

    /**
     * Calculate mi weighted covariance matrix
     *
     * @param size         Width / Height of the square matrix
     * @param observations Array of SparseObservations
     * @param mi           Current Weights
     * @return covarianceMatrix
     */
    double[][] getWeightedCovarianceMatrix(int size, SparseObservation[] observations, double[] mi) {

        List<double[][]> covarianceMatrixResults = new LinkedList<>();

        // retrieve datasetRanges
        List<DatasetRange> datasetRanges = MultiThreadingUtil
            .splitDatasetIntoRanges(observations, this.numTrainingThreads);

        for (int i = 0; i < datasetRanges.size(); i++) {
            covarianceMatrixResults.add(new double[size][size]);
            WeightedCovarianceMatrixTask weightedCovarianceMatrixTask = new WeightedCovarianceMatrixTask(
                datasetRanges.get(i), mi, covarianceMatrixResults.get(i));
            MultiThreadingUtil.submitTask(this.completionService, weightedCovarianceMatrixTask);
        }

        MultiThreadingUtil.waitForThreadCompletion(this.completionService, this.numTrainingThreads);

        double[][] covarianceMatrix = new double[size][size];
        for (double[][] covarianceMatrixResult : covarianceMatrixResults) {
            // TODO, VVAL-249: maybe find a better way to do this
            for (int i = 0; i < covarianceMatrixResult.length; i++) {
                for (int j = 0; j < covarianceMatrixResult[i].length; j++) {
                    covarianceMatrix[i][j] += covarianceMatrixResult[i][j];
                }
            }
        }
        return covarianceMatrix;
    }

    /**
     * Calculate Mi and Zi coefficients for each training observation
     */
    class MiZiTask implements Callable<Boolean> {
        private DatasetRange datasetRange;
        private double[] oldBetasWithBeta0;
        private double[] mi;
        private double[] zi;

        MiZiTask(DatasetRange datasetRange, double[] oldBetasWithBeta0, double[] mi, double[] zi) {
            this.datasetRange = datasetRange;
            this.oldBetasWithBeta0 = oldBetasWithBeta0;
            this.mi = mi;
            this.zi = zi;
        }

        @Override
        public Boolean call() {
            for (int i = this.datasetRange.getStartIdx(); i < this.datasetRange.getEndIdx(); ++i) {
                SparseObservation o = (SparseObservation) this.datasetRange.getDataset()[i];
                double betasDotXi = LRUtil.betasDotXi(o.getX(), this.oldBetasWithBeta0);
                double prob = LRUtil.calcProb(betasDotXi);
                double probBounded = Math.min(1.0 - PROB_EPSILON, Math.max(PROB_EPSILON, prob));
                double wi = o.getWeight();
                // fill in mi and zi
                this.mi[i] = wi * probBounded * (1 - probBounded);
                this.zi[i] = betasDotXi + (o.getY() - wi * prob) / this.mi[i];
            }
            return true;
        }
    }


    /**
     * Calculate Aj and Cj coefficients for each weight
     */
    class AjCjTask implements Callable<Boolean> {
        private DatasetRange datasetRange;
        private double[] mi;
        private double[] zi;
        private double[] aj;
        private double[] cj_1;
        private double totalWeights;

        AjCjTask(DatasetRange datasetRange, double[] mi, double[] zi, double[] aj, double[] cj_1, double totalWeights) {
            this.datasetRange = datasetRange;
            this.mi = mi;
            this.zi = zi;
            this.aj = aj;
            this.cj_1 = cj_1;
            this.totalWeights = totalWeights;
        }

        public Boolean call() {
            for (int i = this.datasetRange.getStartIdx(); i < this.datasetRange.getEndIdx(); ++i) {
                SparseObservation o = (SparseObservation) this.datasetRange.getDataset()[i];
                this.aj[0] += this.mi[i]; // a-term for intercept
                this.cj_1[0] += this.mi[i] * this.zi[i]; // c-term first part for intercept
                for (SparseArray.Entry xj : o.getX()) {
                    int j = xj.i;
                    double xij = xj.x;
                    this.aj[j + 1] += this.mi[i] * xij * xij; // a-terms
                    this.cj_1[j + 1] += this.mi[i] * xij * this.zi[i]; // c-terms first part
                }
            }
            // normalize everything
            for (int i = 0; i < this.aj.length; i++) {
                this.aj[i] /= this.totalWeights;
                this.cj_1[i] /= this.totalWeights;
            }
            return true;
        }
    }


    /**
     * Calculate covariance matrix of training observations weighted by Mi
     */
    class WeightedCovarianceMatrixTask implements Callable<Boolean> {
        private DatasetRange datasetRange;
        private double[] mi;
        private double[][] weightedCovarianceMatrix;

        WeightedCovarianceMatrixTask(DatasetRange datasetRange, double[] mi, double[][] weightedCovarianceMatrix) {
            this.datasetRange = datasetRange;
            this.mi = mi;
            this.weightedCovarianceMatrix = weightedCovarianceMatrix;
        }

        @Override
        public Boolean call() {
            for (int i = this.datasetRange.getStartIdx(); i < this.datasetRange.getEndIdx(); ++i) {
                SparseObservation o = (SparseObservation) this.datasetRange.getDataset()[i];
                for (SparseArray.Entry xRowj : o.getX()) {
                    int j = xRowj.i + 1;
                    for (SparseArray.Entry xRowk : o.getX()) {
                        int k = xRowk.i + 1;
                        if (j < k) {
                            double value = this.mi[i] * xRowj.x * xRowk.x;
                            this.weightedCovarianceMatrix[j][k] += value;
                            this.weightedCovarianceMatrix[k][j] += value;
                        }
                    }
                }
                for (SparseArray.Entry xRowj : o.getX()) {
                    int j = xRowj.i + 1;
                    if (j != 0) {
                        double value = this.mi[i] * xRowj.x * 1; // multiply by 1?
                        this.weightedCovarianceMatrix[j][0] += value;
                        this.weightedCovarianceMatrix[0][j] += value;

                    }
                }
            }
            return true;
        }
    }
}
