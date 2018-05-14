package com.appnexus.opt.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CoordinateDescentTrainer implements IModelTrainer {
    private static final double PROB_EPSILON = 1e-15;

    @Override
    public LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights, double[] oldBetasWithBeta0, double alpha, double lambda, double[] lambdaScaleFactors, double tolerance, int maxIterations) {
        LRResult lrResult = new LRResult();
        List<LRIterationMetadata> metadataList = new ArrayList<>();
        lrResult.setMetaDataList(metadataList);
        long start = System.currentTimeMillis();

        /*
          Calculate mi (current weight) and zi (current target) terms for each observation
         */
        double[] mi = new double[observations.length];
        double[] zi = new double[observations.length];
        for (int i = 0; i < observations.length; ++i) {
            double betasDotXi = LRUtil.betasDotXi(observations[i].getX(), oldBetasWithBeta0);
            double prob = LRUtil.calcProb(betasDotXi);
            double probBounded = Math.min(1.0 - PROB_EPSILON, Math.max(PROB_EPSILON, prob));
            double wi = observations[i].getWeight();
            mi[i] = wi * probBounded * (1 - probBounded);
            zi[i] = betasDotXi + (observations[i].getY() - wi * prob) / mi[i];
        }

        /*
          Calculate a-term and static c-term components for each beta weight
         */
        double[] aj = new double[oldBetasWithBeta0.length];
        double[] cjStaticTerm = new double[oldBetasWithBeta0.length];
        for (int i = 0; i < observations.length; ++i) {
            aj[0] += mi[i]; // a-term for intercept
            cjStaticTerm[0] += mi[i] * zi[i]; // c-term first part for intercept
            for (SparseArray.Entry xj : observations[i].getX()) {
                int j = xj.i;
                double xij = xj.x;
                aj[j + 1] += mi[i] * xij * xij; // a-terms
                cjStaticTerm[j + 1] += mi[i] * xij * zi[i]; // c-terms first part
            }
        }
        for (int j = 0; j < oldBetasWithBeta0.length; ++j) {
            aj[j] /= totalWeights;
            cjStaticTerm[j] /= totalWeights;
        }

        /*
          Update and refine betas until convergence
         */
        double[] scaledLambdaMulAlpha = new double[oldBetasWithBeta0.length - 1];
        for (int i = 0; i < scaledLambdaMulAlpha.length; ++i) {
            scaledLambdaMulAlpha[i] = lambda * alpha * lambdaScaleFactors[i];
        }
        double[] scaledLambdaMulOneMinusAlpha = new double[oldBetasWithBeta0.length - 1];
        for (int i = 0; i < scaledLambdaMulOneMinusAlpha.length; ++i) {
            scaledLambdaMulOneMinusAlpha[i] = lambda * (1 - alpha) * lambdaScaleFactors[i];
        }
        double[] newBetasWithBeta0;
        double maxAbsDifferencePct;
        double trainingEntropy;
        int iterations = 0;
        // Pre-processing: compute weighted covariance matrix
        // long preProcStart = System.currentTimeMillis();
        double[][] weightedCovarianceMatrix = getWeightedCovarianceMatrix(oldBetasWithBeta0.length, observations, mi);
        // long preProcEnd = System.currentTimeMillis();
        // Update betas
        do {
            long startLoop = System.currentTimeMillis();
            newBetasWithBeta0 = Arrays.copyOf(oldBetasWithBeta0, oldBetasWithBeta0.length);
            for (int j = 0; j < newBetasWithBeta0.length; ++j) {
                if (aj[j] == 0) {
                    newBetasWithBeta0[j] = 0;
                } else {
                    double denominator = j == 0 ? aj[0] : aj[j] + scaledLambdaMulOneMinusAlpha[j - 1];
                    if (denominator != 0) {
                        double cj = calculateCj2(j, weightedCovarianceMatrix, newBetasWithBeta0, cjStaticTerm[j], totalWeights);
                        if (j == 0) {
                            newBetasWithBeta0[0] = cj / denominator;
                        } else if (cj < -scaledLambdaMulAlpha[j - 1]) {
                            newBetasWithBeta0[j] = denominator == 0 ? 0 : (cj + scaledLambdaMulAlpha[j - 1]) / denominator;
                        } else if (cj > scaledLambdaMulAlpha[j - 1]) {
                            newBetasWithBeta0[j] = denominator == 0 ? 0 : (cj - scaledLambdaMulAlpha[j - 1]) / denominator;
                        } else {
                            newBetasWithBeta0[j] = 0;
                        }
                    }
                }
            }
            ++iterations;

            /*
              Calculate convergence error
             */
            maxAbsDifferencePct = LRUtil.getMaxAbsDifferencePct(oldBetasWithBeta0, newBetasWithBeta0);
            trainingEntropy = LREvalUtil.getEntropy(observations, newBetasWithBeta0);
            long endLoop = System.currentTimeMillis();

            /*
              Record Metrics
             */
            LRIterationMetadata iterationMetadata = new LRIterationMetadata();
            iterationMetadata.setAlpha(alpha);
            iterationMetadata.setLambda(lambda);
            iterationMetadata.setIteration(iterations);
            iterationMetadata.setMaxAbsDifferencePct(maxAbsDifferencePct);
            iterationMetadata.setTrainingEntropy(trainingEntropy);
            iterationMetadata.setBetas(newBetasWithBeta0);
            iterationMetadata.setTrainingTimeMillis(endLoop - startLoop);

            metadataList.add(iterationMetadata);

            /*
              Set New betas to old for next Iteration
             */
            oldBetasWithBeta0 = Arrays.copyOf(newBetasWithBeta0, newBetasWithBeta0.length);
        } while (!LRUtil.hasConverged(maxAbsDifferencePct, tolerance) && iterations < maxIterations);
        long trainingTimeMillis = System.currentTimeMillis() - start;

        /*
          Create LRResult
         */
        lrResult.setAlpha(alpha);
        lrResult.setLambda(lambda);
        lrResult.setIteration(iterations);
        lrResult.setMaxAbsDifferencePct(maxAbsDifferencePct);
        lrResult.setTrainingEntropy(trainingEntropy);
        lrResult.setBetasWithBeta0(newBetasWithBeta0);
        lrResult.setTrainingTimeMillis(trainingTimeMillis);
        return lrResult;
    }

    /**
     * Calculate the Cj term. This is re-computed after calculating every 'j'th beta
     * 
     * @param j index of the 'j'th beta starting from beta0
     * @param weightedCovarianceMatrix mi weighted covariance matrix with diagonal terms zeroed out
     * @param currentBetasWithBeta0 current betas with beta0
     * @param cjStaticTerm cjStaticTerm[0] += mi[i] * zi[i] / W; AND cjStaticTerm[j + 1] += mi[i] * xij * zi[i] / W; // c-terms first part
     * @param totalWeights sum of all weights / total trials
     */
    static double calculateCj2(int j, double[][] weightedCovarianceMatrix, double[] currentBetasWithBeta0, double cjStaticTerm, double totalWeights) {
        double residual = 0;
        for (int k = 0; k < currentBetasWithBeta0.length; ++k) {
            residual += weightedCovarianceMatrix[j][k] * currentBetasWithBeta0[k];
        }
        return cjStaticTerm - residual / totalWeights;
    }

    /**
     * Calculate mi weighted covariance matrix
     * 
     * @param size dimensions of the square matrix
     * @param observations array of sparse observations
     * @param mi weights
     * @return weighted covariance matrix of observation data
     */
    static double[][] getWeightedCovarianceMatrix(int size, SparseObservation[] observations, double[] mi) {
        double[][] weightedCovarianceMatrix = new double[size][size];
        for (int i = 0; i < observations.length; ++i) {
            /*
                compute sum of Xj * Xk where j < k
                note: matrix is symmetrical
             */
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                for (SparseArray.Entry xRowk : observations[i].getX()) {
                    int k = xRowk.i + 1;
                    if (j < k) {
                        double value = mi[i] * xRowj.x * xRowk.x;
                        weightedCovarianceMatrix[j][k] += value;
                        weightedCovarianceMatrix[k][j] += value;
                    }
                }
            }
            /*
                add in the entries of the matrix for the 0th row and the 0th column (i.e. beta 0)
                note: beta0 will always have an X value of 1 since it's "always present"
             */
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                if (j != 0) {
                    double value = mi[i] * xRowj.x * 1;
                    weightedCovarianceMatrix[j][0] += value;
                    weightedCovarianceMatrix[0][j] += value;

                }
            }
        }
        return weightedCovarianceMatrix;
    }
}
