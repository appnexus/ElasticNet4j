package com.appnexus.opt.ml;

import java.util.ArrayList;
import java.util.Arrays;

public class CoordinateDescentTrainer implements IModelTrainer {
    private static final double PROB_EPSILON = 1e-15;

    @Override
    public LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights, double[] oldBetasWithBeta0, double alpha, double lambda, double[] lambdaScaleFactors, double tolerance, int maxIterations) {
        LRResult lrResult = new LRResult();
        ArrayList<LRIterationMetaData> metaDataList = new ArrayList<>();
        lrResult.setMetaDataList(metaDataList);
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
          Calculate a-term and c-term components for each beta weight
         */
        double[] aj = new double[oldBetasWithBeta0.length];
        double[] cj_1 = new double[oldBetasWithBeta0.length];
        for (int i = 0; i < observations.length; ++i) {
            aj[0] += mi[i]; // a-term for intercept
            cj_1[0] += mi[i] * zi[i]; // c-term first part for intercept
            for (SparseArray.Entry xj : observations[i].getX()) {
                int j = xj.i;
                double xij = xj.x;
                aj[j + 1] += mi[i] * xij * xij; // a-terms
                cj_1[j + 1] += mi[i] * xij * zi[i]; // c-terms first part
            }
        }
        for (int j = 0; j < oldBetasWithBeta0.length; ++j) {
            aj[j] /= totalWeights;
            cj_1[j] /= totalWeights;
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
        double[] newBetasWithBeta0 = null;
        double maxAbsDifferencePct = 0;
        double trainingEntropy = 0;
        int iterations = 0;

        // Pre-processing
        // long preProcStart = System.currentTimeMillis();
        double[][] weightedCovar = getWeightedCovarianceMatrix(oldBetasWithBeta0.length, observations, mi);
        // long preProcEnd = System.currentTimeMillis();

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
            // Create MetaData
            LRIterationMetaData lrmd = new LRIterationMetaData();
            lrmd.setAlpha(alpha);
            lrmd.setLambda(lambda);
            lrmd.setIteration(iterations);
            lrmd.setMaxAbsDifferencePct(maxAbsDifferencePct);
            lrmd.setTrainingEntropy(trainingEntropy);
            lrmd.setBetas(newBetasWithBeta0);
            lrmd.setTrainingTimeMillis(endLoop - startLoop);

            metaDataList.add(lrmd);
            // System.out.println(
            // alpha + ", " + lambda + ", " + iterations + ", " + maxAbsDifferencePct + ", " + (endLoop - startLoop));
            /*
              Set New betas to old for next Iteration
             */
            oldBetasWithBeta0 = Arrays.copyOf(newBetasWithBeta0, newBetasWithBeta0.length);
        } while (!LRUtil.hasConverged(maxAbsDifferencePct, tolerance) && iterations < maxIterations);

        // if (iterations == maxIterations) System.out.println("Did not converge");

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
     * @param j : index of the 'j'th beta starting from beta0
     * @param weightedCovar : mi weighted covar matrix with diagonal terms zeroed out
     * @param currentbetasWithBeta0
     * @param cj_1 -> cj_1[0] += mi[i] * zi[i] / W; AND cj_1[j + 1] += mi[i] * xij * zi[i] / W; // c-terms first part
     * @param totalWeights -> sum of all weights / total trials
     */
    private static double calculateCj2(int j, double[][] weightedCovar, double[] currentbetasWithBeta0, double cj_1, double totalWeights) {
        double residual = 0;
        for (int k = 0; k < currentbetasWithBeta0.length; ++k) {
            residual += weightedCovar[j][k] * currentbetasWithBeta0[k];
        }
        return cj_1 - residual / totalWeights;
    }

    /**
     * Calculate mi weighted covariance matrix
     * 
     * @param size Width / Height of the square matrix
     * @param observations Array of SparseObservations
     * @param mi Current Weights
     * @return
     */
    static double[][] getWeightedCovarianceMatrix(int size, SparseObservation[] observations, double[] mi) {
        double[][] weightedCovar = new double[size][size];
        for (int i = 0; i < observations.length; ++i) {
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                for (SparseArray.Entry xRowk : observations[i].getX()) {
                    int k = xRowk.i + 1;
                    if (j < k) {
                        double value = mi[i] * xRowj.x * xRowk.x;
                        weightedCovar[j][k] += value;
                        weightedCovar[k][j] += value;
                    }
                }
            }
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                if (j != 0) {
                    double value = mi[i] * xRowj.x * 1; // Multipy by 1 ?
                    weightedCovar[j][0] += value;
                    weightedCovar[0][j] += value;

                }
            }
        }
        return weightedCovar;
    }
}
