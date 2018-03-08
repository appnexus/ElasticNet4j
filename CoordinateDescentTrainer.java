package com.appnexus.opt.ml;

import java.util.ArrayList;
import java.util.Arrays;

public class CoordinateDescentTrainer implements IModelTrainer {
    private static final double PROB_EPSILON = 1e-15;

    @Override
    public LRResult trainNewBetasWithBeta0(SparseObservation[] observations, double totalWeights, double[] oldBetasWithBeta0, double alpha, double lambda, double[] lambdaScaleFactors, double tolerance, int maxIterations) {
        LRResult lrResult = new LRResult();
        ArrayList<LRIterationMetaData> metaDataList = new ArrayList<LRIterationMetaData>();
        lrResult.setMetaDataList(metaDataList);
        long start = System.currentTimeMillis();
        /**
         * Calculate mi (current weight), zi (current target) terms
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

        /**
         * Calculate a-term and c-term parts
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

        /**
         * update and refine betas
         */
        double lambdaMulAlpha = lambda * alpha;
        double lambdaMulOneMinusAlpha = lambda * (1 - alpha);
        double[] newBetasWithBeta0 = null;
        double maxAbsDifferencePct = 0;
        double trainingEntropy = 0;
        int iters = 0;

        // Pre-processing
        // long preProcStart = System.currentTimeMillis();
        double[][] weightedCovar = new double[oldBetasWithBeta0.length][oldBetasWithBeta0.length];
        for (int i = 0; i < observations.length; ++i) {
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                for (SparseArray.Entry xRowk : observations[i].getX()) {
                    int k = xRowk.i + 1;
                    if (j != k) {
                        weightedCovar[j][k] += mi[i] * xRowj.x * xRowk.x;
                    }
                }
            }
            for (SparseArray.Entry xRowj : observations[i].getX()) {
                int j = xRowj.i + 1;
                if (j != 0) {
                    weightedCovar[j][0] += mi[i] * xRowj.x * 1;
                }
            }
            for (SparseArray.Entry xRowk : observations[i].getX()) {
                int k = xRowk.i + 1;
                if (k != 0) {
                    weightedCovar[0][k] += mi[i] * 1 * xRowk.x;
                }
            }
        }
        // long preProcEnd = System.currentTimeMillis();

        do {
            long startLoop = System.currentTimeMillis();

            newBetasWithBeta0 = Arrays.copyOf(oldBetasWithBeta0, oldBetasWithBeta0.length);
            for (int j = 0; j < newBetasWithBeta0.length; ++j) {
                if (aj[j] == 0) {
                    newBetasWithBeta0[j] = 0;
                } else {
                    double denominator = j == 0 ? aj[0] : aj[j] + lambdaMulOneMinusAlpha;
                    if (denominator != 0) {
                        double cj = calculateCj2(j, weightedCovar, newBetasWithBeta0, cj_1[j], totalWeights);
                        if (j == 0) {
                            newBetasWithBeta0[0] = cj / denominator;
                        } else if (cj < -lambdaMulAlpha) {
                            newBetasWithBeta0[j] = denominator == 0 ? 0 : (cj + lambdaMulAlpha) / denominator;
                        } else if (cj > lambdaMulAlpha) {
                            newBetasWithBeta0[j] = denominator == 0 ? 0 : (cj - lambdaMulAlpha) / denominator;
                        } else {
                            newBetasWithBeta0[j] = 0;
                        }
                    }
                }
            }
            ++iters;
            /**
             * Calculate convergence error
             */
            maxAbsDifferencePct = LRUtil.getMaxAbsDifferencePct(oldBetasWithBeta0, newBetasWithBeta0);
            trainingEntropy = LREvalUtil.getEntropy(observations, newBetasWithBeta0);
            long endLoop = System.currentTimeMillis();

            /**
             * Record Metrics
             */
            // Create MetaData
            LRIterationMetaData lrmd = new LRIterationMetaData();
            lrmd.setAlpha(alpha);
            lrmd.setLambda(lambda);
            lrmd.setIteration(iters);
            lrmd.setMaxAbsDifferencePct(maxAbsDifferencePct);
            lrmd.setTrainingEntropy(trainingEntropy);
            lrmd.setBetas(newBetasWithBeta0);
            lrmd.setTrainingTimeMillis(endLoop - startLoop);

            metaDataList.add(lrmd);
            // System.out.println(
            // alpha + ", " + lambda + ", " + iters + ", " + maxAbsDifferencePct + ", " + (endLoop - startLoop));
            /**
             * Set New betas to old for next Iteration
             */
            oldBetasWithBeta0 = Arrays.copyOf(newBetasWithBeta0, newBetasWithBeta0.length);
        } while (!LRUtil.hasConverged(maxAbsDifferencePct, tolerance) && iters < maxIterations);

        // if (iters == maxIterations) System.out.println("Did not converge");

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
}
