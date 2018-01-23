package com.appnexus.opt.ml;

import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class LRUtil {

    public static double calcProb(SparseArray xRow, double[] betasWithBeta0) {
        return calcProb(betasDotXi(xRow, betasWithBeta0));
    }

    public static double calcProb(double betasDotXi) {
        return expit(betasDotXi);
    }

    public static double expit(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double betasDotXi(SparseArray xRow, double[] betasWithBeta0) {
        double betasDotXi = betasWithBeta0[0];
        for (Entry entry : xRow) {
            betasDotXi += entry.x * betasWithBeta0[entry.i + 1];
        }
        return betasDotXi;
    }

    public static double[] getLambdaGrid(int size, double start, double end) {
        if (size > 0) {
            double[] grid = new double[size];
            double step = (end + 1 - start) / size;
            for (int i = 0; i < size - 1; ++i) {
                grid[i] = Math.exp(-1 * (i * step + start));
            }
            grid[size - 1] = 0;
            return grid;
        }
        return null;
    }

    /**
     * 
     * @param oldBetas Betas from previous iteration
     * @param newBetas Betas from latest iteration
     * @param tolerance
     * @return
     */
    public static boolean hasConverged(double[] oldBetas, double[] newBetas, double tolerance) {
        return hasConverged(getMaxAbsDifferencePct(oldBetas, newBetas), tolerance);
    }

    /**
     * 
     * @param maxAbsDifferencePct Max absolute difference percentage between old and new betas
     * @param tolerance
     * @return
     */
    public static boolean hasConverged(double maxAbsDifferencePct, double tolerance) {
        return maxAbsDifferencePct <= tolerance;
    }

    /**
     * Get Max Absolute % difference between oldBetas[j] and newBetas[j] across all values of j
     * 
     * @param oldBetas
     * @param newBetas
     * @return
     */
    public static double getMaxAbsDifferencePct(double[] oldBetas, double[] newBetas) {
        double sumAbsOfNewBetas = 0;
        for (int i = 0; i < newBetas.length; ++i) {
            sumAbsOfNewBetas += Math.abs(newBetas[i]);
        }
        if (sumAbsOfNewBetas == 0.0) {
            return -Double.MAX_VALUE; // TODO - is this bad?
        }
        double maxAbsBetaDiff = -Double.MAX_VALUE;
        for (int i = 0; i < newBetas.length; ++i) {
            maxAbsBetaDiff = Math.max(maxAbsBetaDiff, Math.abs(newBetas[i] - oldBetas[i]));
        }
        return maxAbsBetaDiff / sumAbsOfNewBetas;
    }

    /**
     * Generate Scale Factors for scaling lambda for each beta[j]
     * 
     * @param trainingObsArr
     * @param featureVectorLen
     * @return
     */
    public static double[] generateLambdaScaleFactors(SparseObservation[] trainingObsArr, int featureVectorLen) {
        double[] lambdaScaleFactors = new double[featureVectorLen];
        double imps = 0;
        for (SparseObservation trainingObs : trainingObsArr) {
            imps += trainingObs.getWeight();
            for (smile.math.SparseArray.Entry feature : trainingObs.getX()) {
                lambdaScaleFactors[feature.i] += trainingObs.getY();
            }
        }
        // TODO think about 1/imps vs ?/imps
        for (int i = 0; i < lambdaScaleFactors.length; ++i) {
            lambdaScaleFactors[i] = lambdaScaleFactors[i] == 0 ? 1 / imps : lambdaScaleFactors[i] / imps;
        }
        return lambdaScaleFactors;
    }
}
