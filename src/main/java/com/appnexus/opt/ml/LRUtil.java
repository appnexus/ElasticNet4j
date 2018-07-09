package com.appnexus.opt.ml;

public class LRUtil {

    /**
     * @param xRow           data
     * @param betasWithBeta0 beta weights
     * @return probability(Xi)
     */
    public static double calcProb(SparseArray xRow, double[] betasWithBeta0) {
        return calcProb(betasDotXi(xRow, betasWithBeta0));
    }

    /**
     * @param betasDotXi dot product of beta weights and data
     * @return probability(Xi)
     */
    public static double calcProb(double betasDotXi) {
        return expit(betasDotXi);
    }

    /**
     * @param z input
     * @return inverse logit of z
     */
    public static double expit(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * @param xRow           data
     * @param betasWithBeta0 beta weights
     * @return dot product of data and beta weights
     */
    public static double betasDotXi(SparseArray xRow, double[] betasWithBeta0) {
        double betasDotXi = betasWithBeta0[0];
        for (SparseArray.Entry entry : xRow) {
            betasDotXi += entry.x * betasWithBeta0[entry.i + 1];
        }
        return betasDotXi;
    }

    public static double[] getLambdaGrid(int size, double start, double end) {
        if (size > 0) {
            double[] grid = new double[size];
            double step = (size == 1) ? 0.0 : (end - start) / (size - 1);
            for (int i = 0; i < size; ++i) {
                grid[i] = Math.exp(-1 * (i * step + start));
            }
            return grid;
        }
        return null;
    }

    /**
     * @param oldBetas  Betas from previous iteration
     * @param newBetas  Betas from latest iteration
     * @param tolerance
     * @return
     */
    public static boolean hasConverged(double[] oldBetas, double[] newBetas, double tolerance) {
        return hasConverged(getMaxAbsDifferencePct(oldBetas, newBetas), tolerance);
    }

    /**
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
        double totalWeight = 0;
        for (SparseObservation trainingObs : trainingObsArr) {
            totalWeight += trainingObs.getWeight();
            for (SparseArray.Entry feature : trainingObs.getX()) {
                lambdaScaleFactors[feature.i] += trainingObs.getY();
            }
        }
        // TODO think about 1/totalWeight vs ?/totalWeight
        for (int i = 0; i < lambdaScaleFactors.length; ++i) {
            lambdaScaleFactors[i] = lambdaScaleFactors[i] == 0 ? 1 / totalWeight : lambdaScaleFactors[i] / totalWeight;
        }
        return lambdaScaleFactors;
    }
}
