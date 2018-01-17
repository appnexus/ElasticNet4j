package com.appnexus.opt.ml;

import java.util.Arrays;
import java.util.LinkedList;

public class LR {
    private final SparseObservation[] observations; // contains x, y and weight. x does not hold the intercept feature
                                                    // for beta0
    private final int numOfFeatures;
    private final double totalSuccesses;
    private final double totalWeights;
    private final double[] initialBetasWithBeta0; // beta0 in first index
    private final double alpha;
    private final double[] lambdaGrid;
    private final double[] lambdaScaleFactors;
    private final double tolerance;
    private final int maxIterations;
    private final IModelTrainer modelTrainer;

    public LR(SparseObservation[] observations, int numOfFeatures, double[] initialBetasWithBeta0, double alpha, double[] lambdaGrid, double[] lambdaScaleFactors, double tolerance, int maxIterations, IModelTrainer modelTrainer) {
        this.observations = observations;
        this.numOfFeatures = numOfFeatures;
        this.totalSuccesses = getTotalSuccesses(this.observations);
        // System.out.println("TOTAL SUCCESSES -> " + this.totalSuccesses); // TODO --
        // remove
        // System.out.println("TOTAL FAILURES -> " +
        // getTotalFailure(this.observations)); // TODO -- remove
        this.totalWeights = getTotalWeights(this.observations);
        // System.out.println("TOTAL WEIGHTS -> " + this.totalWeights);// TODO -- remove
        this.initialBetasWithBeta0 = getInitialBetasWithBeta0(initialBetasWithBeta0);
        this.alpha = alpha;
        this.lambdaGrid = lambdaGrid;
        this.lambdaScaleFactors = lambdaScaleFactors;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.modelTrainer = modelTrainer;
    }

    private double[] getInitialBetasWithBeta0(double[] initBetasWithBeta0) {
        double betasWithBeta0[] = initBetasWithBeta0;
        if (betasWithBeta0 == null) {
            betasWithBeta0 = new double[this.numOfFeatures + 1];
            betasWithBeta0[0] = guessInitialBetaZero();
        }
        return betasWithBeta0;
    }

    private double guessInitialBetaZero() {
        double globalCtr = this.totalSuccesses / this.totalWeights;
        return Math.log(globalCtr / (1 - globalCtr));
    }

    public LinkedList<LRResult> calculateBetas(boolean warmStart) {
        LinkedList<LRResult> lrResultList = new LinkedList<LRResult>();
        LRResult lrResult = null;
        for (double lambda : this.lambdaGrid) {
            double[] startBetasWithBeta0 = ((warmStart && lrResult != null) ? Arrays.copyOf(lrResult.getBetasWithBeta0(), lrResult.getBetasWithBeta0().length) : Arrays.copyOf(this.initialBetasWithBeta0, this.initialBetasWithBeta0.length));
            lrResult = calculateBetas(startBetasWithBeta0, lambda, this.lambdaScaleFactors);
            lrResultList.add(lrResult);
        }
        return lrResultList;
    }

    public LRResult calculateBetas(double[] startBetasWithBeta0, double lambda, double[] lambdaScaleFactors) {
        LRResult lrResult = this.modelTrainer.trainNewBetasWithBeta0(this.observations, this.totalWeights, startBetasWithBeta0, this.alpha, lambda, lambdaScaleFactors, tolerance, maxIterations);
        return lrResult;
    }

    /**
     * HELPERS
     */
    private static double getTotalSuccesses(SparseObservation[] observations) {
        double totalSuccesses = 0;
        for (SparseObservation obs : observations) {
            totalSuccesses += obs.getY();
        }
        return totalSuccesses;
    }

    private static double getTotalWeights(SparseObservation[] observations) {
        double totalWeights = 0;
        for (SparseObservation obs : observations) {
            totalWeights += obs.getWeight();
        }
        return totalWeights;
    }

}
