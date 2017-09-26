package com.appnexus.opt.ml;

public class LREvalUtil {

    private static final double EPS = 1e-15;

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @return
     */
    public static double getEntropy(SparseObservation[] obs, double[] betasWithBeta0) {
        double error = 0;
        for (SparseObservation o : obs) {
            double prob = LRUtil.calcProb(o.getX(), betasWithBeta0);
            double pred = Math.min(1.0 - EPS, Math.max(EPS, prob));
            double errorObs = -1 * o.getY() * Math.log(pred) - (o.getWeight() - o.getY()) * Math.log(1.0 - pred);
            error = error + errorObs;
        }
        return error;
    }

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @param c
     * @return
     */
    public static double getEntropyScaled(SparseObservation[] obs, double[] betasWithBeta0, double c) {
        double error = 0;
        for (SparseObservation o : obs) {
            double prob = c * LRUtil.calcProb(o.getX(), betasWithBeta0);
            double pred = Math.min(1.0 - EPS, Math.max(EPS, prob));
            double errorObs = -1 * o.getY() * Math.log(pred) - (o.getWeight() - o.getY()) * Math.log(1.0 - pred);
            error = error + errorObs;
        }
        return error;
    }

    /**
     *
     * @param obs
     * @param betasWithBeta0
     * @return (number of predicted clicks - number of actual clicks)/(number of actual clicks)
     */
    public static double getBias(SparseObservation[] obs, double[] betasWithBeta0) {
        double pred = 0;
        double actual = 0;
        for (SparseObservation o : obs) {
            actual = actual + o.getY();
            pred = pred + LRUtil.calcProb(o.getX(), betasWithBeta0) * o.getWeight();
        }
        return actual == 0 ? Double.MAX_VALUE : (pred - actual) / actual;
    }

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @param c
     * @return
     */
    public static double getBiasScaled(SparseObservation[] obs, double[] betasWithBeta0, double c) {
        double pred = 0;
        double actual = 0;
        for (SparseObservation o : obs) {
            actual = actual + o.getY();
            pred = pred + c * LRUtil.calcProb(o.getX(), betasWithBeta0) * o.getWeight();
        }
        return actual == 0 ? Double.MAX_VALUE : (pred - actual) / actual;
    }

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @return
     */
    public static double getPredRatio(SparseObservation[] obs, double[] betasWithBeta0) {
        double clickWeight = 0;
        double nonClickWeight = 0;
        double clickProb = 0;
        double nonClickProb = 0;
        for (SparseObservation o : obs) {
            double prob = LRUtil.calcProb(o.getX(), betasWithBeta0);
            clickWeight = clickWeight + o.getY();
            clickProb = clickProb + o.getY() * prob;
            nonClickWeight = nonClickWeight + o.getWeight() - o.getY();
            nonClickProb = nonClickProb + ((o.getWeight() - o.getY()) * prob);
        }
        return (clickWeight == 0 || nonClickProb == 0) ? 0 : (clickProb / clickWeight) / (nonClickProb / nonClickWeight);
    }

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @param c
     * @return
     */
    public static double getPredRatioScaled(SparseObservation[] obs, double[] betasWithBeta0, double c) {
        double clickWeight = 0;
        double nonClickWeight = 0;
        double clickProb = 0;
        double nonClickProb = 0;
        for (SparseObservation o : obs) {
            double prob = c * LRUtil.calcProb(o.getX(), betasWithBeta0);
            clickWeight = clickWeight + o.getY();
            clickProb = clickProb + o.getY() * prob;
            nonClickWeight = nonClickWeight + o.getWeight() - o.getY();
            nonClickProb = nonClickProb + ((o.getWeight() - o.getY()) * prob);
        }
        return (clickWeight == 0 || nonClickProb == 0) ? 0 : (clickProb / clickWeight) / (nonClickProb / nonClickWeight);
    }

    /**
     * 
     * @param obs
     * @param betasWithBeta0
     * @return
     */
    public static double calculateScaleFactor(SparseObservation[] obs, double[] betasWithBeta0) {
        double actualClicks = 0;
        double predictedClicks = 0;
        for (SparseObservation o : obs) {
            actualClicks += o.getY();
            predictedClicks += o.getWeight() * LRUtil.calcProb(o.getX(), betasWithBeta0);
        }
        return predictedClicks == 0 ? 0 : actualClicks / predictedClicks;
    }
}
