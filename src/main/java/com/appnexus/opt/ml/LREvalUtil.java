package com.appnexus.opt.ml;

public class LREvalUtil {

    private static final double EPS = 1e-15;

    /**
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @return cross entropy of p from data vs true distribution
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
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @return cross entropy but normalized by number of impressions
     */
    public static double getEntropyNormalized(SparseObservation[] obs, double[] betasWithBeta0) {
        double error = 0;
        double totalWeight = 0;
        for (SparseObservation o : obs) {
            totalWeight += o.getWeight();
            double prob = LRUtil.calcProb(o.getX(), betasWithBeta0);
            double pred = Math.min(1.0 - EPS, Math.max(EPS, prob));
            double errorObs = -1 * o.getY() * Math.log(pred) - (o.getWeight() - o.getY()) * Math.log(1.0 - pred);
            error = error + errorObs;
        }
        return error / totalWeight;
    }

    /**
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @param scale          scale factor
     * @return cross entropy but with probabilities scaled by scale factor
     */
    public static double getEntropyScaled(SparseObservation[] obs, double[] betasWithBeta0, double scale) {
        double error = 0;
        for (SparseObservation o : obs) {
            double prob = scale * LRUtil.calcProb(o.getX(), betasWithBeta0);
            double pred = Math.min(1.0 - EPS, Math.max(EPS, prob));
            double errorObs = -1 * o.getY() * Math.log(pred) - (o.getWeight() - o.getY()) * Math.log(1.0 - pred);
            error = error + errorObs;
        }
        return error;
    }

    /**
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @return bias = (number of predicted y - number of actual y) / (number of actual y)
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
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @param scale          scale factor
     * @return bias but with probabilities scaled by scale factor
     */
    public static double getBiasScaled(SparseObservation[] obs, double[] betasWithBeta0, double scale) {
        double pred = 0;
        double actual = 0;
        for (SparseObservation o : obs) {
            actual = actual + o.getY();
            pred = pred + scale * LRUtil.calcProb(o.getX(), betasWithBeta0) * o.getWeight();
        }
        return actual == 0 ? Double.MAX_VALUE : (pred - actual) / actual;
    }

    /**
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @return prediction ratio
     */
    public static double getPredRatio(SparseObservation[] obs, double[] betasWithBeta0) {
        double yWeight = 0;
        double nonYWeight = 0;
        double yProb = 0;
        double nonYProb = 0;
        for (SparseObservation o : obs) {
            double prob = LRUtil.calcProb(o.getX(), betasWithBeta0);
            yWeight = yWeight + o.getY();
            yProb = yProb + o.getY() * prob;
            nonYWeight = nonYWeight + o.getWeight() - o.getY();
            nonYProb = nonYProb + ((o.getWeight() - o.getY()) * prob);
        }
        return (yWeight == 0 || nonYProb == 0) ?
            0 :
            (yProb / yWeight) / (nonYProb / nonYWeight);
    }

    /**
     * @param obs            data
     * @param betasWithBeta0 beta weights
     * @param scale          scale factor
     * @return prediction ratio but with probabilities scaled by scale factor
     */
    public static double getPredRatioScaled(SparseObservation[] obs, double[] betasWithBeta0, double scale) {
        double yWeight = 0;
        double nonYWeight = 0;
        double yProb = 0;
        double nonYProb = 0;
        for (SparseObservation o : obs) {
            double prob = scale * LRUtil.calcProb(o.getX(), betasWithBeta0);
            yWeight = yWeight + o.getY();
            yProb = yProb + o.getY() * prob;
            nonYWeight = nonYWeight + o.getWeight() - o.getY();
            nonYProb = nonYProb + ((o.getWeight() - o.getY()) * prob);
        }
        return (yWeight == 0 || nonYProb == 0) ?
            0 :
            (yProb / yWeight) / (nonYProb / nonYWeight);
    }

}
