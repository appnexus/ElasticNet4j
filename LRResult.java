package com.appnexus.opt.ml;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

public class LRResult implements Serializable {
    private static final long serialVersionUID = 4817212848479794019L;

    private double alpha;
    private double lambda;
    private double[] betasWithBeta0;
    private int iteration;
    private double maxAbsDifferencePct;
    private long trainingTimeMillis;

    // MetaData
    private List<LRIterationMetaData> metaDataList;

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double[] getBetasWithBeta0() {
        return betasWithBeta0;
    }

    public void setBetasWithBeta0(double[] betasWithBeta0) {
        this.betasWithBeta0 = betasWithBeta0;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    public double getMaxAbsDifferencePct() {
        return maxAbsDifferencePct;
    }

    public void setMaxAbsDifferencePct(double maxAbsDifferencePct) {
        this.maxAbsDifferencePct = maxAbsDifferencePct;
    }

    public List<LRIterationMetaData> getMetaDataList() {
        return metaDataList;
    }

    public void setMetaDataList(List<LRIterationMetaData> metaDataList) {
        this.metaDataList = metaDataList;
    }

    public long getTrainingTimeMillis() {
        return trainingTimeMillis;
    }

    public void setTrainingTimeMillis(long trainingTimeMillis) {
        this.trainingTimeMillis = trainingTimeMillis;
    }

    public static long getSerialversionuid() {
        return serialVersionUID;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LRResult [alpha=");
        builder.append(alpha);
        builder.append(", lambda=");
        builder.append(lambda);
        builder.append(", betasWithBeta0=");
        builder.append(Arrays.toString(betasWithBeta0));
        builder.append(", iteration=");
        builder.append(iteration);
        builder.append(", maxAbsDifferencePct=");
        builder.append(maxAbsDifferencePct);
        builder.append(", trainingTimeMillis=");
        builder.append(trainingTimeMillis);
        builder.append("]");
        return builder.toString();
    }

}
