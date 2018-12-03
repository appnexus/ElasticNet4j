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

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * This object represents the trained LR model for a specific set of training parameters. It also holds a list of {@link LRIterationMetadata} representing the intermediate results of each iteration during the training process
 */
public class LRResult implements Serializable {
    private static final long serialVersionUID = 4817212848479794019L;

    private double alpha;
    private double lambda;
    private double[] betasWithBeta0;
    private int iteration;
    private double maxAbsDifferencePct;
    private double trainingEntropy;
    private long trainingTimeMillis;

    // MetaData
    private List<LRIterationMetadata> metaDataList;
    // Ephemeral fields
    private long miZiCalcMillis;
    private long ajCj1CalcMillis;
    private long weightedCovarCalcMillis;
    private long betasUpdateMillis;

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

    public List<LRIterationMetadata> getMetaDataList() {
        return metaDataList;
    }

    public void setMetaDataList(List<LRIterationMetadata> metaDataList) {
        this.metaDataList = metaDataList;
    }

    public long getTrainingTimeMillis() {
        return trainingTimeMillis;
    }

    public void setTrainingTimeMillis(long trainingTimeMillis) {
        this.trainingTimeMillis = trainingTimeMillis;
    }

    public double getTrainingEntropy() {
        return this.trainingEntropy;
    }

    public void setTrainingEntropy(double trainingEntropy) {
        this.trainingEntropy = trainingEntropy;
    }

    public static long getSerialversionuid() {
        return serialVersionUID;
    }

    public long getMiZiCalcMillis() {
        return this.miZiCalcMillis;
    }

    public void setMiZiCalcMillis(long miZiCalcMillis) {
        this.miZiCalcMillis = miZiCalcMillis;
    }

    public long getAjCj1CalcMillis() {
        return this.ajCj1CalcMillis;
    }

    public void setAjCj1CalcMillis(long ajCj1CalcMillis) {
        this.ajCj1CalcMillis = ajCj1CalcMillis;
    }

    public long getWeightedCovarCalcMillis() {
        return this.weightedCovarCalcMillis;
    }

    public void setWeightedCovarCalcMillis(long weightedCovarCalcMillis) {
        this.weightedCovarCalcMillis = weightedCovarCalcMillis;
    }

    public long getBetasUpdateMillis() {
        return this.betasUpdateMillis;
    }

    public void setBetasUpdateMillis(long betasUpdateMillis) {
        this.betasUpdateMillis = betasUpdateMillis;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("LRResult [alpha=");
        builder.append(this.alpha);
        builder.append(", lambda=");
        builder.append(this.lambda);
        builder.append(", betasWithBeta0=");
        builder.append(Arrays.toString(this.betasWithBeta0));
        builder.append(", iteration=");
        builder.append(this.iteration);
        builder.append(", maxAbsDifferencePct=");
        builder.append(this.maxAbsDifferencePct);
        builder.append(", trainingEntropy=");
        builder.append(this.trainingEntropy);
        builder.append(", trainingTimeMillis=");
        builder.append(this.trainingTimeMillis);
        builder.append(", metaDataList=");
        builder.append(this.metaDataList);
        builder.append("]");
        return builder.toString();
    }
}

