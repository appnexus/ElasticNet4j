package com.appnexus.opt.ml;

package com.appnexus.opt.ml;

import com.appnexus.opt.tj.TJConstants;
import com.appnexus.opt.tj.feh.objects.TJSparseObservation;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;

public class LRModelEvalUtilMT {

    private static final double EPS = 1e-15;

    // TODO, VVAL-258: make chosenLambda optional?

    /**
     * @param obs               sparse observations
     * @param lrResults         logistic regression results
     * @param testShard         test shard
     * @param chosenLambda      chosen lambda
     * @param completionService completion service
     * @param numThreads        number of threads
     * @return model error metrics for every logistic regression result
     */
    public static Map<LRResult, ModelErrorMetrics> populateModelErrorMetrics(TJSparseObservation[] obs,
        List<LRResult> lrResults, int testShard, double chosenLambda, CompletionService<Boolean> completionService,
        int numThreads) {
        Map<LRResult, ModelErrorMetrics> modelErrorMetricsMap = new LinkedHashMap<>();

        /*
            process intermediate data
         */

        List<DatasetRange> datasetRanges = MultiThreadingUtil.splitDatasetIntoRanges(obs, numThreads);
        List<Map<LRResult, IntermediateLREvalUtilData>> intermediateTrainedModelData = new LinkedList<>();
        for (int i = 0; i < datasetRanges.size(); ++i) {
            intermediateTrainedModelData.add(new HashMap<>());
            LREvalTask lrEvalTask = new LREvalTask(datasetRanges.get(i), lrResults,
                intermediateTrainedModelData.get(i));
            MultiThreadingUtil.submitTask(completionService, lrEvalTask);
        }

        MultiThreadingUtil.waitForThreadCompletion(completionService, numThreads);

        /*
            reduce: intermediate data -> list of trained models
         */

        for (LRResult lrResult : lrResults) {
            double errorForEntropyNormalized = 0;
            double totalImpsForEntropyNormalized = 0;
            double predForBias = 0;
            double actualForBias = 0;
            double clickWeightForPredRatio = 0;
            double nonClickWeightForPredRatio = 0;
            double clickProbForPredRatio = 0;
            double nonClickProbForPredRatio = 0;
            double actualClicksForScaleFactor = 0;
            double predClicksForScaleFactor = 0;
            for (Map<LRResult, IntermediateLREvalUtilData> intermediateLREvalUtilDataMap : intermediateTrainedModelData) {
                IntermediateLREvalUtilData intermediateLREvalUtilData = intermediateLREvalUtilDataMap.get(lrResult);
                errorForEntropyNormalized += intermediateLREvalUtilData.getErrorForEntropyNormalized();
                totalImpsForEntropyNormalized += intermediateLREvalUtilData.getTotalImpsForEntropyNormalized();
                predForBias += intermediateLREvalUtilData.getPredForBias();
                actualForBias += intermediateLREvalUtilData.getActualForBias();
                clickWeightForPredRatio += intermediateLREvalUtilData.getClickWeightForPredRatio();
                nonClickWeightForPredRatio += intermediateLREvalUtilData.getNonClickWeightForPredRatio();
                clickProbForPredRatio += intermediateLREvalUtilData.getClickProbForPredRatio();
                nonClickProbForPredRatio += intermediateLREvalUtilData.getNonClickProbForPredRatio();
                actualClicksForScaleFactor += intermediateLREvalUtilData.getActualClicksForScaleFactor();
                predClicksForScaleFactor += intermediateLREvalUtilData.getPredClicksForScaleFactor();
            }
            // compute entropyNormalized
            double entropyNormalized = errorForEntropyNormalized / totalImpsForEntropyNormalized;
            // compute bias
            double bias = actualForBias == 0 ? Double.MAX_VALUE : (predForBias - actualForBias) / actualForBias;
            // compute predRatio
            double predRatio = (clickWeightForPredRatio == 0 || nonClickProbForPredRatio == 0) ?
                0 :
                (clickProbForPredRatio / clickWeightForPredRatio) / (nonClickProbForPredRatio
                    / nonClickWeightForPredRatio);
            // compute scaleFactor
            double scaleFactor =
                predClicksForScaleFactor == 0 ? 0 : actualClicksForScaleFactor / predClicksForScaleFactor;
            // incorporate chosenModel
            boolean chosenModel = testShard == TJConstants.MT_NO_TEST_SHARD_ID && lrResult.getLambda() == chosenLambda;
            // make model error metrics
            ModelErrorMetrics modelErrorMetrics = new ModelErrorMetrics(scaleFactor, predRatio, bias, entropyNormalized,
                chosenModel);
            modelErrorMetricsMap.put(lrResult, modelErrorMetrics);
        }

        return modelErrorMetricsMap;
    }

    static class LREvalTask implements Callable<Boolean> {
        private DatasetRange datasetRange;
        private List<LRResult> lrResults;
        private Map<LRResult, IntermediateLREvalUtilData> lrResultIntermediateLREvalUtilDataMap;

        LREvalTask(DatasetRange datasetRange, List<LRResult> lrResults,
            Map<LRResult, IntermediateLREvalUtilData> lrResultIntermediateLREvalUtilDataMap) {
            this.datasetRange = datasetRange;
            this.lrResults = lrResults;
            this.lrResultIntermediateLREvalUtilDataMap = lrResultIntermediateLREvalUtilDataMap;
        }

        @Override
        public Boolean call() {
            for (LRResult lrResult : this.lrResults) {
                double errorForEntropyNormalized = 0;
                double totalImpsForEntropyNormalized = 0;
                double predForBias = 0;
                double actualForBias = 0;
                double clickWeightForPredRatio = 0;
                double nonClickWeightForPredRatio = 0;
                double clickProbForPredRatio = 0;
                double nonClickProbForPredRatio = 0;
                double actualClicksForScaleFactor = 0;
                double predClicksForScaleFactor = 0;
                for (int i = this.datasetRange.getStartIdx(); i < this.datasetRange.getEndIdx(); ++i) { // subset of obs
                    TJSparseObservation o = (TJSparseObservation) this.datasetRange.getDataset()[i];
                    // ENTROPY NORMALIZED
                    double prob = LRUtil.calcProb(o.getX(), lrResult.getBetasWithBeta0());
                    double pred = Math.min(1.0 - EPS, Math.max(EPS, prob));
                    double errorObs =
                        -1 * o.getY() * Math.log(pred) - (o.getWeight() - o.getY()) * Math.log(1.0 - pred);
                    errorForEntropyNormalized += errorObs;
                    totalImpsForEntropyNormalized += o.getWeight();
                    // BIAS
                    predForBias += prob * o.getWeight();
                    actualForBias += o.getY();
                    // PRED RATIO
                    clickWeightForPredRatio += o.getY();
                    clickProbForPredRatio += o.getY() * prob;
                    nonClickWeightForPredRatio += o.getWeight() - o.getY();
                    nonClickProbForPredRatio += (o.getWeight() - o.getY()) * prob;
                    // SCALE FACTORS
                    actualClicksForScaleFactor += o.getY();
                    predClicksForScaleFactor += o.getCadenceModifierSum() * prob;
                }
                // create IntermediateLREvalUtilData object
                IntermediateLREvalUtilData intermediateLREvalUtilData = new IntermediateLREvalUtilData(
                    errorForEntropyNormalized, totalImpsForEntropyNormalized, predForBias, actualForBias,
                    clickWeightForPredRatio, clickProbForPredRatio, nonClickWeightForPredRatio,
                    nonClickProbForPredRatio, actualClicksForScaleFactor, predClicksForScaleFactor);
                this.lrResultIntermediateLREvalUtilDataMap.put(lrResult, intermediateLREvalUtilData);
            }
            return true;
        }
    }
}


class IntermediateLREvalUtilData {
    private double errorForEntropyNormalized;
    private double totalImpsForEntropyNormalized;
    private double predForBias;
    private double actualForBias;
    private double clickWeightForPredRatio;
    private double nonClickWeightForPredRatio;
    private double clickProbForPredRatio;
    private double nonClickProbForPredRatio;
    private double actualClicksForScaleFactor;
    private double predClicksForScaleFactor;

    IntermediateLREvalUtilData(double errorForEntropyNormalized, double totalImpsForEntropyNormalized,
        double predForBias, double actualForBias, double clickWeightForPredRatio, double nonClickWeightForPredRatio,
        double clickProbForPredRatio, double nonClickProbForPredRatio, double actualClicksForScaleFactor,
        double predClicksForScaleFactor) {
        this.errorForEntropyNormalized = errorForEntropyNormalized;
        this.totalImpsForEntropyNormalized = totalImpsForEntropyNormalized;
        this.predForBias = predForBias;
        this.actualForBias = actualForBias;
        this.clickWeightForPredRatio = clickWeightForPredRatio;
        this.nonClickWeightForPredRatio = nonClickWeightForPredRatio;
        this.clickProbForPredRatio = clickProbForPredRatio;
        this.nonClickProbForPredRatio = nonClickProbForPredRatio;
        this.actualClicksForScaleFactor = actualClicksForScaleFactor;
        this.predClicksForScaleFactor = predClicksForScaleFactor;
    }

    double getErrorForEntropyNormalized() {
        return this.errorForEntropyNormalized;
    }

    double getTotalImpsForEntropyNormalized() {
        return this.totalImpsForEntropyNormalized;
    }

    double getPredForBias() {
        return this.predForBias;
    }

    double getActualForBias() {
        return this.actualForBias;
    }

    double getClickWeightForPredRatio() {
        return this.clickWeightForPredRatio;
    }

    double getNonClickWeightForPredRatio() {
        return this.nonClickWeightForPredRatio;
    }

    double getClickProbForPredRatio() {
        return this.clickProbForPredRatio;
    }

    double getNonClickProbForPredRatio() {
        return this.nonClickProbForPredRatio;
    }

    double getActualClicksForScaleFactor() {
        return this.actualClicksForScaleFactor;
    }

    double getPredClicksForScaleFactor() {
        return this.predClicksForScaleFactor;
    }
}

