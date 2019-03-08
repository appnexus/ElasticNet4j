package com.appnexus.opt.ml;

import com.appnexus.opt.concurrent.MultiThreadingUtil;
import org.junit.Assert;
import org.junit.Test;

import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CoordinateDescentTrainerMTTest {

    @Test
    public void testGetWeightedCovarianceMatrix() throws Exception {
        int numTrainingThreads = 4;
        ExecutorService execPool = Executors.newFixedThreadPool(numTrainingThreads);
        CompletionService<Boolean> completionService = new ExecutorCompletionService<>(execPool);

        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = CoordinateDescentTrainerTest.makeSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = CoordinateDescentTrainerTest.makeSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = CoordinateDescentTrainerTest.makeSparseObservation(xi3, xv3, 0, 10);

        SparseObservation[] soArr = {so1, so2, so3};
        double[] mi = {1, 1, 1};
        CoordinateDescentTrainerMT coordinateDescentTrainerMT = new CoordinateDescentTrainerMT(completionService,
            numTrainingThreads);
        double[][] weightedCovarianceMatrix = coordinateDescentTrainerMT.getWeightedCovarianceMatrix(11, soArr, mi);

        MultiThreadingUtil.closeExecutorPool(execPool);

        double[] expectedCol0 = {0.0, 0.0, 1.0, 2.0, 6.0, 8.0, 5.0, 12.0, 14.0, 8.0, 9.0};
        double[] expectedCol1 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] expectedCol2 = {1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] expectedCol3 = {2.0, 0.0, 2.0, 0.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] expectedCol4 = {6.0, 0.0, 3.0, 6.0, 0.0, 24.0, 15.0, 18.0, 21.0, 0.0, 0.0};
        double[] expectedCol5 = {8.0, 0.0, 4.0, 8.0, 24.0, 0.0, 20.0, 24.0, 28.0, 0.0, 0.0};
        double[] expectedCol6 = {5.0, 0.0, 0.0, 0.0, 15.0, 20.0, 0.0, 30.0, 35.0, 0.0, 0.0};
        double[] expectedCol7 = {12.0, 0.0, 0.0, 0.0, 18.0, 24.0, 30.0, 0.0, 84.0, 48.0, 54.0};
        double[] expectedCol8 = {14.0, 0.0, 0.0, 0.0, 21.0, 28.0, 35.0, 84.0, 0.0, 56.0, 63.0};
        double[] expectedCol9 = {8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48.0, 56.0, 0.0, 72.0};
        double[] expectedCol10 = {9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54.0, 63.0, 72.0, 0.0};

        Assert.assertArrayEquals(expectedCol0, weightedCovarianceMatrix[0], 0.0001);
        Assert.assertArrayEquals(expectedCol1, weightedCovarianceMatrix[1], 0.0001);
        Assert.assertArrayEquals(expectedCol2, weightedCovarianceMatrix[2], 0.0001);
        Assert.assertArrayEquals(expectedCol3, weightedCovarianceMatrix[3], 0.0001);
        Assert.assertArrayEquals(expectedCol4, weightedCovarianceMatrix[4], 0.0001);
        Assert.assertArrayEquals(expectedCol5, weightedCovarianceMatrix[5], 0.0001);
        Assert.assertArrayEquals(expectedCol6, weightedCovarianceMatrix[6], 0.0001);
        Assert.assertArrayEquals(expectedCol7, weightedCovarianceMatrix[7], 0.0001);
        Assert.assertArrayEquals(expectedCol8, weightedCovarianceMatrix[8], 0.0001);
        Assert.assertArrayEquals(expectedCol9, weightedCovarianceMatrix[9], 0.0001);
        Assert.assertArrayEquals(expectedCol10, weightedCovarianceMatrix[10], 0.0001);
    }
}
