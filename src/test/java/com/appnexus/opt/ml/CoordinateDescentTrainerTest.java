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

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class CoordinateDescentTrainerTest {
    private static final double SPARCE_PCT = 0.1;
    private static final double TOLERANCE = 1e-6;
    private static final long COL_SEED = 8;
    private static final long BETA_SEED = 16;
    private static final long DATA_SEED = 32;
    private static final long WEIGHT_SEED = 64;

    @Test
    public void testCalculateCj2() throws Exception {
        int j = 2;
        double[][] covarianceMatrix = new double[][] {{0.0, 0.0, 1.0, 2.0, 6.0, 8.0, 5.0, 12.0, 14.0, 8.0, 9.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {2.0, 0.0, 2.0, 0.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {6.0, 0.0, 3.0, 6.0, 0.0, 24.0, 15.0, 18.0, 21.0, 0.0, 0.0},
            {8.0, 0.0, 4.0, 8.0, 24.0, 0.0, 20.0, 24.0, 28.0, 0.0, 0.0},
            {5.0, 0.0, 0.0, 0.0, 15.0, 20.0, 0.0, 30.0, 35.0, 0.0, 0.0},
            {12.0, 0.0, 0.0, 0.0, 18.0, 24.0, 30.0, 0.0, 84.0, 48.0, 54.0},
            {14.0, 0.0, 0.0, 0.0, 21.0, 28.0, 35.0, 84.0, 0.0, 56.0, 63.0},
            {8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48.0, 56.0, 0.0, 72.0},
            {9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54.0, 63.0, 72.0, 0.0}};
        double[] currentBetasWithBeta0 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        double cj_1 = 12;
        double totalWeights = 9;
        Assert.assertEquals(
            CoordinateDescentTrainer.calculateCj2(j, covarianceMatrix, currentBetasWithBeta0, cj_1, totalWeights),
            6.667, 0.01);
    }

    @Test
    public void testGetWeightedCovarianceMatrix() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = makeSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = makeSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = makeSparseObservation(xi3, xv3, 0, 10);

        SparseObservation[] soArr = {so1, so2, so3};
        double[] mi = {1, 1, 1};
        double[][] weightedCovarMatrix = CoordinateDescentTrainer.getWeightedCovarianceMatrix(11, soArr, mi);

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

        // for (double[] col : weightedCovarMatrix) {
        // System.out.println(Arrays.toString(col));
        // }

        Assert.assertArrayEquals(expectedCol0, weightedCovarMatrix[0], 0.0001);
        Assert.assertArrayEquals(expectedCol1, weightedCovarMatrix[1], 0.0001);
        Assert.assertArrayEquals(expectedCol2, weightedCovarMatrix[2], 0.0001);
        Assert.assertArrayEquals(expectedCol3, weightedCovarMatrix[3], 0.0001);
        Assert.assertArrayEquals(expectedCol4, weightedCovarMatrix[4], 0.0001);
        Assert.assertArrayEquals(expectedCol5, weightedCovarMatrix[5], 0.0001);
        Assert.assertArrayEquals(expectedCol6, weightedCovarMatrix[6], 0.0001);
        Assert.assertArrayEquals(expectedCol7, weightedCovarMatrix[7], 0.0001);
        Assert.assertArrayEquals(expectedCol8, weightedCovarMatrix[8], 0.0001);
        Assert.assertArrayEquals(expectedCol9, weightedCovarMatrix[9], 0.0001);
        Assert.assertArrayEquals(expectedCol10, weightedCovarMatrix[10], 0.0001);

    }

    /**
     * Test training on randomly generated data while assuming a set of Betas. Compare trained Betas (for lambda == 0) with the assumed set of betas
     */
    @Ignore
    @Test
    public void testLRCoordinateDescentTrainer() {
        double alpha = 1;
        double[] lambdaGrid = LRUtil.getLambdaGrid(128, 5, 36);
        int numOfUniqueObservations = 1000;
        int numOfFeatures = 200;
        int maxIterations = 1000;
        SparseObservation[] obs = LRTestUtils
            .createTestData(numOfUniqueObservations, numOfFeatures, SPARCE_PCT, COL_SEED, BETA_SEED, DATA_SEED,
                WEIGHT_SEED);
        double[] lambdaScaleFactors = LRUtil.generateLambdaScaleFactors(obs, numOfFeatures);
        LR lr = new LR(obs, numOfFeatures, null, alpha, lambdaGrid, lambdaScaleFactors, TOLERANCE, maxIterations,
            new CoordinateDescentTrainer());
        // Calculated Betas
        List<LRResult> lrResultList = lr.calculateBetas(false);

        // Test Source Of truth Betas
        double[] testBetas = LRTestUtils.makeBetas(201, BETA_SEED);
        System.out.println(Arrays.toString(testBetas));
        double[] calculatedBetasUnregularized = null;
        for (LRResult lrResult : lrResultList) {
            if (lrResult.getLambda() == 0) {
                calculatedBetasUnregularized = lrResult.getBetasWithBeta0();
                break;
            }
        }
        System.out.println(Arrays.toString(calculatedBetasUnregularized));
        Assert.assertArrayEquals(testBetas, calculatedBetasUnregularized, 0.001);
    }

    /*
        helper methods
     */

    /**
     * @param xIndices x indices
     * @param xValues  x values
     * @param y        y
     * @param weight   weight
     * @return sparse observation
     * @throws Exception
     */
    private static SparseObservation makeSparseObservation(int[] xIndices, double[] xValues, int y, int weight)
        throws Exception {
        if (xIndices == null || xValues == null || xIndices.length != xValues.length) {
            throw new Exception("array lengths for x values and indices are unequal");
        }
        SparseArray x = new SparseArray();
        for (int i = 0; i < xIndices.length; ++i) {
            x.set(xIndices[i], xValues[i]);
        }
        return new SparseObservation(x, y, weight);
    }
}
