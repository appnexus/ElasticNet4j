package com.appnexus.opt.ml;

import org.junit.Assert;
import org.junit.Test;

public class LRTest {

    @Test
    public void testGetInitialBetasWithBeta0() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = LRUtilTest.makeTjSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = LRUtilTest.makeTjSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = LRUtilTest.makeTjSparseObservation(xi3, xv3, 0, 10);
        SparseObservation[] soArr = {so1, so2, so3};
        LR lr = new LR(soArr, 10, new double[]{}, 0, new double[]{}, new double[]{}, 0, 0, new CoordinateDescentTrainer());
        double[] actualBetas = new double[]{-0.6931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        Assert.assertArrayEquals(lr.getInitialBetasWithBeta0(null), actualBetas, 0.01);

    }

    @Test
    public void testGuessInitialBetaZero() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = LRUtilTest.makeTjSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = LRUtilTest.makeTjSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = LRUtilTest.makeTjSparseObservation(xi3, xv3, 0, 10);
        SparseObservation[] soArr = {so1, so2, so3};
        LR lr = new LR(soArr, 11, new double[]{}, 0, new double[]{}, new double[]{}, 0, 0, new CoordinateDescentTrainer());
        Assert.assertEquals(lr.guessInitialBetaZero(), -0.6931, 0.01);
    }

    @Test
    public void testGetTotalSuccesses() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = LRUtilTest.makeTjSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = LRUtilTest.makeTjSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = LRUtilTest.makeTjSparseObservation(xi3, xv3, 0, 10);
        SparseObservation[] soArr = {so1, so2, so3};
        Assert.assertEquals(LR.getTotalSuccesses(soArr), 10, 1e-10);
    }

    @Test
    public void testGetTotalWeights() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = LRUtilTest.makeTjSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = LRUtilTest.makeTjSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = LRUtilTest.makeTjSparseObservation(xi3, xv3, 0, 10);
        SparseObservation[] soArr = {so1, so2, so3};
        Assert.assertEquals(LR.getTotalWeights(soArr), 30, 1e-10);
    }
}
