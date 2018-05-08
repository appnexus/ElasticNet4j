package com.appnexus.opt.ml;

import org.junit.Assert;
import org.junit.Test;

public class LREvalUtilTest {

    @Test
    public void testGetEntropy() throws Exception {
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
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getEntropy(soArr, betasWithBeta0), 591.34, 0.01);
    }

    @Test
    public void testGetEntropyNormalized() throws Exception {
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
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getEntropyNormalized(soArr, betasWithBeta0), 19.7114, 0.01);
    }

    @Test
    public void testGetBias() throws Exception {
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
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getBias(soArr, betasWithBeta0), 1.999, 0.01);
    }

    @Test
    public void testGetPredRatio() throws Exception {
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
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getPredRatio(soArr, betasWithBeta0), 0.999, 0.01);
    }
}
