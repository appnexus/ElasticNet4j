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
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

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
        List<SparseObservation> obs = new LinkedList<>();
        obs.add(so1);
        obs.add(so2);
        obs.add(so3);
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getEntropy(obs, betasWithBeta0), 591.34, 0.01);
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
        List<SparseObservation> obs = new LinkedList<>();
        obs.add(so1);
        obs.add(so2);
        obs.add(so3);
        double[] betasWithBeta0 = LRTestUtils.makeBetas(11, 99);
        Assert.assertEquals(LREvalUtil.getEntropyNormalized(obs, betasWithBeta0), 19.7114, 0.01);
    }

    @Test
    public void testGetEntropyScaled() throws Exception {
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
        Assert.assertEquals(LREvalUtil.getEntropyScaled(soArr, betasWithBeta0, 0.01), 46.253, 0.01);
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
    public void testGetBiasScaled() throws Exception {
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
        Assert.assertEquals(LREvalUtil.getBiasScaled(soArr, betasWithBeta0, 0.5), 0.499, 0.01);
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

    @Test
    public void testGetPredRatioScaled() throws Exception {
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
        Assert.assertEquals(LREvalUtil.getPredRatioScaled(soArr, betasWithBeta0, 0.5), 0.999, 0.01);
    }
}
