package com.appnexus.opt.ml;

import org.junit.Assert;
import org.junit.Test;

public class LRUtilTest {
    public static final int MT_LAMBDA_GRID_SIZE = 64;
    public static final double MT_LAMBDA_GRID_START = -5;
    public static final double MT_LAMBDA_GRID_END = 5;

    @Test
    public void testCalcProbWithSparseArrayAndBetas() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = makeTjSparseObservation(xi1, xv1, 5, 10);
        double[] betasWithBeta0 = new double[] {-1, 0, 1, 0, 0, 0};
        Assert.assertEquals(LRUtil.calcProb(so1.getX(), betasWithBeta0), 0.5, 1e-10);
    }

    @Test
    public void testCalcProbWithDotProduct() {
        Assert.assertEquals(0, LRUtil.calcProb(-Double.MAX_VALUE), 1e-10);
        Assert.assertEquals(0.5, LRUtil.calcProb(0), 1e-10);
        Assert.assertEquals(1, LRUtil.calcProb(Double.MAX_VALUE), 1e-10);
    }

    @Test
    public void testExpit() {
        Assert.assertEquals(0, LRUtil.expit(-Double.MAX_VALUE), 1e-10);
        Assert.assertEquals(0.5, LRUtil.expit(0), 1e-10);
        Assert.assertEquals(1, LRUtil.expit(Double.MAX_VALUE), 1e-10);
    }

    @Test
    public void testBetasDotXi() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = makeTjSparseObservation(xi1, xv1, 5, 10);
        double[] betasWithBeta0 = new double[] {-1, 0, 1, 0, 2, 0};
        Assert.assertEquals(LRUtil.betasDotXi(so1.getX(), betasWithBeta0), 6.0, 1e-10);
    }

    @Test
    public void hasConverged() {
        double[] oldBetas = new double[3];
        double[] newBetas = new double[3];
        double tolerance = 0.1;

        // Sum of new betas == 0.0
        Assert.assertTrue(LRUtil.hasConverged(oldBetas, newBetas, tolerance));

        // betas_diff_pct > tolerance
        newBetas[0] = 1;
        newBetas[1] = 2;
        newBetas[2] = 3;
        Assert.assertFalse(LRUtil.hasConverged(oldBetas, newBetas, tolerance));

        // betas_diff_pct == tolerance
        oldBetas[0] = 0.4;
        oldBetas[1] = 2;
        oldBetas[2] = 3;
        Assert.assertTrue(LRUtil.hasConverged(oldBetas, newBetas, tolerance));

        // betas_diff_pct < tolerance
        oldBetas[0] = 0.75;
        oldBetas[1] = 2;
        oldBetas[2] = 3;
        Assert.assertTrue(LRUtil.hasConverged(oldBetas, newBetas, tolerance));

        // betas_diff_pct > tolerance (negative values)
        oldBetas[0] = -0.75;
        oldBetas[1] = 2;
        oldBetas[2] = 3;
        Assert.assertFalse(LRUtil.hasConverged(oldBetas, newBetas, tolerance));
    }

    @Test
    public void testGenerateLambdaScaleFactors() throws Exception {
        int[] xi1 = {1, 2, 3, 4};
        double[] xv1 = {1, 2, 3, 4};
        SparseObservation so1 = makeTjSparseObservation(xi1, xv1, 5, 10);
        int[] xi2 = {6, 7, 8, 9};
        double[] xv2 = {6, 7, 8, 9};
        SparseObservation so2 = makeTjSparseObservation(xi2, xv2, 5, 10);
        int[] xi3 = {3, 4, 5, 6, 7};
        double[] xv3 = {3, 4, 5, 6, 7};
        SparseObservation so3 = makeTjSparseObservation(xi3, xv3, 0, 10);

        SparseObservation[] soArr = {so1, so2, so3};

        double[] lambdaScaleFactors = LRUtil.generateLambdaScaleFactors(soArr, 11);
        double[] expectedLambdaScaleFactors = {0.03333333333333333, 0.16666666666666666, 0.16666666666666666,
            0.16666666666666666, 0.16666666666666666, 0.03333333333333333, 0.16666666666666666, 0.16666666666666666,
            0.16666666666666666, 0.16666666666666666, 0.03333333333333333};
        Assert.assertArrayEquals(expectedLambdaScaleFactors, lambdaScaleFactors, 0.001);
    }

    @Test
    public void testGetLambdaGrid() {
        double[] expectedLambdas = {148.4131591025766, 126.63004974120119, 108.04412219523138, 92.18611510297703,
            78.65564220535094, 67.1110833103801, 57.26095899559481, 48.856571274987246, 41.68572442056974,
            35.567367400530664, 30.347022665152508, 25.892885865529063, 22.09249802996929, 18.849906176505833,
            16.083240672062946, 13.722648170944325, 11.708527943042188, 9.990027062215784, 8.523756461042604,
            7.272695434625858, 6.205256934142649, 5.2944900504700305, 4.517399552029853, 3.8543652964024084,
            3.288646856892647, 2.805960856757554, 2.3941203395414017, 2.0427270702661424, 1.7429089986334574,
            1.487096255654802, 1.2688300280258125, 1.0825994846655687, 0.9237026381080498, 0.788127627745311,
            0.6724514275370012, 0.573753420737433, 0.4895416595569535, 0.41768994794620595, 0.3563841589563572,
            0.3040764312848336, 0.2594460885514358, 0.22136629458659662, 0.18887560283756194, 0.16115368156599397,
            0.13750060194174002, 0.11731916609425083, 0.1000998289366196, 0.08540783306532164, 0.07287223191492677,
            0.06217652402211632, 0.053050661930953376, 0.04526423397858692, 0.03862064681369838, 0.03295216176670604,
            0.028115659748972045, 0.023989027752305363, 0.020468075714350477, 0.017463905906237646,
            0.014900668424247155, 0.012713646115675244, 0.010847620586711397, 0.009255478036954581,
            0.007897019720388171, 0.006737946999085467};
        Assert.assertArrayEquals(expectedLambdas,
            LRUtil.getLambdaGrid(MT_LAMBDA_GRID_SIZE, MT_LAMBDA_GRID_START, MT_LAMBDA_GRID_END), 0.00001);
    }

    @Test
    public void testGetLambdaGridNull() {
        Assert.assertArrayEquals(LRUtil.getLambdaGrid(0, 0, 0), null, 1e-10);
    }

    /**
     * HELPERS
     */
    static SparseObservation makeTjSparseObservation(int[] xIndixes, double[] xValues, int y, int weight)
        throws Exception {
        if (xIndixes == null || xValues == null || xIndixes.length != xValues.length) {
            throw new Exception("WTF Son !!!");
        }
        SparseArray x = new SparseArray();
        for (int i = 0; i < xIndixes.length; ++i) {
            x.set(xIndixes[i], xValues[i]);
        }
        return new SparseObservation(x, y, weight);
    }
}
