package com.appnexus.opt.concurrent;

import org.junit.Assert;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

public class MultiThreadingUtilTest {

    @Test
    public void testSplitDatasetIntoRanges() {
        // case 1
        Object[] dataset = new Integer[] {0, 1, 2, 3, 4, 5, 6, 7};
        List<DatasetRange> datasetRangeList = MultiThreadingUtil.splitDatasetIntoRanges(dataset, 3);
        List<DatasetRange> expected = new LinkedList<>();
        expected.add(new DatasetRange(0, 2, dataset));
        expected.add(new DatasetRange(2, 4, dataset));
        expected.add(new DatasetRange(4, 8, dataset));
        Assert.assertEquals(datasetRangeList, expected);
        // case 2
        Object[] dataset2 = new Integer[] {0, 1, 2, 3};
        List<DatasetRange> datasetRangeList2 = MultiThreadingUtil.splitDatasetIntoRanges(dataset2, 4);
        List<DatasetRange> expected2 = new LinkedList<>();
        expected2.add(new DatasetRange(0, 1, dataset2));
        expected2.add(new DatasetRange(1, 2, dataset2));
        expected2.add(new DatasetRange(2, 3, dataset2));
        expected2.add(new DatasetRange(3, 4, dataset2));
        Assert.assertEquals(datasetRangeList2, expected2);
    }
}
