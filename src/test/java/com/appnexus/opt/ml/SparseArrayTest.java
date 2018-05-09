package com.appnexus.opt.ml;

import org.junit.Assert;
import org.junit.Test;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class SparseArrayTest {

    @Test
    public void testSet() {
        SparseArray sparseArray = new SparseArray();
        boolean setOne = sparseArray.set(1, 1);
        boolean setTwo = sparseArray.set(1, 2);
        boolean setThree = sparseArray.set(0, 3);
        boolean setZero = sparseArray.set(0, 0);
        Assert.assertTrue(setOne);
        Assert.assertFalse(setTwo);
        Assert.assertTrue(setThree);
        Assert.assertFalse(setZero);
        List<SparseArray.Entry> actualEntries = new LinkedList<>();
        actualEntries.add(new SparseArray.Entry(1, 2));
        Iterator<SparseArray.Entry> itSparseArray = sparseArray.iterator();
        Iterator<SparseArray.Entry> itActualEntries = actualEntries.iterator();
        testSparseArrayEquality(itSparseArray, itActualEntries);
    }

    @Test
    public void testAppend() {
        SparseArray sparseArray = new SparseArray();
        sparseArray.append(0, 1);
        sparseArray.append(0, 2);
        List<SparseArray.Entry> actualEntries = new LinkedList<>();
        actualEntries.add(new SparseArray.Entry(0, 1));
        actualEntries.add(new SparseArray.Entry(0, 2));
        Iterator<SparseArray.Entry> itSparseArray = sparseArray.iterator();
        Iterator<SparseArray.Entry> itActualEntries = actualEntries.iterator();
        testSparseArrayEquality(itSparseArray, itActualEntries);
    }

    private void testSparseArrayEquality(Iterator<SparseArray.Entry> itSparseArray,
        Iterator<SparseArray.Entry> itActualEntries) {
        while(itSparseArray.hasNext()) {
            SparseArray.Entry eSparseArray = itSparseArray.next();
            SparseArray.Entry eActualEntries = itActualEntries.next();
            Assert.assertEquals(eSparseArray.i, eActualEntries.i);
            Assert.assertEquals(eSparseArray.x, eActualEntries.x, 1e-10);
        }
        Assert.assertEquals(itActualEntries.hasNext(), false);
    }
}
