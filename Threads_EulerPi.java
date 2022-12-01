// =================================================================
//
// File: Threads_EulerPi.java
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using Java's Threads.
//  To compile:
//   javac Threads_EulerPi.java
//  To run:
//   java Threads_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

public class Threads_EulerPi {
  private static final long MAX_N = 394_656_595L;
  private static final long N = 394_656_595L;
  private static final int THREADS = Runtime.getRuntime().availableProcessors();
  private static final long BLOCK = N / (long)THREADS;
  private static final long N2 = N * N;

  private static class EulerPi extends Thread {
    private long start, end, sum;

    public EulerPi(long start, long end) {
      this.start = start;
      this.end = end;
      this.sum = 0;
    }

    public double getSum() {
      return (double)sum / N2;
    }

    public long getLongSum() {
      return sum;
    }

    @Override
    public void run() {
      for (long i = start; i < end; i++) {
        if (i==0) continue;
        sum += N2 / (i * i);
      }
    }
  }

  public static void main(String args[]) {
    EulerPi threads[] = new EulerPi[THREADS];
    double pi;

    if (N > MAX_N) {
      System.out.printf("N must be <= %d %n", MAX_N);
      System.exit(-1);
    }

    for (int i = 0; i < THREADS; i++) {
      threads[i] = new EulerPi(i * BLOCK , (i + 1) * BLOCK);
      threads[i].start();
    }

    try {
      for (int i = 0; i < THREADS; i++) {
        threads[i].join();
      }
    } catch (InterruptedException ie) {
      System.out.printf("main: %s %n", ie);
    }


    pi = Math.sqrt((double)(6 * sum(threads))/N2);
    System.out.printf("pi = %.10f %n", pi);
  }

  private static long sum(EulerPi threads[]) {
    long sum = 0;

    for (int i = 0; i < THREADS; i++) {
      sum += threads[i].getLongSum();
    }
    return sum;
  }
}
