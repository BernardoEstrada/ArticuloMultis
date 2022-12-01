// =================================================================
//
// File: ForkJoin_EulerPi.c
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using Java's Fork-Join.
//  To compile:
//   javac ForkJoin_EulerPi.java
//  To run:
//   java ForkJoin_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class ForkJoin_EulerPi {
  private static final long MAX_N = 394_656_595L;
  private static final long N = 394_656_595L;
  private static final int THREADS = Runtime.getRuntime().availableProcessors();
  private static final long BLOCK = N / (long)THREADS;
  private static final long N2 = N * N;

  private static class EulerPi extends RecursiveTask<Long> {
    private long start, end;

    public EulerPi(long start, long end) {
      this.start = start;
      this.end = end;
    }

    @Override
    protected Long compute() {
      long sum = 0;

      if ((end - start) <= BLOCK) {
        for (long i = start; i < end; i++) {
          if (i==0) continue;
          sum += N2 / (i * i);
        }
      } else {
        long mid = (start + end) / 2;
        EulerPi t1 = new EulerPi(start, mid);
        EulerPi t2 = new EulerPi(mid, end);
        t1.fork();
        t2.fork();
        sum = t1.join() + t2.join();
      }
      return sum;
    }
  }

  public static void main(String args[]) {
    EulerPi task;
    double pi;

    if (N > MAX_N) {
      System.out.printf("N must be <= %d %n", MAX_N);
      System.exit(-1);
    }

    task = new EulerPi(0, N);
    ForkJoinPool pool = new ForkJoinPool(THREADS);
    pi = Math.sqrt((double)(6 * pool.invoke(task))/N2);
    System.out.printf("pi = %.10f%n", pi);
  }
}