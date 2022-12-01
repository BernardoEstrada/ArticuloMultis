// =================================================================
//
// File: OpenMP_EulerPi.c
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using Intel's TBB.
//  To compile:
//      g++ -o TBB_EulerPi TBB_EulerPi.cpp -ltbb
//  To run:
//      ./TBB_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

// implement euler's method to calculate pi using intel tbb

#include <iostream>
#include <iomanip>
#include <math.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

using namespace std;
using namespace tbb;

const long MAX_N = 394656595L;
const long N = 394656595L;
const long N2 = N*N;

class EulerPi {
private:
  long *sum;
public:
  EulerPi(long *sum) : sum(sum) { }
  void operator()(const blocked_range<long> &r) const {
    long local_sum = 0.0;
    for (long i = r.begin(); i != r.end(); i++) {
      if (i != 0)
        local_sum += N2 / (i * i);
    }
    *sum += local_sum;
  }
};

double eulerPi() {
  long sum = 0;
  parallel_for(blocked_range<long>(0, N), EulerPi(&sum));
  return sum;
}

int main(int argc, char* argv[]) {
  double pi;
  long sum = eulerPi();

  pi = sqrt((double)(6 * sum)/N2);
  cout << "pi = " << setprecision(10) << pi << endl;
  return 0;
}