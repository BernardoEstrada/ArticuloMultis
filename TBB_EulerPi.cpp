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
#include <tbb/parallel_for.h>

using namespace std;
using namespace tbb;

const long MAX_N = 394656595L;
const long N = 394656595L;
const long N2 = N*N;

double eulerPi() {
  long sum = 0;
  parallel_for(0, N, 1, [&sum](int i) {
    if (i == 0) continue;
    sum += N2 / (i * i);
  });
  return sum;
}

int main(int argc, char* argv[]) {
  double pi;
  long sum = eulerPi();

  pi = sqrt((double)(6 * sum)/N2);
  cout << "pi = " << setprecision(10) << pi << endl;
  return 0;
}