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

// implement euler's method to calculate pi using threads