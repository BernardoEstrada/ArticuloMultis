// =================================================================
//
// File: CUDA_EulerPi.c
// Author: Bernardo Estrada
// Description: This file contains the code to calculate Pi
//              with Euler's method using Nvidia's CUDA.
//  To compile:
//      nvcc -o CUDA_EulerPi CUDA_EulerPi.cu
//  To run:
//      ./CUDA_EulerPi
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

// implement euler's method to calculate pi using threads