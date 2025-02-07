#pragma once
#include <iostream>
#include "support.h"
#include <math.h>
#include <omp.h>
#include <chrono>
#include "type.h"
#include "cjson.h"

#define EOS_ENABLE	1
#define BOS_ENABLE	1
#define EOS_DISABLE 0
#define BOS_DISABLE 0

void softmax(float* x, int size);
void matmul(float* xout, float* x, float* weight, int n, int d);
void matmul_with_debug(float* xout, float* x, float* weight, int n, int d, int layer);
void rmsnorm(float *o, float *x, float *weight, int size);
void silu(float* x, int hidden_dim);
void elemul(float* x_out, float* x1, float* x2, int size); 

