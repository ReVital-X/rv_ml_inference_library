#ifndef ML_HEADER_H
#define ML_HEADER_H

#include <stdint.h>

typedef struct {
    int8_t* data;
    uint16_t rows;
    uint16_t cols;
} TensorInt8;

typedef struct {
    int32_t* data;
    uint16_t length;
} TensorInt32;

void dense_layer_forward(
    const int8_t* inputs,
    const int8_t* weights,
    const int32_t* biases,
    int8_t* outputs,
    int in_features,
    int out_features,
    int32_t req_multiplier,
    int req_shift
);

#endif