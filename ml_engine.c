#include "ml_engine.h"

static int8_t requantize(int32_t acc, int32_t multiplier, int shift) {
    int64_t scaled = (int64_t)acc * (int64_t)multiplier;
    int32_t shifted = (int32_t)(scaled >> shift);

    if (shifted > 127) return 127;
    if (shifted < -128) return -128;

    return (int8_t)shifted;
}


void dense_layer_forward(
    const int8_t* inputs,
    const int8_t* weights,
    const int32_t* biases,
    int8_t* outputs,
    int in_features,
    int out_features,
    int32_t req_multiplier,
    int req_shift
) {
    for (int i = 0; i < out_features; i++) {
        int32_t accumulator = biases[i];

        for (int j = 0; j < in_features; j++) {
            accumulator += (int32_t)inputs[j] * (int32_t)weights[i * in_features + j];
        }

        outputs[i] = requantize(accumulator, req_multiplier, req_shift);
    }
}