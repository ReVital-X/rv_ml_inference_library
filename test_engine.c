#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include "ml_engine.h"

void test_linear_layer() {
    int8_t weight_data[2]  = {2, 1};
    int32_t bias_data[1]   = {0};
    int8_t input_data[2]   = {4, 2};
    int8_t output_data[1]  = {0};

    TensorInt8 weights = {
        .data = weight_data,
        .cols = 2,
        .rows = 1
    };

    TensorInt32 biases = {
        .data = bias_data,
        .length = 1
    };

    TensorInt8 inputs = {
        .data = input_data,
        .rows = 1,
        .cols = 2
    };

    TensorInt8 outputs = {
        .data = output_data,
        .rows = 1,
        .cols = 1
    };

    LinearLayer dense_layer = {
        .weights = weights,
        .biases = biases,
        .scale_mult = 1,
        .scale_shift = 0
    };

    Layer new_layer = {
        .type = LAYER_LINEAR,
        .params = &dense_layer
    };

    layer_forward(&inputs, &new_layer, &outputs);

    assert(outputs.data[0] == 10);
    printf("[PASS] test_linear_layer passed successfully!\n");
}

int main() {
    printf("Running ML Engine Host Tests...\n");
    test_linear_layer();
    printf("All test");

    return 0;
}