#include <stdint.h>
#include <assert.h>
#include "ml_engine.h"

void test_linear_layer() {

    TensorInt8 weights = {
        .data = {2, 1},
        .cols = 2,
        .rows = 1
    };

    TensorInt32 biases = {
        .data = {0},
        .length = 1
    };

    TensorInt8 inputs = {
        .data = {4, 2},
        .rows = 1,
        .cols = 2
    };

    TensorInt8 outputs = {
        .data = {0},
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