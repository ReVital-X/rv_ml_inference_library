#include <stdint.h>
#include <stdio.h>
#include "ml_engine.h"
#include "unity.h"

void setUp(void) {

}

void tearDown(void) {

}

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

    TEST_ASSERT_EQUAL_INT8_MESSAGE(10, outputs.data[0], "[FAIL] Linear MAC Assertion failed.");
}

void test_ping_pong_multi_layer_execution(void) {
    int8_t input_data[2]  = {4, 2};
    int8_t w1_data[2]     = {2, 1};
    int32_t b1_data[1]    = {0};
    
    int8_t w2_data[1]     = {3};
    int32_t b2_data[1]    = {5};
    int8_t output_data[1] = {0};

    TensorInt8 inputs  = { .data = input_data, .rows = 1, .cols = 2 };
    TensorInt8 outputs = { .data = output_data, .rows = 1, .cols = 1 };

    // Layer 1: MAC = (4*2 + 2*1) + 0 = 10
    TensorInt8 t_w1 = { .data = w1_data, .rows = 1, .cols = 2 };
    TensorInt32 t_b1 = { .data = b1_data, .length = 1 };
    LinearLayer l1_params = { .weights = t_w1, .biases = t_b1, .scale_mult = 1, .scale_shift = 0 };
    Layer layer1 = { .type = LAYER_LINEAR, .params = &l1_params };

    // Layer 2: MAC = (10*3) + 5 = 35
    TensorInt8 t_w2 = { .data = w2_data, .rows = 1, .cols = 1 };
    TensorInt32 t_b2 = { .data = b2_data, .length = 1 };
    LinearLayer l2_params = { .weights = t_w2, .biases = t_b2, .scale_mult = 1, .scale_shift = 0 };
    Layer layer2 = { .type = LAYER_LINEAR, .params = &l2_params };

    Layer model[] = { layer1, layer2 };

    network_forward(&inputs, model, 2, &outputs);

    TEST_ASSERT_EQUAL_INT8_MESSAGE(35, outputs.data[0], "[FAIL]: Two buffer network forward layer failed unit test.");
}


int main() {
    UNITY_BEGIN();
    RUN_TEST(test_linear_layer);
    RUN_TEST(test_ping_pong_multi_layer_execution);
    return UNITY_END();
}