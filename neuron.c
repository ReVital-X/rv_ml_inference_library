#include <stdio.h>
#include <stdint.h>

int32_t accumulate_neuron(const int8_t* inputs, const int8_t* weights, int32_t bias, int32_t length) {
    int32_t accumulator = bias;

    for (int i = 0; i < length; i++) {
        accumulator += (int32_t)inputs[i] * (int32_t)weights[i];
    }

    return accumulator;
}


int main() {
    int8_t my_inputs[3] = {10, -5, 20};
    int8_t my_weights[3] = {2, 4, -1};
    int32_t my_bias = 5;
    int32_t result = accumulate_neuron(my_inputs, my_weights, my_bias, 3);

    printf("Neuron Output: %d\n", result);

    return 0;

}