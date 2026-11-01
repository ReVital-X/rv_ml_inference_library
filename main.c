#include <stdint.h>
#include "ml_engine.h"

#define UART_TX_REGISTER *((volatile char*)0x10000000)

void print_char(char c) {
    UART_TX_REGISTER = c;
}

void print_int(int32_t val) {
    if (val == 0) {
        print_char('0');
        return;
    }

    if (val < 0) {
        print_char('-');
        val = -val;
    }

    char buffer[12];
    int idx = 0;
    while (val > 0) {
        buffer[idx++] = (val % 10) + '0';
        val /= 10;
    }
    
    for (int i = idx - 1; i >= 0; i--) {
        print_char(buffer[i]);
    }
}

int main() {
    print_char('S'); print_char('T'); print_char('A'); 
    print_char('R'); print_char('T'); print_char('\n');

    int8_t input_data[2]  = {4, 2};
    int8_t weight_data[2] = {2, 1}; 
    int32_t bias_data[1]  = {0};
    int8_t output_data[1] = {0};

    TensorInt8 input_tensor  = {input_data, 1, 2};
    TensorInt8 output_tensor = {output_data, 1, 1};

    LinearLayer my_dense = {
        .weights = {weight_data, 1, 2},
        .biases = {bias_data, 1},
        .scale_mult = 1,   
        .scale_shift = 0   
    };

    Layer my_layer = {
        .type = LAYER_LINEAR,
        .params = &my_dense
    };

    layer_forward(&input_tensor, &my_layer, &output_tensor);

    print_char('R'); print_char(':'); print_char(' ');
    print_int(output_tensor.data[0]);
    print_char('\n');

    return 0;
}