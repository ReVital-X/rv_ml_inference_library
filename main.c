#include <stdint.h>
#include "ml_engine.h"

#define UART_TX_REGISTER (*((volatile uint8_t*)0x10000000))
#define UART_LSR_REGISTER (*((volatile uint8_t*)0x10000005))


void print_char(char c) {
    while ((UART_LSR_REGISTER & 0x20) == 0);
    UART_TX_REGISTER = c;
}

int main() {

    print_char('S'); print_char('T'); print_char('A'); print_char('R'); print_char('T'); print_char('\n');

    int in_features = 3;
    int out_features = 2;

    int8_t inputs[3] = {10, -5, 2};

    int8_t weights[6] = {
        2, 4, -1,
        1, -2, 3
    };

    int32_t bias[2] = {5, -10};

    int8_t outputs[2];

    int32_t scale_mult = 15000;
    int scale_shift = 12;

    print_char('R');
    print_char(':');
    print_char(' ');

    dense_layer_forward(inputs, weights, bias, outputs, in_features, out_features, scale_mult, scale_shift);

    print_char(outputs[0] + '0');
    print_char('\n');

    return 0;
}