#include "ml_engine.h"

static int8_t scratch_buf_0[MAX_ACTIVATION_SIZE];
static int8_t scratch_buf_1[MAX_ACTIVATION_SIZE];

void linear_forward(const TensorInt8* input, const LinearLayer* layer, TensorInt8* output) {
    uint16_t out_nodes = layer->weights.rows;
    uint16_t in_nodes = layer->weights.cols;

    for (uint16_t i = 0; i < out_nodes; i++) {
        int32_t acc = 0;

        for (uint16_t j = 0; j < in_nodes; j++) {
            acc += input->data[j] * layer->weights.data[i * in_nodes + j];
        }
        
        acc += layer->biases.data[i];

        int32_t scaled_acc = (acc * layer->scale_mult) >> layer->scale_shift;

        if (scaled_acc > 127) scaled_acc = 127;
        if (scaled_acc < -127) scaled_acc = -127;

        output->data[i] = (int8_t)scaled_acc;
    }

}

void layer_forward(const TensorInt8* input, const Layer* layer, TensorInt8* output) {
    switch(layer->type) {
        case LAYER_LINEAR:
            linear_forward(input, (const LinearLayer*)layer->params, output);
            break;

        default:
            while(1);
    }
}

void network_forward(const TensorInt8* input, const Layer* layers, uint16_t num_layers, TensorInt8* final_output) {
    if (num_layers == 0 || input == NULL || final_output == NULL) return;

    TensorInt8 buffers[2] = {
        { .data = scratch_buf_0, .rows = 0, .cols = 0},
        { .data = scratch_buf_1, .rows = 0, .cols = 0},
    };

    const TensorInt8* current_input = input;
    TensorInt8* current_output = &buffers[0];
    uint8_t current_buffer = 0;

    for (uint16_t i = 0; i < num_layers; i++) {
        current_output = (i == num_layers - 1) ? final_output : &buffers[current_buffer];
        layer_forward(current_input, &layers[i], current_output);
        current_input = current_output;
        current_buffer ^= 1;
    }

}
