#include "ml_engine.h"


void linear_forward(const TensorInt8* input, const LinearLayer* layer, TensorInt8* output) {
    uint16_t out_nodes = layer->weights.rows;
    uint16_t in_nodes = layer->weights.cols;

    for (uint16_t i = 0; i < out_nodes; i++) {
        int32_t acc = 0;

        for (uint16_t j = 0; j < in_nodes; j++) {
            acc = input->data[j] * layer->weights.data[i * in_nodes + j];
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

        default:
            while(1);
    }
}