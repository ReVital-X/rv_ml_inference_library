#include "ml_engine.h"

static int8_t scratch_buf_0[MAX_ACTIVATION_SIZE];
static int8_t scratch_buf_1[MAX_ACTIVATION_SIZE];

static inline int32_t requantize(int32_t input, int32_t scale_multiply, int32_t scale_shift) {
    int64_t total = (int64_t)input * (int64_t)scale_multiply;

    if (scale_shift > 0) {
        total += (1LL << (scale_shift - 1));
    }

    return (int32_t)(total >> scale_shift);
}

static inline int32_t apply_activation(int32_t val, ActivationType act) {
    switch (act) {
        case ACTIVATION_RELU:
            return (val > 0) ? val : 0;
    
        case ACTIVATION_RELU6:
            if (val < 0) return 0;
            if (val > 6) return 6;
            return val;
        
        case ACTIVATION_NONE:
        default:
            return val;
    }
}

void linear_forward(const TensorInt8* input, const LinearLayer* layer, TensorInt8* output) {
    output->rows = input->rows;
    output->cols = layer->weights.rows;

    for (uint16_t r = 0; r < output->rows; r++) {
        for (uint16_t c = 0; c < output->cols; c++) {
            int32_t acc = layer->biases.data[c];

            for (uint16_t k = 0; k < input->cols; k++) {
                acc += input->data[r * input->cols + k] * layer->weights.data[c*layer->weights.cols + k];
            }
            

            acc = requantize(acc, layer->scale_mult, layer->scale_shift);
            acc = apply_activation(acc, layer->activation);

            if (acc > 127) acc = 127;
            if (acc < -127) acc = -127;

            output->data[r*output->cols + c] = (int8_t)acc;
        }
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
