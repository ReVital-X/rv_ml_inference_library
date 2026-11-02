#ifndef ML_HEADER_H
#define ML_HEADER_H

#include <stdint.h>
#include <stddef.h>

#ifndef MAX_ACTIVATION_SIZE
#define MAX_ACTIVATION_SIZE 256
#endif

typedef enum {
    LAYER_LINEAR
} LayerType;

typedef enum {
    ACTIVATION_NONE = 0,
    ACTIVATION_RELU,
    ACTIVATION_RELU6
} ActivationType;

typedef struct {
    int8_t* data;
    uint16_t rows;
    uint16_t cols;
} TensorInt8;

typedef struct {
    int32_t* data;
    uint16_t length;
} TensorInt32;

typedef struct {
    TensorInt8 weights;
    TensorInt32 biases;
    int32_t scale_mult;
    int32_t scale_shift;
    ActivationType activation;
} LinearLayer;

typedef struct {
    LayerType type;
    const void* params;
} Layer;

void network_forward(const TensorInt8* input, const Layer* layers, uint16_t num_layers, TensorInt8* final_output);
void layer_forward(const TensorInt8* input, const Layer* layer, TensorInt8* output);
void linear_forward(const TensorInt8* input, const LinearLayer* layer, TensorInt8* output);

#endif