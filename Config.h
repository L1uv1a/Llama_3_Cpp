#pragma once
#include "cjson.h"
#include <iostream>
class Config
{
public:
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
    Config();
    Config(int max_seq_len);
    Config read_config_param();
};

