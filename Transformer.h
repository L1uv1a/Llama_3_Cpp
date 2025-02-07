#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <omp.h>
#include "Config.h" // Ensure this is included
#include "win.h"
#include "support.h"

class TransformerWeight
{
public:
    float* token_embedding_table;
    float *rms_att_weight; 
    float *rms_ffn_weight; 
    float* att_wq;
    float* att_wk;
    float* att_wv;
    float* att_wo;
    float* ffn_w1;
    float* ffn_w3;
    float* ffn_w2;
    float* rms_final_out;  
    float *wcls; 
    TransformerWeight();
    TransformerWeight(Config *p, float *ptr, int shared_weights);
    void free_TransformerWeight();
    void read_checkpoint();
};

class RunState
{
public:
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // same, but inside a residual branch (dim,)
    float *xb2;    // an additional buffer just for convenience (dim,)
    float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;      // query (dim,)
    float *k;      // key (dim,)
    float *v;      // value (dim,)
    float *att;    // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   
    float *value_cache; 
    RunState(Config *p);
    ~RunState();
};

class Transformer
{
public:
    Config *p;         // configuration
    TransformerWeight *w; // weights
    RunState *state;   // run state
    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    Transformer(Config* p, TransformerWeight* w, RunState* state);
    ~Transformer();
    float* forward(int token_input, int pos);
    void free_transformer();
};