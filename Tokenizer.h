#pragma once
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    char* str;
    int id;
} TokenIndex;

class Tokenizer
{
public:
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
    // Tokenizer();
    void create_tokenizer(Tokenizer* t, int vocab_size);

    void free_tokenizer(Tokenizer *t);
    int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
    void encode(Tokenizer *t, const char* text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
    char* decode(Tokenizer* t, int token);
};
int compare_tokens(const void *a, const void *b);
