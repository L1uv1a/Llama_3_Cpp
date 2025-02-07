#include <iostream>
#include <stdio.h>
#include <fstream>
#include <omp.h>
#include "cjson.h"
#include "support.h"
#include "Config.h"
#include "Tokenizer.h"
#include "Transformer.h"
#include "Sampler.h"
#include "Generate.h"

#define VOCAB_SIZE      128256

unsigned long matmul_call_count = 0;
double total_matmul_time = 0.0;

int main() {
    omp_set_num_threads(1);
    Config config(2048);
	float temperature = 0.0f;
	float topp = 1.0f;
	unsigned long long rng_seed = 0;
	srand(rng_seed);
    // config = config.read_config_param();
    //char* tokenizer_path = "D:\\Project\\LLM\\llama2c\\llama2.c\\tokenizer.bin";
    Tokenizer tokenizer;
    tokenizer.create_tokenizer(&tokenizer, VOCAB_SIZE);
    // for (int i = 0; i < VOCAB_SIZE ; i++) {
    //     //printf("%c", t.byte_pieces[i * 2]);
    //     printf("%f", t.vocab_scores[i]);
    //     printf("%s  ", t.vocab[i]);
    // }
    const char* prompt = "Absolute Cinema";
    //int num_prompt_tokens = 0;
    //int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    //t.encode(&t, prompt, 1, 1, prompt_tokens, &num_prompt_tokens);
    //for (int i = 0; i < num_prompt_tokens; i++) {
    //    printf("%d ", prompt_tokens[i]);
    //}

    // Initialize RunState
    TransformerWeight weight;
    RunState state(&config);
    Transformer transformer(&config, &weight, &state);
    //print_w3_from_transformer(transformer, "ffn_w3_weights_layer10_onward.txt");
    //float* output_token = (float*)malloc(sizeof(float));
    //transformer.forward(prompt_tokens, output_token, 0);
	Sampler sampler(tokenizer.vocab_size, temperature, topp, rng_seed);
	Generate generate(transformer, tokenizer, sampler);
    //const char* output_file = "my_weights.bin";
    //write_weights_to_file(output_file, transformer.w, transformer.p);
	generate.generating(prompt, 10);
    tokenizer.free_tokenizer(&tokenizer);
    weight.free_TransformerWeight();
	transformer.free_transformer();
	sampler.free_sampler(&sampler);
    std::cout << "Total time: " << total_matmul_time << "s " << matmul_call_count << "\n";
    return 0;
}

