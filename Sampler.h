#pragma once

#include <iostream>
#include "support.h"
typedef struct ProbIndex
{
	int index;
	float prob;
} ProbIndex; // Yes, what probability is this and where is it coming from? (Ex: take the number from logit and its index)

class Sampler
{
public:
	int vocab_size;
	ProbIndex* probindex; // buffer used in top-p sampling
	float temperature;
	float topp;
	unsigned long long rng_state;
	Sampler();
	Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
	~Sampler();
	int sample_argmax(float* probability, int n);
	int sample_multinomial(float* probability, int n, float random_number);
	int sample_top_p(float* probability, int n, float top_p, ProbIndex* probindex, float random_number);
	int do_the_sample(float* logit);
	void free_sampler(Sampler* sampler);
};

int compare_probindex(const void* a, const void* b);

