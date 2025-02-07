#pragma once

#include <iostream>
#include <chrono>
#include "Tokenizer.h"
#include "Transformer.h"
#include "Sampler.h"
#include "support.h"
class Generate
{
public:
	Transformer* transformer;
	Tokenizer* tokenizer;
	Sampler* sampler;
	Generate();
	Generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler);
	~Generate();
	void generating(const char* prompt, int step);
	void safe_printf(char* piece);
};

