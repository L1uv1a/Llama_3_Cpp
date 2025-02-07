#include "Generate.h"

Generate::Generate()
{
	this->transformer = NULL;
	this->tokenizer = NULL;
	this->sampler = NULL;
}

Generate::Generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler)
{
	this->transformer = &transformer;
	this->tokenizer = &tokenizer;
	this->sampler = &sampler;
}

Generate::~Generate()
{
}

void Generate::generating(const char* prompt, int step)
{
	const char* empty_prompt = "";
	if (prompt == NULL)
	{
		prompt = empty_prompt;
	}
	int num_prompt_tokens = 0;
	int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
	this->tokenizer->encode(this->tokenizer, prompt, BOS_ENABLE, EOS_DISABLE, prompt_tokens, &num_prompt_tokens);
	if (num_prompt_tokens < 1)
	{
		fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
		exit(EXIT_FAILURE);
	}

	long start = 0;
	int next;
	int token = prompt_tokens[0];
	int pos = 0;
	auto start_time = std::chrono::high_resolution_clock::now();
	while (pos < step )
	{
		float* logits = this->transformer->forward(token, pos);
		FILE* file = fopen("logits.bin", "ab");
		if (file == NULL) {
			perror("Error opening file for writing");
			return;
		}
		fwrite(logits, sizeof(float), 128256, file);
		fclose(file);
		if (pos < num_prompt_tokens - 1)
		{
			next = prompt_tokens[pos + 1];
		}
		else
		{
			next = this->sampler->do_the_sample(logits);
		}
		pos++;
		if ( (next == 128001 || next == 128009) && pos > num_prompt_tokens)
		{
			break;
		}
		if (token < 128000)
		{
			char* output_token = tokenizer->decode(this->tokenizer, token);
			safe_printf(output_token);
		}
		fflush(stdout);
		token = next;
		if (start == 0)
		{
			//start_time = std::chrono::high_resolution_clock::now();
		}
	}
	printf("\n");
	if (pos > 1)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end_time - start_time;
		std::cout << "Time spent: " << elapsed.count() << " seconds\n";
		std::cout << "Token/s: " << pos/elapsed.count() << " Tokens/s\n";
	}
}

void Generate::safe_printf(char* piece)
{
	if (piece == NULL) {
		return;
	}
	if (piece[0] == '\0') {
		return;
	}
	if (piece[1] == '\0') {
		unsigned char byte_val = piece[0];
		if (!(isprint(byte_val) || isspace(byte_val))) {
			return; // bad byte, don't print it
		}
	}
	printf("%s", piece);
}
