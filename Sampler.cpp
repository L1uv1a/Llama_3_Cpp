#include "Sampler.h"

Sampler::Sampler()
{
	this->vocab_size = 0;
	this->probindex = nullptr;
	this->temperature = 1.0f;
	this->topp = 0.0f;
	this->rng_state = 0;
}

Sampler::Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
	this->vocab_size = vocab_size;
	this->temperature = temperature;
	this->topp = topp;
	this->rng_state = rng_seed;
	this->probindex = (ProbIndex*) malloc(this->vocab_size * sizeof(ProbIndex));
	if (this->probindex == nullptr) {
		// Handle allocation failure
		throw std::bad_alloc();
	}
}

Sampler::~Sampler()
{

}

int Sampler::sample_argmax(float* probability, int n)
{
	int max_index = 0;
	float max_prob = probability[0];
	for (int i = 1; i < n; i++) {
		if (probability[i] > max_prob) {
			max_index = i;
			max_prob = probability[i];
		}
	}
	return max_index;
}

int Sampler::sample_multinomial(float* probability, int n, float random_number)
{
	float cdf = 0.0f;
	for (int i = 0; i < n; i++)
	{
		cdf += probability[i];
		if (random_number < cdf)
		{
			return i;
		}
	}
	return n-1;
}

int Sampler::sample_top_p(float* probability, int n, float top_p, ProbIndex* probindex, float random_number)
{
	int num_selected = 0;
	const float cutoff = (1.0f - top_p) / (n - 1);
	for (int i = 0; i < n; i++)
	{
		if (probability[i] > cutoff)
		{
			probindex[num_selected].index = i;
			probindex[num_selected].prob = probability[i];
			num_selected++;
		}
	}
	qsort(probindex, num_selected, sizeof(ProbIndex), compare_probindex);
	float cumulative_prob = 0.0f;
	int last_index = num_selected - 1;
	for (int i = 0; i < num_selected; i++)
	{
		cumulative_prob += probindex[last_index - i].prob;
		if (top_p < cumulative_prob)
		{
			last_index = i;
			break;
		}
	}
	float r = random_number * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i <= last_index; i++) {
		cdf += probindex[i].prob;
		if (r < cdf) {
			return probindex[i].index;
		}
	}
	return probindex[last_index].index;
}

int Sampler::do_the_sample(float* logit)
{
	int next;
	if (this->temperature == 0.0f)
	{
		next = sample_argmax(logit, this->vocab_size);
	}
	else
	{
		for (int q = 0; q < this->vocab_size; q++)
		{
			logit[q] /= this->temperature;
		}
		softmax(logit, this->vocab_size);
		float random_number = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if (this->topp <= 0.0f || this->topp >= 1)
		{
			next = sample_multinomial(logit, this->vocab_size, random_number);
		}
		else
		{
			next = sample_top_p(logit, this->vocab_size, this->topp, this->probindex, random_number);
		}
	}
	return next;
}

void Sampler::free_sampler(Sampler* sampler)
{
	if (sampler->probindex != nullptr)
	{
		free(sampler->probindex);
		sampler->probindex = nullptr;
	}
}

int compare_probindex(const void* a, const void* b)
{
	ProbIndex* pa = (ProbIndex*)a;
	ProbIndex* pb = (ProbIndex*)b;
	if (pa->prob < pb->prob)
	{
		return 1;
	}
	else if (pa->prob > pb->prob)
	{
		return -1;
	}
	else
	{
		return 0;
	}
}
