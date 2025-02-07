#include "Transformer.h"
#include "Config.h" // Ensure this is included
#include "win.h"

TransformerWeight::TransformerWeight()
{
    this->token_embedding_table = NULL;
    this->rms_att_weight = NULL;
    this->rms_ffn_weight = NULL;
    this->att_wq = NULL;
    this->att_wk = NULL;
    this->att_wv = NULL;
    this->att_wo = NULL;
    this->ffn_w1 = NULL;
    this->ffn_w2 = NULL;
    this->ffn_w3 = NULL;
    this->rms_final_out = NULL;
    this->wcls = NULL;
}

TransformerWeight::TransformerWeight(Config *p, float *ptr, int shared_weights)
{
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    this->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    this->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    
    this->att_wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    this->att_wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    this->att_wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    this->att_wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;

    this->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;

    this->ffn_w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    this->ffn_w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    this->ffn_w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;

    this->rms_final_out = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_img (for RoPE)
    this->wcls = shared_weights ? this->token_embedding_table : ptr;
}

void TransformerWeight::free_TransformerWeight()
{
    //if (this->token_embedding_table) {
    //    free(this->token_embedding_table);
    //    this->token_embedding_table = NULL;
    //}
    //if (this->rms_att_weight) {
    //    free(this->rms_att_weight);
    //    this->rms_att_weight = NULL;
    //}
    //if (this->att_wq) {
    //    free(this->att_wq);
    //    this->att_wq = NULL;
    //}
    //if (this->att_wk) {
    //    free(this->att_wk);
    //    this->att_wk = NULL;
    //}
    //if (this->att_wv) {
    //    free(this->att_wv);
    //    this->att_wv = NULL;
    //}
    //if (this->att_wo) {
    //    free(this->att_wo);
    //    this->att_wo = NULL;
    //}
    //if (this->ffn_w1) {
    //    free(this->ffn_w1);
    //    this->ffn_w1 = NULL;
    //}
    //if (this->ffn_w2) {
    //    free(this->ffn_w2);
    //    this->ffn_w2  = NULL;
    //}
    //if (this->ffn_w3) {
    //    free(this->ffn_w3) ;
    //    this->ffn_w3  = NULL;
    //}
    //if (this->rms_final_out) {
    //    free(this->rms_final_out);
    //    this->rms_final_out = NULL;
    //}
    this->token_embedding_table = NULL;
    this->rms_att_weight = NULL;
    this->rms_ffn_weight = NULL;
    this->att_wq = NULL;
    this->att_wk = NULL;
    this->att_wv = NULL;
    this->att_wo = NULL;
    this->ffn_w1 = NULL;
    this->ffn_w2 = NULL;
    this->ffn_w3 = NULL;
    this->rms_final_out = NULL;
    this->wcls = NULL;
}

RunState::RunState(Config *p)
{
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    this->x = (float *)calloc(p->dim, sizeof(float));
    this->xb = (float *)calloc(p->dim, sizeof(float));
    this->xb2 = (float *)calloc(p->dim, sizeof(float));
    this->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    this->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    this->q = (float *)calloc(p->dim, sizeof(float));
    this->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    this->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    this->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    this->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!this->x || !this->xb || !this->xb2 || !this->hb || !this->hb2 || !this->q || !this->key_cache || !this->value_cache || !this->att || !this->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

RunState::~RunState()
{
    if (this->x) {
        free(this->x);
        this->x = NULL;
    }
    if (this->xb) {
        free(this->xb);
        this->xb = NULL;
    }
    if (this->xb2) {
        free(this->xb2);
        this->xb2 = NULL;
    }
    if (this->hb) {
        free(this->hb);
        this->hb = NULL;
    }
    if (this->hb2) {
        free(this->hb2);
        this->hb2 = NULL;
    }
    if (this->q) {
        free(this->q);
        this->q = NULL;
    }
    if (this->att) {
        free(this->att);
        this->att = NULL;
    }
    if (this->logits) {
        free(this->logits);
        this->logits = NULL;
    }
    if (this->key_cache) {
        free(this->key_cache);
        this->key_cache = NULL;
    }
    if (this->value_cache) {
        free(this->value_cache);
        this->value_cache = NULL;
    }
}

Transformer::Transformer(Config* p, TransformerWeight* w, RunState* state)
{
    HANDLE file = CreateFile(L"C:\\Users\\DELL\\.llama\\checkpoints\\llama3_1b.bin", GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Couldn't open file %s\n", "C:\\Users\\DELL\\.llama\\checkpoints\\llama3_1b.bin");
        exit(EXIT_FAILURE);
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(file, &fileSize)) {
        fprintf(stderr, "Couldn't get file size\n");
        CloseHandle(file);
        exit(EXIT_FAILURE);
    }
    file_size = fileSize.QuadPart;

    HANDLE fileMapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (fileMapping == NULL) {
        fprintf(stderr, "Couldn't create file mapping\n");
        CloseHandle(file);
        exit(EXIT_FAILURE);
    }

    this->data = (float*)MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
    if (data == NULL) {
        fprintf(stderr, "Couldn't map view of file\n");
        CloseHandle(fileMapping);
        CloseHandle(file);
        exit(EXIT_FAILURE);
    }
    int a = sizeof(Config);
    int b = sizeof(float);
    float *weights_ptr = this->data + sizeof(Config) / sizeof(float);
    // Initialize weights
    *w = TransformerWeight(p, weights_ptr, 1);
    this->p = p;
    this->w = w;
    this->state = state;
    CloseHandle(fileMapping);
    CloseHandle(file);
}

Transformer::~Transformer()
{
}

void Transformer::free_transformer()
{
    if (this->data) {
        UnmapViewOfFile(this->data);
        this->data = NULL;
    }
    this->w = NULL;
    
    this->p = NULL;
    this->state->~RunState();
    this->state = NULL;
}

float* Transformer::forward(int token_input, int pos)
{
    Config* p = this->p;
    TransformerWeight *w = this->w;
    RunState *s = this->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float* token_param = w->token_embedding_table + token_input * dim;
    int size_x = sizeof(*x);
    memcpy(x, token_param, dim * sizeof(*x));

    for (long layer = 0; layer < p->n_layers; layer++)
    {
        rmsnorm(s->xb, x, w->rms_att_weight + layer * dim, dim);

        // key and value point to the kv cache
        int loff = layer * p->seq_len * kv_dim; // kv cache layer offset
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->att_wq + layer * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->att_wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->att_wv + layer * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < p->n_heads; i++) 
        {
            for (int j = 0; j < head_size; j += 2) 
            {
                float freq = 1.0f / powf(500000.0f, (float)j / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                float q0 = s->q[i * head_size + j];
                float q1 = s->q[i * head_size + j + 1];
                s->q[i * head_size + j] = q0 * fcr - q1 * fci;
                s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
                if (i < p->n_kv_heads) {
                float k0 = s->k[i * head_size + j];
                float k1 = s->k[i * head_size + j + 1];
                s->k[i * head_size + j] = k0 * fcr - k1 * fci;
                s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
                }
            }
        }
        int head;
         #pragma omp parallel for private(head)
        for (head = 0; head < p->n_heads; head++)
        {
            float *q = s->q + head * head_size;
            float *att = s->att + head * p->seq_len;
            for (int t = 0; t <= pos; t++) 
            {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * kv_dim + (head / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) 
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + head * head_size;
			memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
				float* v = s->value_cache + loff + t * kv_dim + (head / kv_mul) * head_size;
                float the_att = att[t];
				for (int i = 0; i < head_size; i++)
				{
					xb[i] += the_att * v[i];
				}
            }
        }
		matmul(s->xb2, s->xb, w->att_wo + layer * dim * dim, dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
		} // Now x is the output of the attention layer a.k.a x_ffn_input

        //ffn rms norm
		rmsnorm(s->xb, x, w->rms_ffn_weight + layer * dim, dim);
        // self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->ffn_w1 + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->ffn_w3 + layer * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->ffn_w2 + layer * dim * hidden_dim, hidden_dim, dim);
		for (int i = 0; i < dim; i++) {
			x[i] += s->xb[i];
		}
    }
	rmsnorm(x, x, w->rms_final_out, dim); // Output of the whole Transformer (all blocks)
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size); // Have a size of (128_256 x 1), this shit will using probability shit to calculate final result.
    return s->logits;
}


