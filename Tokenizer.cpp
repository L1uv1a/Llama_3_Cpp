#include "Tokenizer.h"

void Tokenizer::create_tokenizer(Tokenizer* t, int vocab_size)
{
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    // FILE* file = fopen("D:\\Project\\LLM\\llama2c\\llama2.c\\tokenizer.bin", "rb");
    FILE* file = fopen("C:\\Users\\DELL\\.llama\\checkpoints\\Llama3.2-1B-Instruct\\tokenizer.bin", "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", "D:\\Project\\LLM\\llama2c\\llama2.c\\tokenizer.bin"); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

int compare_tokens(const void *a, const void *b) 
{ 
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str); 
}

void Tokenizer::free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

int Tokenizer::str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) 
{
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok;
    tok.str = str;// acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    if (res == NULL) {
        return -1;
    }
    else
    {
        return res->id;
    }
}

void Tokenizer::encode(Tokenizer *t, const char* text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
    if (text == NULL) {
        printf ("LoL, text is NULL\n");
        exit(EXIT_FAILURE);
    }
    
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos)    tokens[(*n_tokens)++] = 128000;

    for (const char* c = text; *c != '\0'; c++)
    {
        if ( (*c & 0xC0) != 0x80 ) //UTF-8 start byte
        {
        str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ( (*c & 0xC0) == 0x80 && str_len <4) //UTF-8 continuation byte max 3 bytes
        {
            continue;
        }
        int token_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (token_id != -1)
        {
            tokens[(*n_tokens)++] = token_id;
        }
        else 
        {
            // if the string is not in the vocabulary, we add each byte separately
            for (int i = 0; i < str_len; i++) 
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }


    //BPE Time
    TokenIndex* temp_token = (TokenIndex*)malloc(((*n_tokens)) * sizeof(TokenIndex));
    int temp_token_len = *n_tokens;
    while (temp_token_len > 1)
    {
        int best_score = INT32_MAX;
        int merge_idx = -1;
        for (size_t i = 1; i < *n_tokens - 1; i++)
        {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] < best_score)
            {
                temp_token[i].id = id;
                temp_token[i].str = str_buffer;
                best_score = t->vocab_scores[id];
                merge_idx = i;
            } 
        }
        if (merge_idx != -1)
        {
            tokens[merge_idx] = temp_token[merge_idx].id;
            for (size_t i = merge_idx + 1; i < *n_tokens - 1; i++)
            {
                tokens[i] = tokens[i + 1];
            }
            (*n_tokens)--;
            temp_token_len--;
        }
        else
        {
            *n_tokens = temp_token_len;
            break;
        }
    }
    if (eos)    tokens[(*n_tokens)++] = 128001;

    free(str_buffer);

}

char* Tokenizer::decode(Tokenizer* t, int token)
{
    char* piece = t->vocab[token];

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}
