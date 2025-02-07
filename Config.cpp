#include "Config.h"

Config::Config()
    : dim(0),
      hidden_dim(0),
      n_layers(0),
      n_heads(0),
      n_kv_heads(0),
      vocab_size(0),
      seq_len(0)
{
}

Config::Config(int max_seq_len)
{
    *this = read_config_param();
    this->seq_len = max_seq_len;
}

Config Config::read_config_param()
{
    Config config_from_json;
    // open the file 
    FILE* fp = fopen("C:\\Users\\DELL\\.llama\\checkpoints\\Llama3.2-1B-Instruct\\params.json", "r");
    if (fp == NULL) {
        printf("Error: Unable to open the file.\n");
    }
    // read the file contents into a string 
    char buffer[1024];
    int len = fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);

    // parse the JSON data 
    cJSON* json = cJSON_Parse(buffer);
    if (json == NULL) {
        const char* error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            printf("Error: %s\n", error_ptr);
        }
        cJSON_Delete(json);
    }

    // access the JSON data 
    // "dim": 2048,
    cJSON* dim = cJSON_GetObjectItemCaseSensitive(json, "dim");
    if (cJSON_IsNumber(dim) && (dim->valueint != NULL)) {
        config_from_json.dim = dim->valueint;
        /*printf("dim: %d\n", dim->valueint);*/
    }
    // "n_layers": 16,
    cJSON* n_layers = cJSON_GetObjectItemCaseSensitive(json, "n_layers");
    if (cJSON_IsNumber(n_layers) && (n_layers->valueint != NULL)) {
        config_from_json.n_layers = n_layers->valueint;
        /*printf("n_layers: %d\n", n_layers->valueint);*/
    }
    // "n_heads": 32,
    cJSON* n_heads = cJSON_GetObjectItemCaseSensitive(json, "n_heads");
    if (cJSON_IsNumber(n_heads) && (n_heads->valueint != NULL)) {
        config_from_json.n_heads = n_heads->valueint;
        /*printf("n_heads: %d\n", n_heads->valueint);*/
    }
    // "n_kv_heads" : 8,
    cJSON* n_kv_heads = cJSON_GetObjectItemCaseSensitive(json, "n_kv_heads");
    if (cJSON_IsNumber(n_kv_heads) && (n_kv_heads->valueint != NULL)) {
        config_from_json.n_kv_heads = n_kv_heads->valueint;
        /*printf("n_kv_heads: %d\n", n_kv_heads->valueint);*/
    }
    // "vocab_size" : 128256,
    cJSON* vocab_size = cJSON_GetObjectItemCaseSensitive(json, "vocab_size");
    if (cJSON_IsNumber(vocab_size) && (vocab_size->valueint != NULL)) {
        config_from_json.vocab_size = vocab_size->valueint;
        /*printf("vocab_size: %d\n", vocab_size->valueint);*/
    }
    cJSON* ffn_dim_multiplier = cJSON_GetObjectItemCaseSensitive(json, "ffn_dim_multiplier");
    if (cJSON_IsNumber(ffn_dim_multiplier) && (ffn_dim_multiplier->valuedouble != NULL)) {
        //printf("ffn_dim_multiplier: %f\n", ffn_dim_multiplier->valuedouble);
    }
    cJSON* multiple_of = cJSON_GetObjectItemCaseSensitive(json, "multiple_of");
    if (cJSON_IsNumber(multiple_of) && (multiple_of->valueint != NULL)) {
        //printf("multiple_of: %d\n", multiple_of->valueint);
    }
    config_from_json.hidden_dim = 4 * config_from_json.dim;
    config_from_json.hidden_dim = int(2.0 * config_from_json.hidden_dim / 3.0);
    config_from_json.hidden_dim = int(ffn_dim_multiplier->valuedouble * config_from_json.hidden_dim);
    config_from_json.hidden_dim = multiple_of->valueint * 
                                  int((config_from_json.hidden_dim + multiple_of->valueint - 1.0) / multiple_of->valueint);

    // delete the JSON object 
    cJSON_Delete(json);
    return config_from_json;
}
