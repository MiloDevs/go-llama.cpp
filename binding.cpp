#include "common.h"
#include "llama.h"
#include "sampling.h"

#include "binding.h"
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <signal.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
            _exit(130);
    }
}
#endif

struct llama_binding_state {
    llama_context * ctx;
    llama_model   * model;
};

static void parse_logit_bias(const char *logit_bias_str, common_params_sampling &sparams) {
    if (!logit_bias_str || logit_bias_str[0] == '\0') return;
    std::stringstream ss(logit_bias_str);
    llama_token key;
    char sign;
    std::string value_str;
    if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
        float val = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
        sparams.logit_bias.push_back({key, val});
    }
}


int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    llama_model*  model = state->model;
    common_params params = *params_p;

    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        params.sampling.seed = time(NULL);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;

    auto embd_inp = common_tokenize(ctx, params.prompt, add_bos, true);

    if (embd_inp.empty()) {
        fprintf(stderr, "%s: empty input\n", __func__);
        return 1;
    }

    auto batch = llama_batch_get_one(embd_inp.data(), (int32_t)embd_inp.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);
    const float * embeddings = llama_get_embeddings(ctx);
    if (!embeddings) {
        fprintf(stderr, "%s: failed to get embeddings\n", __func__);
        return 1;
    }

    for (int i = 0; i < n_embd; i++) {
        res_embeddings[i] = embeddings[i];
    }

    return 0;
}


int get_token_embeddings(void* params_ptr, void* state_pr, int *tokens, int tokenSize, float * res_embeddings) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;

    for (int i = 0; i < tokenSize; i++) {
        std::string token_str = common_token_to_piece(ctx, (llama_token)tokens[i]);
        params_p->prompt += token_str;
    }

    return get_embeddings(params_ptr, state_pr, res_embeddings);
}


int eval(void* params_ptr, void* state_pr, char *text) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;

    std::string str = std::string(text);
    auto tokens = common_tokenize(ctx, str, true, true);

    if (tokens.empty()) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    const int n_batch = params_p->n_batch > 0 ? params_p->n_batch : 512;
    for (int i = 0; i < (int)tokens.size(); i += n_batch) {
        int n_eval = std::min((int)tokens.size() - i, n_batch);
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
    }

    return 0;
}


static llama_context ** g_ctx;
static common_params  * g_params;

int llama_predict(void* params_ptr, void* state_pr, char* result, bool debug) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    llama_context* ctx = state->ctx;
    llama_model*  model = state->model;

    common_params params = *params_p;
    g_params = &params;
    g_ctx = &ctx;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        params.sampling.seed = time(NULL);
    }

    if (params.rope_freq_base != 10000.0f) {
        fprintf(stderr, "%s: warning: changing RoPE frequency base to %g (default 10000.0)\n",
                __func__, params.rope_freq_base);
    }
    if (params.rope_freq_scale != 1.0f) {
        fprintf(stderr, "%s: warning: scaling RoPE frequency by %g (default 1.0)\n",
                __func__, params.rope_freq_scale);
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        if (debug) {
            fprintf(stderr, "%s: attempting to load saved session from '%s'\n",
                    __func__, path_session.c_str());
        }
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(),
                                       session_tokens.data(),
                                       session_tokens.capacity(),
                                       &n_token_count_out)) {
                fprintf(stderr, "%s: error: failed to load session file '%s'\n",
                        __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            if (debug) {
                fprintf(stderr, "%s: loaded session with %d tokens\n",
                        __func__, (int)session_tokens.size());
            }
        } else {
            if (debug) {
                fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
            }
        }
    }

    const bool add_bos = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;

    std::vector<llama_token> embd_inp;
    if (!params.prompt.empty() || session_tokens.empty()) {
        embd_inp = common_tokenize(ctx, params.prompt, add_bos, true);
    } else {
        embd_inp = session_tokens;
    }

    if (embd_inp.empty()) {
        embd_inp.push_back(llama_vocab_bos(vocab));
    }

    if ((int)embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n",
                __func__, (int)embd_inp.size(), n_ctx - 4);
        return 1;
    }

    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() ||
                id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (debug) {
            if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
                fprintf(stderr, "%s: using full prompt from session file\n", __func__);
            } else if (n_matching_session_tokens >= embd_inp.size()) {
                fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
            } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
                fprintf(stderr, "%s: warning: low session similarity (%zu / %zu tokens)\n",
                        __func__, n_matching_session_tokens, embd_inp.size());
            }
        }
    }

    if (!embd_inp.empty() &&
        n_matching_session_tokens == embd_inp.size() &&
        session_tokens.size() > embd_inp.size()) {
        session_tokens.resize(embd_inp.size() - 1);
    }

    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        fprintf(stderr, "%s: error: failed to initialise sampler\n", __func__);
        return 1;
    }

    bool need_to_save_session = !path_session.empty() &&
                                n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<llama_token> embd;
    std::string res = "";
    bool is_antiprompt = false;

    // warm-up: eval BOS so logits are valid before the first sample
    {
        llama_token bos = llama_vocab_bos(vocab);
        auto tmp_batch = llama_batch_get_one(&bos, 1);
        llama_decode(ctx, tmp_batch);
        llama_perf_context_reset(ctx);
    }

    while (n_remain != 0) {
        if (!embd.empty()) {
            llama_memory_t mem = llama_get_memory(ctx);
            if (n_past + (int)embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;
                n_past = std::max(1, params.n_keep);

                // drop the first half of non-kept tokens to make room
                llama_memory_seq_rm (mem, 0, params.n_keep + 1,
                                          params.n_keep + 1 + n_left / 2);
                llama_memory_seq_add(mem, 0,
                                     params.n_keep + 1 + n_left / 2,
                                     n_ctx,
                                     -(n_left / 2));
                n_past = params.n_keep + n_left / 2 + 1;
                path_session.clear();
            }

            if (n_session_consumed < (int)session_tokens.size()) {
                size_t i = 0;
                for (; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }
                    n_past++;
                    n_session_consumed++;
                    if (n_session_consumed >= (int)session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
                int n_eval = std::min((int)embd.size() - i, params.n_batch);
                auto batch = llama_batch_get_one(&embd[i], n_eval);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    common_sampler_free(smpl);
                    return 1;
                }
                n_past += n_eval;
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = (int)session_tokens.size();
            }
        }

        embd.clear();

        if ((int)embd_inp.size() <= n_consumed) {
            if (!path_session.empty() && need_to_save_session &&
                !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(),
                                      session_tokens.data(),
                                      session_tokens.size());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);

            embd.push_back(id);
            --n_remain;

            std::string token_str = common_token_to_piece(ctx, id);
            if (!tokenCallback(state_pr, (char*)token_str.c_str())) {
                break;
            }

            if (debug) {
                printf("%s", token_str.c_str());
            }
            res += token_str;

            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

        } else {
            while ((int)embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int)embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        if ((int)embd_inp.size() <= n_consumed && !params.antiprompt.empty()) {
            for (const std::string & antiprompt : params.antiprompt) {
                if (res.size() >= antiprompt.size()) {
                    size_t search_start = res.size() - antiprompt.size() - 2;
                    if (search_start > res.size()) search_start = 0;
                    if (res.find(antiprompt, search_start) != std::string::npos) {
                        is_antiprompt = true;
                        break;
                    }
                }
            }
        }

        if (is_antiprompt) {
            break;
        }
    }

    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        if (debug) {
            fprintf(stderr, "\n%s: saving final output to session file '%s'\n",
                    __func__, path_session.c_str());
        }
        llama_state_save_file(ctx, path_session.c_str(),
                              session_tokens.data(),
                              session_tokens.size());
    }

    if (debug) {
        llama_perf_context_print(ctx);
        llama_perf_context_reset(ctx);
        common_perf_print(ctx, smpl);
    }

    common_sampler_free(smpl);

    strcpy(result, res.c_str());
    return 0;
}


int speculative_sampling(void* params_ptr, void* target_model, void* draft_model,
                         char* result, bool debug) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* target_state = (llama_binding_state*) target_model;
    llama_binding_state* draft_state  = (llama_binding_state*) draft_model;

    common_params params = *params_p;

    llama_context * ctx_tgt = target_state->ctx;
    llama_context * ctx_dft = draft_state->ctx;
    llama_model   * model_tgt = target_state->model;

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    if ((int)inp.size() > max_tokens_list_size) {
        fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n",
                __func__, (int)inp.size(), max_tokens_list_size);
        return 1;
    }

    const int n_input = (int)inp.size();
    const int n_draft = params.speculative.n_max;

    const auto t_enc_start = ggml_time_us();

    // evaluate all but the last token in target, then the last separately to get logits
    if (n_input > 1) {
        auto batch_tgt = llama_batch_get_one(inp.data(), n_input - 1);
        llama_decode(ctx_tgt, batch_tgt);
    }
    {
        auto batch_tgt = llama_batch_get_one(&inp.back(), 1);
        llama_decode(ctx_tgt, batch_tgt);
    }
    {
        auto batch_dft = llama_batch_get_one(inp.data(), n_input);
        llama_decode(ctx_dft, batch_dft);
    }

    const auto t_enc_end = ggml_time_us();

    common_sampler * smpl_tgt = common_sampler_init(model_tgt, params.sampling);
    common_sampler * smpl_dft = common_sampler_init(draft_state->model, params.sampling);
    if (!smpl_tgt || !smpl_dft) {
        if (smpl_tgt) common_sampler_free(smpl_tgt);
        if (smpl_dft) common_sampler_free(smpl_dft);
        return 1;
    }

    for (auto & id : inp) {
        common_sampler_accept(smpl_tgt, id, false);
        common_sampler_accept(smpl_dft, id, false);
    }

    std::string res = "";
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_past_tgt = n_input;
    int n_past_dft = n_input;
    bool has_eos = false;

    std::vector<llama_token> drafted;

    const auto t_dec_start = ggml_time_us();

    while (true) {
        int i_dft = 0;
        while (true) {
            const llama_token id = common_sampler_sample(smpl_tgt, ctx_tgt, i_dft);
            common_sampler_accept(smpl_tgt, id, true);

            std::string token_str = common_token_to_piece(ctx_tgt, id);
            if (!tokenCallback(draft_model, (char*)token_str.c_str())) {
                break;
            }
            res += token_str;

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            if (i_dft < (int)drafted.size() && id == drafted[i_dft]) {
                ++n_accept;
                ++n_past_tgt;
                ++n_past_dft;
                ++i_dft;
                continue;
            }

            // drafted token rejected — sync draft model to the accepted token
            {
                llama_token id_copy = id;
                auto batch = llama_batch_get_one(&id_copy, 1);
                llama_decode(ctx_dft, batch);
                ++n_past_dft;
            }

            drafted.clear();
            drafted.push_back(id);
            break;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        // sync draft sampler state to target before drafting
        common_sampler_free(smpl_dft);
        smpl_dft = common_sampler_clone(smpl_tgt);

        int n_past_cur = n_past_dft;
        for (int i = 0; i < n_draft; ++i) {
            auto sparams_draft = llama_sampler_chain_default_params();
            sparams_draft.no_perf = true;
            llama_sampler * greedy = llama_sampler_chain_init(sparams_draft);
            llama_sampler_chain_add(greedy, llama_sampler_init_greedy());

            const llama_token id = llama_sampler_sample(greedy, ctx_dft, -1);
            llama_sampler_accept(greedy, id);
            llama_sampler_free(greedy);

            drafted.push_back(id);
            ++n_drafted;

            if (i == n_draft - 1) break;

            auto batch = llama_batch_get_one(&drafted.back(), 1);
            llama_decode(ctx_dft, batch);
            ++n_past_cur;
        }

        if (!drafted.empty()) {
            auto batch = llama_batch_get_one(drafted.data(), (int32_t)drafted.size());
            llama_decode(ctx_tgt, batch);
            ++n_past_tgt;
        }

        // remove the seed token — it was already evaluated before the draft loop
        if (!drafted.empty()) {
            drafted.erase(drafted.begin());
        }
    }

    if (debug) {
        auto t_dec_end = ggml_time_us();
        fprintf(stderr, "\n\n");
        fprintf(stderr, "encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
                n_input,
                (t_enc_end - t_enc_start) / 1e6f,
                n_input / ((t_enc_end - t_enc_start) / 1e6f));
        fprintf(stderr, "decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
                n_predict,
                (t_dec_end - t_dec_start) / 1e6f,
                n_predict / ((t_dec_end - t_dec_start) / 1e6f));
        fprintf(stderr, "\nn_draft   = %d\n", n_draft);
        fprintf(stderr, "n_predict = %d\n", n_predict);
        fprintf(stderr, "n_drafted = %d\n", n_drafted);
        fprintf(stderr, "n_accept  = %d\n", n_accept);
        if (n_drafted > 0) {
            fprintf(stderr, "accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
        }
        fprintf(stderr, "\ndraft:\n");
        llama_perf_context_print(ctx_dft);
        fprintf(stderr, "\ntarget:\n");
        llama_perf_context_print(ctx_tgt);
        fprintf(stderr, "\n\n");
    }

    common_sampler_free(smpl_tgt);
    common_sampler_free(smpl_dft);

    strcpy(result, res.c_str());
    return 0;
}


void llama_binding_free_model(void *state_ptr) {
    llama_binding_state* state = (llama_binding_state*) state_ptr;
    llama_free(state->ctx);
    llama_model_free(state->model);
    delete state;
}


void llama_free_params(void* params_ptr) {
    common_params* params = (common_params*) params_ptr;
    delete params;
}


int llama_tokenize_string(void* params_ptr, void* state_pr, int* result) {
    common_params* params_p = (common_params*) params_ptr;
    llama_binding_state* state = (llama_binding_state*) state_pr;
    const llama_vocab * vocab = llama_model_get_vocab(state->model);

    const bool add_bos = llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM;
    int n_max = params_p->n_ctx > 0 ? params_p->n_ctx : 4096;

    return llama_tokenize(vocab,
                          params_p->prompt.c_str(),
                          (int32_t)params_p->prompt.size(),
                          (llama_token *)result,
                          n_max,
                          add_bos,
                          true);
}


std::vector<std::string> create_vector(const char** strings, int count) {
    std::vector<std::string>* vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++) {
        vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

void delete_vector(std::vector<std::string>* vec) {
    delete vec;
}


int load_state(void *ctx_ptr, char *statefile, char *modes) {
    llama_context* ctx = (llama_context*) ctx_ptr;

    const size_t state_size = llama_state_get_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];

    FILE *fp_read = fopen(statefile, modes);
    if (!fp_read) {
        fprintf(stderr, "\n%s : failed to open state file\n", __func__);
        delete[] state_mem;
        return 1;
    }

    const size_t ret = fread(state_mem, 1, state_size, fp_read);
    fclose(fp_read);

    if (ret != state_size) {
        fprintf(stderr, "\n%s : failed to read state\n", __func__);
        delete[] state_mem;
        return 1;
    }

    llama_state_set_data(ctx, state_mem, state_size);
    delete[] state_mem;
    return 0;
}

void save_state(void *ctx_ptr, char *dst, char *modes) {
    llama_context* ctx = (llama_context*) ctx_ptr;

    const size_t state_size = llama_state_get_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];

    FILE *fp_write = fopen(dst, modes);
    if (fp_write) {
        llama_state_get_data(ctx, state_mem, state_size);
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }

    delete[] state_mem;
}


void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float frequency_penalty,
                            float presence_penalty, int mirostat, float mirostat_eta,
                            float mirostat_tau, bool penalize_nl, const char *logit_bias,
                            const char *session_file, bool prompt_cache_all, bool mlock,
                            bool mmap, const char *maingpu, const char *tensorsplit,
                            bool prompt_cache_ro, const char *grammar,
                            float rope_freq_base, float rope_freq_scale,
                            float negative_prompt_scale, const char* negative_prompt,
                            int n_draft) {
    common_params* params = new common_params;

    params->sampling.seed             = (uint32_t)seed;
    params->sampling.top_k            = top_k;
    params->sampling.top_p            = top_p;
    params->sampling.temp             = temp;
    params->sampling.penalty_repeat   = repeat_penalty;
    params->sampling.penalty_last_n   = repeat_last_n;
    params->sampling.typ_p            = typical_p;
    params->sampling.penalty_freq     = frequency_penalty;
    params->sampling.penalty_present  = presence_penalty;
    params->sampling.mirostat         = mirostat;
    params->sampling.mirostat_eta     = mirostat_eta;
    params->sampling.mirostat_tau     = mirostat_tau;
    params->sampling.ignore_eos       = ignore_eos;

    if (grammar && grammar[0] != '\0') {
        params->sampling.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER,
                                                   std::string(grammar));
    }

    parse_logit_bias(logit_bias, params->sampling);

    params->cpuparams.n_threads = threads;
    params->n_predict           = tokens;
    params->n_batch             = n_batch;
    params->n_keep              = n_keep;
    params->rope_freq_base      = rope_freq_base;
    params->rope_freq_scale     = rope_freq_scale;
    params->use_mlock           = mlock;
    params->use_mmap            = mmap;
    params->prompt_cache_all    = prompt_cache_all;
    params->prompt_cache_ro     = prompt_cache_ro;
    params->path_prompt_cache   = session_file ? session_file : "";
    params->prompt              = prompt ? prompt : "";
    params->speculative.n_max   = n_draft;

    if (antiprompt_count > 0 && antiprompt) {
        params->antiprompt = create_vector(antiprompt, antiprompt_count);
    }

    // removed from llama.cpp: negative prompt/CFG, tfs_z, penalize_nl, memory_f16
    (void)negative_prompt_scale;
    (void)negative_prompt;
    (void)memory_f16;
    (void)tfs_z;
    (void)penalize_nl;

    // maingpu/tensorsplit are model-load params, not inference params
    (void)maingpu;
    (void)tensorsplit;

    return params;
}


void* load_model(const char *fname,
                 int n_ctx, int /*n_seed*/,
                 bool memory_f16, bool mlock, bool embeddings,
                 bool mmap, bool low_vram, int n_gpu_layers, int n_batch,
                 const char *maingpu, const char *tensorsplit,
                 bool numa,
                 float rope_freq_base, float rope_freq_scale,
                 bool mul_mat_q,
                 const char *lora, const char *lora_base, bool perplexity) {

    llama_backend_init();

    if (numa) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_mmap     = mmap;
    mparams.use_mlock    = mlock;

    if (maingpu && maingpu[0] != '\0') {
        mparams.main_gpu = std::stoi(maingpu);
    }

    static float tensor_split_arr[128] = {0};
    if (tensorsplit && tensorsplit[0] != '\0') {
        std::string arg_next = tensorsplit;
        const std::regex regex{R"([,/]+)"};
        std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
        std::vector<std::string> split_arg{it, {}};
        int max_devices = (int)llama_max_devices();
        for (int i = 0; i < max_devices; ++i) {
            if (i < (int)split_arg.size()) {
                tensor_split_arr[i] = std::stof(split_arg[i]);
            } else {
                tensor_split_arr[i] = 0.0f;
            }
        }
        mparams.tensor_split = tensor_split_arr;
    }

    llama_model * model = llama_model_load_from_file(fname, mparams);
    if (!model) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return nullptr;
    }

    // lora_base is obsolete — ignored
    (void)lora_base;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = (uint32_t)n_ctx;
    cparams.n_batch    = (uint32_t)n_batch;
    cparams.n_ubatch   = (uint32_t)n_batch;
    cparams.embeddings = embeddings;
    if (rope_freq_base  != 0.0f) cparams.rope_freq_base  = rope_freq_base;
    if (rope_freq_scale != 0.0f) cparams.rope_freq_scale = rope_freq_scale;

    // obsolete params — ignored
    (void)memory_f16;
    (void)low_vram;
    (void)mul_mat_q;
    (void)perplexity;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "%s: error: failed to create context\n", __func__);
        llama_model_free(model);
        return nullptr;
    }

    if (lora && lora[0] != '\0') {
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, lora);
        if (!adapter) {
            fprintf(stderr, "%s: warning: failed to load LoRA adapter '%s'\n",
                    __func__, lora);
        } else {
            float scale = 1.0f;
            llama_adapter_lora * adapters[1] = {adapter};
            float scales[1] = {scale};
            if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) {
                fprintf(stderr, "%s: warning: failed to apply LoRA adapter\n", __func__);
            }
            llama_adapter_lora_free(adapter);
        }
    }

    llama_binding_state * state = new llama_binding_state;
    state->ctx   = ctx;
    state->model = model;
    return state;
}
