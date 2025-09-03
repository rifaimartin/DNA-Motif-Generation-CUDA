#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <fstream>
#include <set>
#include <cfloat>
#include <algorithm>
#include <cstdio>

#ifndef INFINITY
#define INFINITY HUGE_VALF
#endif

// Macro untuk penanganan kesalahan CUDA
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// ======================= Models / Params =======================
struct Constraints {
    int payload_size;    // 60
    int payload_num;     // 15
    int key_size;        // 20
    int key_num;         // 8
    float min_gc;        // 25.0
    float max_gc;        // 65.0
    int max_hom;         // 5
    int max_hairpin;     // 1
    int loop_size_min;   // 6
    int loop_size_max;   // 7
};

struct Hyperparameters {
    float hom_shape;               // 70
    float gc_shape;                // 10
    float hairpin_shape;           // 8
    float similarity_shape;        // 60
    float no_key_in_payload_shape; // 45
    float hom_weight;              // 1
    float gc_weight;               // 1
    float hairpin_weight;          // 1
    float similarity_weight;       // 1
    float no_key_weight;           // 1
};

__constant__ int nucleotide_complement[4] = {1, 0, 3, 2}; // A<->T, C<->G

// ======================= Device Helpers =======================

__device__ inline bool is_gc(unsigned char b) { return (b == 2 /*C*/ || b == 3 /*G*/); }

__device__ bool would_match_any_key_suffix(
    const unsigned char* seq, int len, unsigned char candidate,
    const unsigned char* keys, int key_num, int key_size)
{
    int new_len = len + 1;
    if (new_len < key_size) return false;

    int start = new_len - key_size;
    for (int k = 0; k < key_num; ++k) {
        bool match = true;
        for (int j = 0; j < key_size - 1; ++j) {
            if (seq[start + j] != keys[k * key_size + j]) { match = false; break; }
        }
        if (match && candidate == keys[k * key_size + (key_size - 1)]) {
            return true;
        }
    }
    return false;
}

__device__ bool would_form_hairpin_after_append(
    const unsigned char* seq, int len, unsigned char candidate, const Constraints* c)
{
    int last = len;
    if (last < c->loop_size_min) return false;

    for (int loop = c->loop_size_min; loop <= c->loop_size_max; ++loop) {
        int stem_len = 0;
        while (true) {
            int left = last - loop - 1 - stem_len;
            if (left < 0) break;

            unsigned char rb = (stem_len == 0) ? candidate : seq[len - stem_len];
            unsigned char lb = seq[left];

            if (lb == nucleotide_complement[rb]) {
                ++stem_len;
                if (stem_len > c->max_hairpin) return true;
            } else {
                break;
            }
        }
    }
    return false;
}

// ======================= Scoring =======================
__device__ void calculate_log_scores(
    const unsigned char* sequence,
    int length,
    float* log_scores,
    const Constraints* constraints,
    const Hyperparameters* hyperparams,
    bool* failed,
    const unsigned char* keys,
    int key_num,
    int key_size,
    bool check_no_key_in_payload)
{
    *failed = false;

    for (int nuc = 0; nuc < 4; ++nuc) {
        float ls = 0.0f;

        // Homopolymer
        int hom_len = 1;
        if (length > 0 && sequence[length - 1] == nuc) {
            hom_len = 2;
            for (int i = length - 2; i >= 0; --i) {
                if (sequence[i] == nuc) ++hom_len;
                else break;
            }
        }
        if (hom_len > constraints->max_hom) {
            log_scores[nuc] = -INFINITY; continue;
        }
        if (hom_len > constraints->max_hom - 1) {
            float penalty = -powf(hyperparams->hom_shape, (float)hom_len / constraints->max_hom);
            ls += hyperparams->hom_weight * penalty;
        }

        // GC Content
        int gc_count = 0;
        for (int i = 0; i < length; ++i) if (is_gc(sequence[i])) ++gc_count;
        if (is_gc(nuc)) ++gc_count;
        float current_gc = (float)gc_count / (float)(length + 1) * 100.0f;

        if (length + 1 >= max(8, constraints->key_size / 2)) {
            if (current_gc < constraints->min_gc || current_gc > constraints->max_gc) {
                log_scores[nuc] = -INFINITY; continue;
            }
        } else {
            if (current_gc < constraints->min_gc) {
                float penalty = hyperparams->gc_weight * (constraints->min_gc - current_gc);
                ls -= penalty * 0.1f;
            } else if (current_gc > constraints->max_gc) {
                float penalty = hyperparams->gc_weight * (current_gc - constraints->max_gc);
                ls -= penalty * 0.1f;
            }
        }

        // Hairpin
        if (would_form_hairpin_after_append(sequence, length, (unsigned char)nuc, constraints)) {
            log_scores[nuc] = -INFINITY; continue;
        }

        // noKeyInPayload
        if (check_no_key_in_payload && keys != nullptr) {
            if (would_match_any_key_suffix(sequence, length, (unsigned char)nuc,
                                           keys, key_num, key_size)) {
                log_scores[nuc] = -INFINITY; continue;
            }
        }

        // Similarity penalty
        if (length > 0 && sequence[length - 1] == nuc) {
            ls -= 0.05f * hyperparams->similarity_weight * hyperparams->similarity_shape;
        }

        log_scores[nuc] = ls;
    }

    bool all_failed = true;
    for (int i = 0; i < 4; ++i) if (log_scores[i] > -INFINITY) { all_failed = false; break; }
    *failed = all_failed;
}

__device__ bool softmax(float* log_scores, float* probabilities) {
    bool has_valid = false;
    for (int i = 0; i < 4; ++i) if (log_scores[i] > -INFINITY) { has_valid = true; break; }
    if (!has_valid) return false;

    float m = -INFINITY;
    for (int i = 0; i < 4; ++i) if (log_scores[i] > m) m = log_scores[i];

    float sum = 0.f;
    for (int i = 0; i < 4; ++i) {
        if (log_scores[i] > -INFINITY) {
            probabilities[i] = expf(log_scores[i] - m);
            sum += probabilities[i];
        } else probabilities[i] = 0.f;
    }
    if (sum <= 0.f) return false;
    for (int i = 0; i < 4; ++i) probabilities[i] /= sum;
    return true;
}

__device__ int sample_nucleotide(float* p, curandState* state) {
    float r = curand_uniform(state);
    float acc = 0.f;
    for (int i = 0; i < 4; ++i) {
        acc += p[i];
        if (r <= acc) return i;
    }
    return 3;
}

// ======================= Kernels =======================

__global__ void key_generation_kernel(
    unsigned char* keys,
    curandState* states,
    const Constraints* constraints,
    const Hyperparameters* hyperparams,
    int num_keys,
    unsigned char* success_flags)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    curandState local_state = states[tid];
    unsigned char* my_key = &keys[tid * constraints->key_size];
    bool key_failed = false;

    for (int pos = 0; pos < constraints->key_size && !key_failed; ++pos) {
        float log_scores[4], probs[4]; bool failed = false;
        calculate_log_scores(my_key, pos, log_scores, constraints, hyperparams,
                             &failed, nullptr, 0, 0, false);
        if (failed) { key_failed = true; break; }
        if (!softmax(log_scores, probs)) { key_failed = true; break; }
        int sel = sample_nucleotide(probs, &local_state);
        my_key[pos] = (unsigned char)sel;
    }

    success_flags[tid] = key_failed ? 0u : 1u;
    states[tid] = local_state;
}

__global__ void mdp_generation_kernel(
    unsigned char* sequences,
    curandState* states,
    const Constraints* constraints,
    const Hyperparameters* hyperparams,
    int num_sequences,
    unsigned char* success_flags,
    const unsigned char* keys,
    int key_num,
    int key_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sequences) return;

    curandState local_state = states[tid];
    unsigned char* my_seq = &sequences[tid * constraints->payload_size];
    bool seq_failed = false;

    for (int pos = 0; pos < constraints->payload_size && !seq_failed; ++pos) {
        float log_scores[4], probs[4]; bool failed = false;
        calculate_log_scores(my_seq, pos, log_scores, constraints, hyperparams,
                             &failed, keys, key_num, key_size, true);
        if (failed) { seq_failed = true; break; }
        if (!softmax(log_scores, probs)) { seq_failed = true; break; }
        int sel = sample_nucleotide(probs, &local_state);
        my_seq[pos] = (unsigned char)sel;
    }

    success_flags[tid] = seq_failed ? 0u : 1u;
    states[tid] = local_state;
}

__global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) curand_init(seed + tid, 0, 0, &states[tid]);
}

// ======================= Host Helpers =======================

std::string decode_sequence(const std::vector<unsigned char>& seq) {
    static const char nuc[] = {'A','T','C','G'};
    std::string s; s.reserve(seq.size());
    for (auto b : seq) s += nuc[b];
    return s;
}

bool validate_gc_hom(const std::string& s, const Constraints& c) {
    int gc = 0, max_consec = 1, cur = 1;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == 'G' || s[i] == 'C') ++gc;
        if (i && s[i] == s[i-1]) { ++cur; max_consec = std::max(max_consec, cur); }
        else cur = 1;
    }
    float gc_ratio = 100.f * (float)gc / (float)s.size();
    if (gc_ratio < c.min_gc || gc_ratio > c.max_gc) return false;
    if (max_consec > c.max_hom) return false;
    return true;
}

bool validate_no_key_in_payload(const std::string& payload,
                               const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (k.size() && payload.find(k) != std::string::npos) return false;
    }
    return true;
}

bool validate_no_hairpin(const std::string& s, const Constraints& c) {
    auto comp = [](char x)->char {
        if (x=='A') return 'T'; if (x=='T') return 'A';
        if (x=='C') return 'G'; return 'C';
    };
    const int n = (int)s.size();

    for (int i = 0; i < n; ++i) {
        for (int loop = c.loop_size_min; loop <= c.loop_size_max; ++loop) {
            int stem_len = 0;
            while (true) {
                int right = i - stem_len;
                int left = i - loop - 1 - stem_len;
                if (left < 0 || right >= n) break;

                if (s[left] == comp(s[right])) {
                    ++stem_len;
                    if (stem_len > c.max_hairpin) return false;
                } else {
                    break;
                }
            }
        }
    }
    return true;
}

std::vector<std::string> generate_motifs(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& payloads)
{
    std::set<std::string> uniq;
    for (const auto& p : payloads) {
        for (size_t i = 0; i < keys.size(); ++i) {
            std::string m1 = keys[i] + p + keys[i];
            std::string m2 = keys[i] + p + keys[(i + 1) % keys.size()];
            uniq.insert(m1); uniq.insert(m2);
        }
    }
    std::vector<std::string> v; v.reserve(uniq.size());
    for (auto& x : uniq) v.push_back(x);
    return v;
}

void save_results(const std::vector<std::vector<std::string>>& all_keys,
                 const std::vector<std::vector<std::string>>& all_payloads,
                 const std::vector<std::vector<std::string>>& all_motifs)
{
    try {
        std::ofstream fk("successful_keys.txt");
        for (size_t s = 0; s < all_keys.size() && s < 10; ++s) {
            fk << "Keys Set " << (s+1) << ": [";
            for (size_t i = 0; i < all_keys[s].size(); ++i) {
                fk << "'" << all_keys[s][i] << "'";
                if (i + 1 < all_keys[s].size()) fk << ", ";
            }
            fk << "]\n";
        }

        std::ofstream fp("successful_payloads.txt");
        for (size_t s = 0; s < all_payloads.size() && s < 10; ++s) {
            fp << "Payloads Set " << (s+1) << ": [";
            for (size_t i = 0; i < all_payloads[s].size(); ++i) {
                fp << "'" << all_payloads[s][i] << "'";
                if (i + 1 < all_payloads[s].size()) fp << ", ";
            }
            fp << "]\n";
        }

        std::ofstream fm("successful_motifs.txt");
        for (size_t s = 0; s < all_motifs.size() && s < 10; ++s) {
            fm << "Motifs Set " << (s+1) << ":\n";
            for (size_t i = 0; i < all_motifs[s].size() && i < 20; ++i) {
                fm << "  " << all_motifs[s][i] << "\n";
            }
            fm << "\n";
        }
    } catch (const std::exception& e) {
        printf("Error menulis file: %s\n", e.what());
    }
}

// ======================= MAIN =======================
int main() {
    printf("=== CUDA MDP DNA Generator  ===\n");

    // Input Parameter 
    Constraints c{};
    c.payload_size = 60; c.payload_num = 15;
    c.key_size = 20;     c.key_num    = 8;
    c.min_gc = 25.f;     c.max_gc     = 65.f;
    c.max_hom = 5;       c.max_hairpin = 1;
    c.loop_size_min = 6; c.loop_size_max = 7;

    Hyperparameters h{};
    h.hom_shape = 70.f; h.gc_shape = 10.f; h.hairpin_shape = 8.f;
    h.similarity_shape = 60.f; h.no_key_in_payload_shape = 45.f;
    h.hom_weight = 1.f; h.gc_weight = 1.f; h.hairpin_weight = 1.f;
    h.similarity_weight = 1.f; h.no_key_weight = 1.f;

    const int num_rounds = 1000;
    const int TPB = 256;
    const int blocks_keys = (c.key_num + TPB - 1) / TPB;
    const int blocks_payload = (c.payload_num + TPB - 1) / TPB;

    printf("Starting motif generation with %d rounds...\n", num_rounds);
    printf("Constraints: {'hom','gcContent','hairpin','noKeyInPayload'}\n");
    printf("------------------------------------------------------------\n");

    std::vector<std::vector<std::string>> all_keys_ok;
    std::vector<std::vector<std::string>> all_payloads_ok;
    std::vector<std::vector<std::string>> all_motifs_ok;
    int successful_sets = 0;

    clock_t t0 = clock();

    // Device buffers
    unsigned char *d_payloads = nullptr, *d_keys = nullptr;
    curandState *d_states = nullptr;
    Constraints *d_c = nullptr; Hyperparameters *d_h = nullptr;
    unsigned char *d_key_success = nullptr, *d_payload_success = nullptr;

    CUDA_CHECK(cudaMalloc(&d_payloads, c.payload_num * c.payload_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_keys, c.key_num * c.key_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_states, (c.payload_num + c.key_num) * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(Constraints)));
    CUDA_CHECK(cudaMalloc(&d_h, sizeof(Hyperparameters)));
    CUDA_CHECK(cudaMalloc(&d_key_success, c.key_num * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_payload_success, c.payload_num * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_c, &c, sizeof(Constraints), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h, &h, sizeof(Hyperparameters), cudaMemcpyHostToDevice));

    for (int round = 0; round < num_rounds; ++round) {
        // Inisialisasi seed acak per ronde
        int total_threads = c.payload_num + c.key_num;
        int total_blocks = (total_threads + TPB - 1) / TPB;
        init_curand_states<<<total_blocks, TPB>>>(d_states, (unsigned long)time(NULL) + round * 1000, total_threads);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 1) Generate KEYS
        key_generation_kernel<<<blocks_keys, TPB>>>(
            d_keys, d_states, d_c, d_h, c.key_num, d_key_success);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) Generate PAYLOADS
        mdp_generation_kernel<<<blocks_payload, TPB>>>(
            d_payloads, &d_states[c.key_num], d_c, d_h,
            c.payload_num, d_payload_success, d_keys, c.key_num, c.key_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back
        std::vector<unsigned char> h_keys(c.key_num * c.key_size);
        std::vector<unsigned char> h_payloads(c.payload_num * c.payload_size);
        std::vector<unsigned char> h_key_success(c.key_num);
        std::vector<unsigned char> h_payload_success(c.payload_num);

        CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, h_keys.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_payloads.data(), d_payloads, h_payloads.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_key_success.data(), d_key_success, h_key_success.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_payload_success.data(), d_payload_success, h_payload_success.size(), cudaMemcpyDeviceToHost));

        // Decode only successful KEYS
        std::vector<std::string> keys_str;
        for (int k = 0; k < c.key_num; ++k) if (h_key_success[k]) {
            std::vector<unsigned char> key_seq(
                h_keys.begin() + k * c.key_size,
                h_keys.begin() + (k + 1) * c.key_size);
            keys_str.push_back(decode_sequence(key_seq));
        }
        if (keys_str.size() < c.key_num) continue; // Memerlukan semua key berhasil

        // Decode & validate PAYLOADS
        std::vector<std::string> payloads_str;
        for (int i = 0; i < c.payload_num; ++i) {
            if (!h_payload_success[i]) continue;

            std::vector<unsigned char> pl_seq(
                h_payloads.begin() + i * c.payload_size,
                h_payloads.begin() + (i + 1) * c.payload_size);
            std::string pl = decode_sequence(pl_seq);

            if (!validate_gc_hom(pl, c)) continue;
            if (!validate_no_hairpin(pl, c)) continue;
            if (!validate_no_key_in_payload(pl, keys_str)) continue;

            payloads_str.push_back(pl);
        }
        if (payloads_str.size() < c.payload_num) continue; // Memerlukan semua payload berhasil

        // Generate motifs
        auto motifs = generate_motifs(keys_str, payloads_str);
        if (!motifs.empty()) {
            all_keys_ok.push_back(keys_str);
            all_payloads_ok.push_back(payloads_str);
            all_motifs_ok.push_back(motifs);
            ++successful_sets;
        }

        // every 100 itteration
        if (round % 100 == 0 && round > 0) {
            double elapsed_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
            double succ_rate = 100.0 * (double)successful_sets / round;
            double avg_ms = elapsed_ms / round;
            double eta_ms = avg_ms * (num_rounds - round);

            printf("Progress: %5d/%d (%4.1f%%) | Success: %4d (%5.1f%%) | Elapsed: %6.1fs | ETA: %6.1fs\n",
                   round, num_rounds, 100.0 * round / num_rounds, successful_sets, succ_rate,
                   elapsed_ms / 1000.0, eta_ms / 1000.0);
        }
    }

    // Final
    double total_ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    double succ_rate = 100.0 * (double)successful_sets / num_rounds;
    printf("------------------------------------------------------------\n");
    printf("COMPLETED!\n");
    printf("Total rounds: %d\n", num_rounds);
    printf("Successful motif sets: %d\n", successful_sets);
    printf("Success rate: %.2f%%\n", succ_rate);
    printf("Total time: %.1f seconds (%.1f minutes)\n", total_ms / 1000.0, total_ms / 60000.0);
    printf("Average time per round: %.3f seconds\n", total_ms / num_rounds / 1000.0);

    if (successful_sets > 0) {
        printf("\nSaving results to files...\n");
        save_results(all_keys_ok, all_payloads_ok, all_motifs_ok);
        printf("Results saved to:\n- successful_keys.txt\n- successful_payloads.txt\n- successful_motifs.txt\n");
        printf("(Showing first 10 successful sets)\n");

        if (!all_keys_ok.empty()) {
            printf("\nSample from first successful set:\n");
            printf("Keys: [");
            for (size_t i = 0; i < all_keys_ok[0].size(); ++i) {
                printf("'%s'", all_keys_ok[0][i].c_str());
                if (i + 1 < all_keys_ok[0].size()) printf(", ");
            }
            printf("]\n");
            printf("Payloads: [");
            for (size_t i = 0; i < 3 && i < all_payloads_ok[0].size(); ++i) {
                printf("'%s'", all_payloads_ok[0][i].c_str());
                if (i + 1 < 3 && i + 1 < all_payloads_ok[0].size()) printf(", ");
            }
            printf("]...\n");
            printf("Motifs: [");
            for (size_t i = 0; i < 5 && i < all_motifs_ok[0].size(); ++i) {
                printf("'%s'", all_motifs_ok[0][i].c_str());
                if (i + 1 < 5 && i + 1 < all_motifs_ok[0].size()) printf(", ");
            }
            printf("]...\n");
        }
    }

    cudaFree(d_payloads); cudaFree(d_keys); cudaFree(d_states);
    cudaFree(d_c); cudaFree(d_h);
    cudaFree(d_key_success); cudaFree(d_payload_success);

    printf("\n Finish.\n");
    return 0;
}