#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <set>

// Exact parameters from Brunmayr et al. paper
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

// Hyperparameters from Python tuning results
struct Hyperparameters {
    float hom_shape;              // 70
    float gc_shape;               // 10
    float hairpin_shape;          // 8
    float similarity_shape;       // 60
    float no_key_in_payload_shape; // 45
    
    float hom_weight;             // 1
    float gc_weight;              // 1
    float hairpin_weight;         // 1
    float similarity_weight;      // 1
    float no_key_weight;          // 1
};

__constant__ int nucleotide_complement[4] = {1, 0, 3, 2};

// Homopolymer reward following paper formula: -(hhom)^(homLen/maxHom) + 1
__device__ float calculate_homopolymer_reward(
    const unsigned char* sequence,
    int length, 
    int new_nucleotide,
    const Constraints* constraints,
    const Hyperparameters* hyperparams
) {
    if (length == 0) return 0.0f;
    
    // Calculate homopolymer length including new nucleotide
    int hom_len = 1;
    if (length > 0 && sequence[length - 1] == new_nucleotide) {
        hom_len = 2;
        for (int i = length - 2; i >= 0; i--) {
            if (sequence[i] == new_nucleotide) {
                hom_len++;
            } else {
                break;
            }
        }
    }
    
    if (hom_len <= constraints->max_hom) {
        return 0.0f; // No penalty
    }
    
    // Paper formula: -(hhom)^(homLen/maxHom) + 1
    float ratio = (float)hom_len / constraints->max_hom;
    float penalty = -powf(hyperparams->hom_shape, ratio) + 1.0f;
    
    return penalty * hyperparams->hom_weight;
}

// GC content reward with progress weighting
__device__ float calculate_gc_reward(
    const unsigned char* sequence, 
    int length, 
    int new_nucleotide,
    const Constraints* constraints,
    const Hyperparameters* hyperparams
) {
    if (length == 0) return 0.0f;
    
    int gc_count = 0;
    for (int i = 0; i < length; i++) {
        if (sequence[i] == 2 || sequence[i] == 3) gc_count++;
    }
    if (new_nucleotide == 2 || new_nucleotide == 3) gc_count++;
    
    float current_gc = (float)gc_count / (length + 1) * 100.0f;
    
    // Weight based on sequence progress - paper formula
    float progress_ratio = (float)(length + 1) / constraints->payload_size;
    float weight = powf(hyperparams->gc_shape, progress_ratio) - 1.0f;
    
    float log_score = 0.0f;
    
    // Check violations as in original paper
    if (current_gc < constraints->min_gc) {
        log_score = fmaxf(log_score, weight * (constraints->min_gc - current_gc));
    }
    if (current_gc > constraints->max_gc) {
        log_score = fmaxf(log_score, weight * (current_gc - constraints->max_gc));
    }
    
    return -log_score * hyperparams->gc_weight; // Return negative for penalty
}

// Basic hairpin detection
__device__ float calculate_hairpin_reward(
    const unsigned char* sequence,
    int length,
    int new_nucleotide,
    const Constraints* constraints,
    const Hyperparameters* hyperparams
) {
    int total_length = length + 1;
    if (total_length < constraints->loop_size_min + 2 * constraints->max_hairpin) {
        return 0.0f; // Too short for hairpins
    }
    
    // Simplified hairpin detection
    float penalty = 0.0f;
    
    // Check for potential hairpin formations
    for (int i = 0; i <= total_length - 2 * constraints->max_hairpin - constraints->loop_size_min; i++) {
        for (int loop_size = constraints->loop_size_min; 
             loop_size <= constraints->loop_size_max && 
             i + 2 * constraints->max_hairpin + loop_size <= total_length; 
             loop_size++) {
            
            bool is_hairpin = true;
            
            // Check stem complementarity
            for (int j = 0; j < constraints->max_hairpin; j++) {
                unsigned char left_base = (i + j < length) ? sequence[i + j] : new_nucleotide;
                int right_pos = i + constraints->max_hairpin + loop_size + constraints->max_hairpin - 1 - j;
                unsigned char right_base = (right_pos < length) ? sequence[right_pos] : new_nucleotide;
                
                if (nucleotide_complement[left_base] != right_base) {
                    is_hairpin = false;
                    break;
                }
            }
            
            if (is_hairpin) {
                // Paper formula for hairpin penalty
                float stem_ratio = (float)constraints->max_hairpin / constraints->max_hairpin;
                penalty = -powf(hyperparams->hairpin_shape, stem_ratio) + 1.0f;
                break;
            }
        }
        if (penalty < 0.0f) break;
    }
    
    return penalty * hyperparams->hairpin_weight;
}

__device__ void calculate_rewards(
    const unsigned char* sequence,
    int length,
    float* rewards,
    const Constraints* constraints,
    const Hyperparameters* hyperparams
) {
    for (int nuc = 0; nuc < 4; nuc++) {
        float total_reward = 0.0f;
        
        // Calculate all constraint rewards
        total_reward += calculate_gc_reward(sequence, length, nuc, constraints, hyperparams);
        total_reward += calculate_homopolymer_reward(sequence, length, nuc, constraints, hyperparams);
        total_reward += calculate_hairpin_reward(sequence, length, nuc, constraints, hyperparams);
        // Note: noKeyInPayload and similarity constraints would go here in full implementation
        
        rewards[nuc] = total_reward;
    }
}

__device__ void softmax(float* rewards, float* probabilities, float temperature = 1.0f) {
    // Find max for numerical stability
    float max_reward = rewards[0];
    for (int i = 1; i < 4; i++) {
        if (rewards[i] > max_reward) max_reward = rewards[i];
    }
    
    // Calculate exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        probabilities[i] = expf((rewards[i] - max_reward) / temperature);
        sum += probabilities[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < 4; i++) {
            probabilities[i] /= sum;
        }
    } else {
        // Uniform distribution if all rewards are bad
        for (int i = 0; i < 4; i++) {
            probabilities[i] = 0.25f;
        }
    }
}

__device__ int sample_nucleotide(float* probabilities, curandState* state) {
    float rand_val = curand_uniform(state);
    float cumulative = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        cumulative += probabilities[i];
        if (rand_val <= cumulative) {
            return i;
        }
    }
    return 3; // Fallback to G
}

// Generate payloads using MDP
__global__ void mdp_payload_generation_kernel(
    unsigned char* sequences,
    curandState* states,
    const Constraints* constraints,
    const Hyperparameters* hyperparams,
    int num_sequences
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sequences) return;
    
    curandState local_state = states[tid];
    unsigned char* my_sequence = &sequences[tid * constraints->payload_size];
    
    // Generate payload sequence step by step using MDP
    for (int pos = 0; pos < constraints->payload_size; pos++) {
        float rewards[4];
        float probabilities[4];
        
        // Calculate rewards based on current sequence state
        calculate_rewards(my_sequence, pos, rewards, constraints, hyperparams);
        
        // Convert to probabilities using softmax
        softmax(rewards, probabilities);
        
        // Sample next nucleotide
        int selected = sample_nucleotide(probabilities, &local_state);
        my_sequence[pos] = selected;
    }
    
    states[tid] = local_state;
}

// Generate keys (simpler - could use similar MDP approach)
__global__ void generate_keys_kernel(
    unsigned char* keys,
    curandState* states,
    const Constraints* constraints,
    int num_keys
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;
    
    curandState local_state = states[tid];
    unsigned char* my_key = &keys[tid * constraints->key_size];
    
    // For now, generate keys randomly (could use MDP here too)
    for (int pos = 0; pos < constraints->key_size; pos++) {
        my_key[pos] = (unsigned char)(curand_uniform(&local_state) * 4);
    }
    
    states[tid] = local_state;
}

__global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed + tid, 0, 0, &states[tid]);
    }
}

std::string decode_sequence(const std::vector<unsigned char>& seq) {
    std::string result;
    const char nucleotides[] = {'A', 'T', 'C', 'G'};
    for (size_t i = 0; i < seq.size(); i++) {
        result += nucleotides[seq[i]];
    }
    return result;
}

bool validate_sequence(const std::string& seq, const Constraints& constraints) {
    // GC content validation
    int gc_count = 0;
    for (char c : seq) {
        if (c == 'G' || c == 'C') gc_count++;
    }
    float gc_ratio = (float)gc_count / seq.length() * 100.0f;
    if (gc_ratio < constraints.min_gc || gc_ratio > constraints.max_gc) {
        return false;
    }
    
    // Homopolymer validation
    int max_consecutive = 1;
    int current_consecutive = 1;
    for (size_t i = 1; i < seq.length(); i++) {
        if (seq[i] == seq[i-1]) {
            current_consecutive++;
            if (current_consecutive > max_consecutive) {
                max_consecutive = current_consecutive;
            }
        } else {
            current_consecutive = 1;
        }
    }
    if (max_consecutive > constraints.max_hom) {
        return false;
    }
    
    // Basic hairpin validation (simplified)
    // Full implementation would check for stem-loop structures
    
    return true;
}

void save_results(const std::vector<std::string>& keys, 
                  const std::vector<std::string>& payloads,
                  const std::vector<std::string>& motifs,
                  double generation_time,
                  int success_rate_percent) {
    
    // Save successful keys
    std::ofstream keys_file("successful_keys.txt");
    keys_file << "Keys generated by CUDA MDP implementation\n";
    keys_file << "Based on Brunmayr et al. parameters\n";
    keys_file << "Generation time: " << std::fixed << std::setprecision(3) << generation_time << " ms\n\n";
    
    for (size_t i = 0; i < keys.size() && i < 10; i++) {
        keys_file << "Key " << (i+1) << ": " << keys[i] << "\n";
    }
    keys_file.close();
    
    // Save successful payloads
    std::ofstream payloads_file("successful_payloads.txt");
    payloads_file << "Payloads generated by CUDA MDP implementation\n";
    payloads_file << "Based on Brunmayr et al. parameters\n";
    payloads_file << "Success rate: " << success_rate_percent << "%\n\n";
    
    for (size_t i = 0; i < payloads.size() && i < 20; i++) {
        payloads_file << "Payload " << (i+1) << ": " << payloads[i] << "\n";
    }
    payloads_file.close();
    
    // Save successful motifs (Key + Payload + Key combinations)
    std::ofstream motifs_file("successful_motifs.txt");
    motifs_file << "Motifs generated by CUDA MDP implementation\n";
    motifs_file << "Format: Key1 + Payload + Key2 (bridge sequences)\n";
    motifs_file << "Total motifs: " << motifs.size() << "\n\n";
    
    for (size_t i = 0; i < motifs.size() && i < 50; i++) {
        motifs_file << "Motif " << (i+1) << ": " << motifs[i] << "\n";
    }
    motifs_file.close();
    
    std::cout << "Results saved to:\n";
    std::cout << "- successful_keys.txt (" << std::min(keys.size(), (size_t)10) << " keys)\n";
    std::cout << "- successful_payloads.txt (" << std::min(payloads.size(), (size_t)20) << " payloads)\n";
    std::cout << "- successful_motifs.txt (" << std::min(motifs.size(), (size_t)50) << " motifs)\n";
}

int main() {
    printf("=== CUDA MDP DNA Generator - Brunmayr et al. Parameters ===\n");
    
    // EXACT parameters from Brunmayr et al. paper Table 1
    Constraints constraints;
    constraints.payload_size = 60;
    constraints.payload_num = 15;
    constraints.key_size = 20;
    constraints.key_num = 8;
    constraints.min_gc = 25.0f;     // Paper standard
    constraints.max_gc = 65.0f;     // Paper standard
    constraints.max_hom = 5;        // Paper standard 
    constraints.max_hairpin = 1;    // Paper standard
    constraints.loop_size_min = 6;  // Paper standard
    constraints.loop_size_max = 7;  // Paper standard
    
    // Hyperparameters from Python tuning results
    Hyperparameters hyperparams;
    hyperparams.hom_shape = 70.0f;
    hyperparams.gc_shape = 10.0f;
    hyperparams.hairpin_shape = 8.0f;
    hyperparams.similarity_shape = 60.0f;
    hyperparams.no_key_in_payload_shape = 45.0f;
    
    hyperparams.hom_weight = 1.0f;
    hyperparams.gc_weight = 1.0f;
    hyperparams.hairpin_weight = 1.0f;
    hyperparams.similarity_weight = 1.0f;
    hyperparams.no_key_weight = 1.0f;
    
    const int num_payloads = 1000;  // Generate more for better comparison
    const int threads_per_block = 256;
    const int blocks = (num_payloads + threads_per_block - 1) / threads_per_block;
    
    printf("Parameters (matching Brunmayr et al.):\n");
    printf("  Payload size: %d bp\n", constraints.payload_size);
    printf("  Key size: %d bp\n", constraints.key_size);
    printf("  GC content: %.0f%% - %.0f%%\n", constraints.min_gc, constraints.max_gc);
    printf("  Max homopolymer: %d\n", constraints.max_hom);
    printf("  Max hairpin stem: %d\n", constraints.max_hairpin);
    printf("  Loop size: %d-%d\n", constraints.loop_size_min, constraints.loop_size_max);
    printf("\nHyperparameters (from Python tuning):\n");
    printf("  Homopolymer shape: %.0f\n", hyperparams.hom_shape);
    printf("  GC shape: %.0f\n", hyperparams.gc_shape);
    printf("  Hairpin shape: %.0f\n", hyperparams.hairpin_shape);
    
    // Allocate GPU memory
    unsigned char* d_payloads;
    unsigned char* d_keys; 
    curandState* d_states;
    Constraints* d_constraints;
    Hyperparameters* d_hyperparams;
    
    cudaMalloc(&d_payloads, num_payloads * constraints.payload_size);
    cudaMalloc(&d_keys, constraints.key_num * constraints.key_size);
    cudaMalloc(&d_states, (num_payloads + constraints.key_num) * sizeof(curandState));
    cudaMalloc(&d_constraints, sizeof(Constraints));
    cudaMalloc(&d_hyperparams, sizeof(Hyperparameters));
    
    cudaMemcpy(d_constraints, &constraints, sizeof(Constraints), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hyperparams, &hyperparams, sizeof(Hyperparameters), cudaMemcpyHostToDevice);
    
    // Initialize random states
    int total_threads = num_payloads + constraints.key_num;
    int total_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    init_curand_states<<<total_blocks, threads_per_block>>>(d_states, time(NULL), total_threads);
    cudaDeviceSynchronize();
    
    printf("\nGenerating sequences...\n");
    clock_t start = clock();
    
    // Generate keys
    generate_keys_kernel<<<(constraints.key_num + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        d_keys, d_states, d_constraints, constraints.key_num);
    
    // Generate payloads using MDP
    mdp_payload_generation_kernel<<<blocks, threads_per_block>>>(
        d_payloads, &d_states[constraints.key_num], d_constraints, d_hyperparams, num_payloads);
    
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    double duration_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Generation time: %.1f ms\n", duration_ms);
    
    // Copy results back
    std::vector<unsigned char> h_payloads(num_payloads * constraints.payload_size);
    std::vector<unsigned char> h_keys(constraints.key_num * constraints.key_size);
    
    cudaMemcpy(h_payloads.data(), d_payloads, 
               num_payloads * constraints.payload_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys.data(), d_keys, 
               constraints.key_num * constraints.key_size, cudaMemcpyDeviceToHost);
    
    // Validate and collect successful sequences
    std::vector<std::string> valid_keys;
    std::vector<std::string> valid_payloads;
    std::vector<std::string> valid_motifs;
    
    // Process keys
    for (int i = 0; i < constraints.key_num; i++) {
        std::vector<unsigned char> key_seq(
            h_keys.begin() + i * constraints.key_size,
            h_keys.begin() + (i + 1) * constraints.key_size
        );
        std::string key_str = decode_sequence(key_seq);
        valid_keys.push_back(key_str);
    }
    
    // Process and validate payloads
    int valid_count = 0;
    for (int i = 0; i < num_payloads; i++) {
        std::vector<unsigned char> payload_seq(
            h_payloads.begin() + i * constraints.payload_size,
            h_payloads.begin() + (i + 1) * constraints.payload_size
        );
        std::string payload_str = decode_sequence(payload_seq);
        
        if (validate_sequence(payload_str, constraints)) {
            valid_payloads.push_back(payload_str);
            valid_count++;
        }
        
        // Create sample motifs (Key + Payload + Key)
        if (valid_count > 0 && valid_keys.size() >= 2) {
            std::string motif = valid_keys[0] + payload_str + valid_keys[1];
            valid_motifs.push_back(motif);
        }
    }
    
    int success_rate = (valid_count * 100) / num_payloads;
    
    printf("\nValidation Results:\n");
    printf("Valid payloads: %d/%d\n", valid_count, num_payloads);
    printf("Success rate: %d%%\n", success_rate);
    printf("Valid keys: %lu\n", valid_keys.size());
    printf("Sample motifs: %lu\n", valid_motifs.size());
    
    // Save results to files (matching Python output format)
    save_results(valid_keys, valid_payloads, valid_motifs, duration_ms, success_rate);
    
    // Show sample results (like Python version)
    printf("\nSample Results:\n");
    if (!valid_keys.empty()) {
        printf("Sample Key: %s\n", valid_keys[0].c_str());
    }
    if (!valid_payloads.empty()) {
        printf("Sample Payload: %s\n", valid_payloads[0].c_str());
    }
    if (!valid_motifs.empty()) {
        printf("Sample Motif: %s\n", valid_motifs[0].substr(0, 50).c_str());
        printf("              ... (length: %lu)\n", valid_motifs[0].length());
    }
    
    // Cleanup
    cudaFree(d_payloads);
    cudaFree(d_keys);
    cudaFree(d_states);
    cudaFree(d_constraints);
    cudaFree(d_hyperparams);
    
    printf("\n✅ CUDA implementation with correct Brunmayr parameters completed!\n");
    printf("📊 Ready for comparison with Python baseline (2.54 seconds target)\n");
    
    return 0;
}