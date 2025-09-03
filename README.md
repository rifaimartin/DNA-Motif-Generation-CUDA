# DNA-Motif-Generation-CUDA

GPU-accelerated DNA motif generation using CUDA implementation of Markov Decision Process (MDP) for motif-based DNA storage. Based on Brunmayr et al. (2025) Motif Generation Tool with significant performance improvements.
This program implements a Markov Decision Process (MDP) to generate DNA motifs using CUDA for parallelization. The program generates keys and payloads that satisfy biological constraints and then combines them into motifs for DNA storage.

## Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA capability
- CUDA Toolkit 11.0+

### Installation

1. **Install CUDA Toolkit**
   ```bash
   # Download from: https://developer.nvidia.com/cuda-toolkit
   # Or use conda:
   conda install cudatoolkit
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install asyncio
   ```

3. **Verify CUDA Installation**
   ```bash
   nvcc --version
   nvidia-smi
   ```

## Usage

### Run CUDA Implementation
```bash
cd cuda_implementation
nvcc -o mdp mdp.cu -lcurand
./mdp
```

### Run Python Baseline (for comparison)
```bash
cd motif_generation_tool
python -m key_payload_builder run
```

### Run Tests
```bash
python3 -m pytest unit_tests/hairpin_tests.py
python3 -m pytest unit_tests/[test_file].py
```

## Project Structure
```
├── cuda_implementation/        # CUDA C++ implementation
│   └── mdp.cu                 # Main CUDA code
├── motif_generation_tool/     # Original Python baseline
├── unit_tests/               # Test files
├── requirements.txt          # Python dependencies
└── README.md
```

## Performance
- **Target**: 10-50x speedup vs CPU implementation
- **Generation time**: <2 seconds (vs >5 minutes with existing tools)
- **Constraint compliance**: ≥99%

## Tools
- **Generation**: https://ssb5018.pythonanywhere.com/
- **Validation**: http://ssb22.pythonanywhere.com/

## Souce Code Brunmayr et al. paper

-- **motiif-generation-tool** : https://github.com/ssb5018/motif-generation-tool
-- **motiif-generation-tool** : https://github.com/ssb5018/dna-validation

## Parameters
Based on Brunmayr et al. paper with hyperparameter tuning:
- Payload: 60bp, Key: 20bp
- GC content: 25-65%
- Max homopolymer: 5, Max hairpin: 1
- Optimized shape parameters from hyperparameter tuning