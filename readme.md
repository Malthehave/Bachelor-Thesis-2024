# Efficient Inference in Transformer-Based Models: Using Compact Models to Approximate Encoder Blocks

This repository contains the code and implementation details for the my bachelor's thesis paper "Efficient Inference in Transformer-Based Models: Using Compact Models to Approximate Encoder Blocks". The goal of this project is to reduce the computational demands of transformer-based language models like BERT by substituting large encoder blocks with smaller, more efficient models while maintaining high performance.

## Overview

Transformer-based language models achieve state-of-the-art performance in various NLP tasks but are computationally intensive. I introduce a method that combines structured pruning with knowledge distillation to create more efficient models. By substituting BERT's encoder blocks with compact, transformer-based encoders, I am able to significantly reduce the model's GMACs and parameter count while maintaining its performance.

## Key Results

- **BERT**: Reduced GMACs by 53.1% and parameter count by 41.5%, maintaining 92.94% of original performance on the IMDB sentiment analysis task.
- **DistilBERT**: Additional parameter reduction of 19.4%, achieving a total reduction of 49.51% in parameters and 65.17% in GMACs efficiency compared to the original BERT, while retaining 98.08% of its performance.

## Methodology

The process involves:
1. **Down-Projection**: Adapting the input dimensionality from the original model's encoder block to the compact encoder's reduced hidden size.
2. **Compact Encoding**: Processing the down-projected inputs through compact transformer encoder blocks.
3. **Up-Projection**: Converting the reduced dimensionality back to the original block's hidden size to ensure compatibility with subsequent layers.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Malthehave/Bachelor-Thesis-2024.git
cd Bachelor-Thesis-2024
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Paper Abstract
> Transformer-based language models excel in various NLP tasks but face substantial computational demands. This work introduces a method to prune these models by systematically substituting large encoder blocks with smaller, more efficient models, significantly reducing computational complexity. Integrating structured pruning with knowledge distillation, this approach retains essential functionalities while optimizing performance.
> 
> This study shows that substituting BERT's encoder blocks with compact transformers reduces GMACs by 53.1% and parameter count by 41.5%, maintaining 92.94% of its original performance on the IMDB sentiment analysis task. This process involves down-projecting inputs, processing them through compact encoders, and up-projecting them back to the original dimensions. Applying this method to DistilBERT results in an additional 19.4% parameter reduction, achieving a total reduction of 49.51% in parameters and 65.17% in GMACs efficiency compared to the original BERT, while retaining 98.08% of its performance. This approach enhances inference efficiency with minimal accuracy loss, making transformer models more accessible for resource-constrained environments. The code, integrated with Transformers and PyTorch, is available at [https://github.com/Malthehave/Bachelor-Thesis-2024](https://github.com/Malthehave/Bachelor-Thesis-2024).






