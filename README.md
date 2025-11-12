<h1 align="center">
   Look As You Think: Unifying Reasoning and Visual Evidence Attribution for Verifiable Document RAG via Reinforcement Learning
</h1>

## Overview

TL;DR: In this paper, we introduce the **Chain of Evidence (CoE)** paradigm, which models stepwise inference by grounding each chain-of-thought (CoT) reasoning step. To realize CoE, we propose **Look As You Think (LAT)**, a two-stage reinforcement learning (RL) framework that trains VLMs to unify CoT reasoning and visual grounding by generating progressive reasoning process paired with an aligned visual attribution for each reference element.


## Dependencies

The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt). 
To install all the required packages along with their dependencies, run
```sh
pip install -r requirements.txt
```

## Run
**1. Download Data**

Prepare Paper-VISA and Wiki-VISA datasets (Hugging face)

To obtain images for the multi-candidate setup, please run `/src/image_address.py`.

**2. Cold start**
```sh
bash scripts/sft_inference.sh
```

**3. Reinforcement Learning**
```sh
bash scripts/grpo.sh
```

**4. Evaluate Model**
```sh
python test.py
```

## Acknowledgement
Our code have been developed based on VLM-R1, VISA. We thank their valuable works.


