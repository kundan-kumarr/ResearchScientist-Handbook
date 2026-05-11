# From First Principles: A Research Scientist Interview Guide

**2026 Edition — Kundan Kumar**

[![Days Complete](https://img.shields.io/badge/Edition-2026-1a1a2e?style=flat-square)](.)
[![Modules](https://img.shields.io/badge/Modules-10-2d6a4f?style=flat-square)](.)
[![Coverage](https://img.shields.io/badge/ML%20%7C%20DL%20%7C%20DRL%20%7C%20LLM-Full%20Stack-0077b6?style=flat-square)](.)
[![GitHub](https://img.shields.io/badge/-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/kundan-kumarr)
[![Substack](https://img.shields.io/badge/-FF6719?style=flat-square&logo=substack&logoColor=white)](https://neuravp.substack.com/)

---

> "Think mathematically, reason scientifically, and build efficiently."

This is a structured preparation curriculum for Research Scientist roles at industrial AI labs, academic research groups, and applied ML organizations. It is built on a single conviction: every algorithm evolved to solve a pain point in its predecessor, and understanding that lineage is what separates a candidate who can recall methods from one who can reason about them.

The guide balances mathematical theory, statistical rigor, and hands-on implementation — tracing the evolution from classical machine learning through modern large language models and deep reinforcement learning systems.

---

## The Lineage

Every method in this guide is positioned within an evolutionary chain:

```
Linear Models  ->  Nonlinear Trees  ->  Ensemble Boosting  ->  Neural Networks
     ->  Transformers  ->  LLMs  ->  DRL  ->  Physics-Informed & Federated Systems
```

Each chapter answers: *why was the next method needed, and what problem did it solve that its predecessor could not?*

---

## Learning Philosophy

1. **Start from intuition.** Understand why each model exists before touching the math.
2. **Derive manually.** Write gradients, losses, and activations by hand before running code.
3. **Implement from scratch.** Rebuild models in PyTorch — no sklearn black boxes for core concepts.
4. **Visualize.** Use loss landscapes, weight evolution plots, and activation maps to build spatial intuition.
5. **Communicate.** Summarize each section as if presenting to a research committee or PhD committee.

---

## Curriculum Overview

| Module | Theme | Core Topics |
|--------|-------|-------------|
| 1 | Mathematical Foundations | Linear algebra, probability theory, optimization, gradient calculus |
| 2 | Classical Machine Learning | Linear/Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost, CatBoost, PCA, ICA, t-SNE, UMAP, K-Means, GMM |
| 3 | Deep Learning | Perceptrons, MLPs, CNNs, RNN/LSTM, Transformers from scratch |
| 4 | Advanced DL Concepts | Activation functions, normalization, regularization, loss functions, optimizers |
| 5 | Deep Reinforcement Learning | Q-Learning, DDPG, PPO, SAC, GRPO, Model-Based RL |
| 6 | Large Language Models | Transformer theory, attention mechanisms, fine-tuning, LoRA, RLHF |
| 7 | Model Compression and Deployment | Pruning, quantization, distillation, ONNX, TensorRT, inference profiling |
| 8 | Research Design and Experimentation | Ablation studies, reproducibility, explainability, uncertainty quantification |
| 9 | Trustworthy and Robust AI | Adversarial robustness, bias mitigation, differential privacy |
| 10 | Emerging Topics | Neural Operators, Diffusion Models, Causal Learning, Multimodal Systems |
| A | Appendices | Derivations, key equations, code templates, reference papers |

---

## Module 1 — Mathematical Foundations

> The backbone of every algorithm. A weak mathematical foundation produces candidates who can run code but cannot reason about why it fails.

**Linear Algebra**
- Matrix decompositions: SVD, eigendecomposition, Cholesky
- Orthogonality, projections, and the four fundamental subspaces
- Low-rank approximation and the connection to PCA, LoRA, and attention

**Probability and Statistics**
- Probability distributions: Gaussian, Bernoulli, Categorical, Dirichlet
- Bayesian inference: prior, likelihood, posterior, MAP vs MLE
- Information theory: entropy, KL divergence, mutual information

**Optimization**
- Gradient descent variants: SGD, momentum, RMSProp, Adam
- Convex vs non-convex landscapes; saddle points and flat minima
- Lagrangian methods and constrained optimization (relevant to safe RL)

**Interview Focus:** Derive the gradient of cross-entropy loss. Explain why Adam converges faster than SGD but may generalize worse. What is the KL divergence and when is it asymmetric?

---

## Module 2 — Classical Machine Learning

> Classical methods remain the interview baseline and the performance ceiling for structured/tabular data.

**The Lineage**

| Transition | Problem Solved |
|------------|----------------|
| Linear Regression to Logistic Regression | Binary classification with probabilistic output |
| Logistic Regression to SVM | Margin maximization; kernel trick for nonlinearity |
| SVM to Decision Trees | Interpretability and feature interactions without scaling |
| Trees to Random Forest | Variance reduction via bagging and feature subsampling |
| Random Forest to XGBoost | Bias reduction via gradient boosting; sequential weak learners |
| XGBoost to CatBoost | Native categorical handling; ordered boosting for leakage prevention |

**Key Derivations**
- Logistic regression: sigmoid function, log-likelihood, gradient derivation
- SVM: margin formulation, dual problem, KKT conditions, kernel functions
- Decision tree: information gain, Gini impurity, CART splitting criterion
- Gradient boosting: functional gradient descent, additive tree construction

**Bias-Variance Analysis** — For each method: where does it sit on the bias-variance spectrum and why?

**Interview Focus:** Implement a decision tree from scratch. Compare RF and XGBoost for a dataset with 50k rows and 500 features. Derive the SVM dual problem.

---

### Dimensionality Reduction

> High-dimensional data is the norm in ML. Dimensionality reduction is both a preprocessing tool and a window into representation structure — understanding it deeply bridges classical ML and modern neural representation learning.

**Principal Component Analysis (PCA)**

PCA finds the orthogonal directions of maximum variance in the data. It is the canonical linear dimensionality reduction method and appears in interviews more often than any other unsupervised technique.

*Derivation from first principles:*
- Center the data: X_c = X - mean(X)
- Compute the covariance matrix: C = (1/n) * X_c^T * X_c, shape (d x d)
- Eigendecompose: C = V * Lambda * V^T, where columns of V are eigenvectors
- Project: Z = X_c * V_k, keeping the top-k eigenvectors by eigenvalue magnitude
- Equivalently via SVD: X_c = U * Sigma * V^T; principal components are columns of V

*What PCA actually optimizes:*
- Maximum variance direction: arg max_{||w||=1} w^T C w — solved by the leading eigenvector
- Minimum reconstruction error: ||X_c - X_c * V_k * V_k^T||_F^2 is minimized by the same V_k
- These two objectives are equivalent — variance maximization = reconstruction error minimization

*Choosing the number of components:*
- Scree plot: plot eigenvalues in descending order, look for the elbow
- Explained variance ratio: cumulative sum of (lambda_i / sum(lambda)); target 90-95%
- Cross-validation on downstream task performance

*Assumptions and limitations:*
- Linear: only captures linear correlations in the data
- Gaussian: optimally suited for elliptically distributed data
- Global: a single projection for the entire dataset
- Sensitive to scale: always standardize features before applying PCA

*Connection to neural methods:*
- Linear autoencoder with tied weights and no nonlinearity recovers PCA solution
- LoRA weight updates have the same low-rank structure as PCA projections
- Logit Lens in transformers is a form of PCA-style linear projection of the residual stream

**Kernel PCA**
- Applies PCA in a feature space defined by a kernel function k(x, x')
- Allows nonlinear manifold structure to be captured
- Common kernels: RBF (Gaussian), polynomial, sigmoid
- Limitation: the kernel matrix is n x n — does not scale to large datasets

**Independent Component Analysis (ICA)**
- PCA finds uncorrelated components; ICA finds statistically independent components
- Assumption: source signals are non-Gaussian and independent
- Algorithm: FastICA — iterative fixed-point optimization of non-Gaussianity (kurtosis or negentropy)
- Use cases: blind source separation (EEG, audio), latent factor discovery
- Key distinction from PCA: ICA is not unique up to rotation; PCA components are ordered by variance, ICA components are not

**Linear Discriminant Analysis (LDA)**
- Supervised dimensionality reduction: maximize class separability
- Maximizes between-class scatter / within-class scatter: J(w) = w^T S_B w / w^T S_W w
- Solves a generalized eigenvalue problem: S_W^{-1} S_B w = lambda w
- Upper bound on components: min(n_classes - 1, d)
- When to use: labeled data, classes are approximately Gaussian with shared covariance

**Comparison: PCA vs LDA vs ICA**

| Method | Supervision | Criterion | Components Ordered? | Assumption |
|--------|-------------|-----------|---------------------|------------|
| PCA | Unsupervised | Maximum variance | Yes, by variance | Linear, Gaussian |
| LDA | Supervised | Class separability | Yes, by discriminability | Gaussian, equal covariance |
| ICA | Unsupervised | Statistical independence | No | Non-Gaussian sources |
| Kernel PCA | Unsupervised | Variance in kernel space | Yes | Kernel choice |

---

### Nonlinear Dimensionality Reduction

Linear methods fail when the data lies on a curved manifold in high-dimensional space. Nonlinear methods recover this structure.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**

t-SNE is the standard tool for visualizing high-dimensional embeddings (word vectors, image features, activations).

*How it works:*
- Compute pairwise similarities in high-d space using a Gaussian kernel: p_{ij} proportional to exp(-||x_i - x_j||^2 / 2 sigma^2)
- Model pairwise similarities in 2D using a Student-t distribution (heavier tails): q_{ij} proportional to (1 + ||y_i - y_j||^2)^{-1}
- Minimize KL divergence: KL(P || Q) = sum_{i,j} p_{ij} log(p_{ij} / q_{ij})
- The heavy-tailed q distribution is key: it allows dissimilar points to be placed far apart without fighting a crowding problem

*Hyperparameters that matter:*
- Perplexity (5–50): effective number of neighbors; controls local vs. global structure balance
- Learning rate: must be tuned; too low → clumped layout; too high → unstable
- Early exaggeration: inflates p values at start to help clusters form

*What t-SNE does not preserve:*
- Global structure: distances between clusters are not meaningful
- Cluster sizes: larger clusters in 2D do not mean larger clusters in high-d
- Not suitable for downstream ML: use only for visualization

**UMAP (Uniform Manifold Approximation and Projection)**

UMAP is increasingly preferred over t-SNE for large datasets and cases where global structure matters.

*Core idea:*
- Constructs a weighted graph of k-nearest neighbors in high-d space
- Constructs a similar graph in low-d space
- Optimizes cross-entropy between the two fuzzy set representations

*UMAP vs t-SNE:*

| Property | t-SNE | UMAP |
|----------|-------|------|
| Speed | O(n log n) with Barnes-Hut | Faster in practice |
| Scalability | Struggles above 100k samples | Scales to millions |
| Global structure | Poorly preserved | Better preserved |
| Determinism | Stochastic (seed-dependent) | More stable across runs |
| New point projection | Not supported natively | Supported |
| Theoretical grounding | Information-theoretic | Topological / Riemannian |

**ISOMAP and LLE**
- ISOMAP: computes geodesic distances along the manifold (shortest paths in k-NN graph), then applies MDS
- LLE (Locally Linear Embedding): reconstructs each point as a linear combination of its neighbors, then finds a low-d embedding that preserves those coefficients
- Both are sensitive to noise and non-uniform sampling; largely superseded by UMAP for practical use

---

### Clustering

**K-Means**
- Objective: minimize within-cluster sum of squared distances: sum_k sum_{x in C_k} ||x - mu_k||^2
- Algorithm (Lloyd's): assign each point to nearest centroid, recompute centroids, repeat until convergence
- Convergence: guaranteed to converge, not guaranteed to find global optimum (use k-means++ initialization)
- Choosing k: elbow method on inertia, silhouette score, gap statistic
- Assumptions: spherical clusters, similar size and density, Euclidean distance is meaningful
- Time complexity: O(n * k * d * iterations)

**Gaussian Mixture Models (GMM)**
- Soft assignment: each point has a probability of belonging to each cluster
- Model: p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
- Fitting: EM algorithm — E-step computes responsibilities, M-step updates pi, mu, Sigma
- More flexible than K-Means: elliptical clusters, overlapping distributions
- Covariance types: spherical, diagonal, tied, full — trade-off between flexibility and parameters
- Model selection: BIC or AIC to choose number of components

**DBSCAN**
- Density-based: clusters are dense regions separated by sparse regions
- Parameters: eps (neighborhood radius), min_samples (core point threshold)
- Does not require specifying number of clusters; handles arbitrary shapes
- Labels noise points explicitly (label = -1)
- Fails with varying density clusters; sensitive to eps in high dimensions

**Hierarchical Clustering**
- Agglomerative (bottom-up): start with each point as its own cluster, merge closest pair
- Linkage criteria: single (min distance), complete (max distance), average, Ward (minimize variance increase)
- Produces a dendrogram: cut at any height to get any number of clusters
- Ward linkage tends to produce compact, similarly-sized clusters

**Clustering Evaluation Metrics**

| Metric | Requires Labels? | Measures |
|--------|-----------------|---------|
| Silhouette score | No | Cohesion vs. separation; range [-1, 1] |
| Davies-Bouldin index | No | Ratio of within-cluster to between-cluster distance |
| Calinski-Harabasz | No | Between-cluster dispersion / within-cluster dispersion |
| Adjusted Rand Index | Yes | Agreement with ground truth labels, corrected for chance |
| Normalized Mutual Information | Yes | Information shared between predicted and true clusters |

---

**Interview Focus:** Derive PCA from the variance maximization objective and show it is equivalent to SVD. Explain why t-SNE distances between clusters are not interpretable. When would you use GMM over K-Means? What breaks K-Means in high dimensions and how does the curse of dimensionality affect it?

---

## Module 3 — Deep Learning

> Neural networks as universal function approximators — but the approximation quality depends on architecture, initialization, and training dynamics.

**Perceptron to MLP**
- Single neuron: weighted sum + nonlinearity
- Universal approximation theorem: what it says and what it does not
- Backpropagation: chain rule applied to computation graphs

**Convolutional Neural Networks**
- Conv2D: local connectivity, weight sharing, equivariance to translation
- Receptive field growth through depth
- ResNet: residual connections as identity shortcuts; gradient highway

**Recurrent Networks**
- Vanishing gradient problem: why RNNs fail on long sequences
- LSTM: cell state as explicit memory; forget/input/output gates
- When to use RNN/LSTM vs Transformer (sequence length, parallelism constraints)

**Transformers**
- Scaled dot-product attention: Q, K, V projections and the softmax gate
- Multi-head attention: parallel feature subspaces
- Positional encoding: sinusoidal and learned
- Full GPT-2 architecture: embedding + N x (Attention + MLP) + LM head

**Interview Focus:** Derive backprop through an LSTM cell. Explain why LayerNorm is placed before (not after) each sub-layer in GPT. What is the computational complexity of self-attention and how does it scale?

---

## Module 4 — Advanced Deep Learning Concepts

> The details that separate theoretical knowledge from production-level understanding.

**Activation Functions**

| Function | Formula | Use Case | Failure Mode |
|----------|---------|----------|--------------|
| ReLU | max(0, x) | Default for most tasks | Dying ReLU for negative weights |
| GELU | x * Phi(x) | Transformers, BERT, GPT | Slightly slower |
| Swish | x * sigmoid(x) | EfficientNet, modern CNNs | Non-monotonic |
| Sigmoid | 1/(1+e^-x) | Binary output layer | Vanishing gradients |

**Normalization**
- Batch Norm: normalize over batch dimension; effective for CNNs
- Layer Norm: normalize over feature dimension; required for Transformers (batch-independent)
- Group Norm, Instance Norm: trade-offs for small-batch and style tasks

**Regularization**
- L1 (Lasso): sparse solutions; drives small weights to zero
- L2 (Ridge): shrinks all weights; closed-form solution exists
- Dropout: stochastic ensemble interpretation; effective at fc layers
- Weight decay: equivalent to L2 with SGD but not with adaptive optimizers

**Optimizers**
- SGD + momentum: simple, generalizes well, sensitive to learning rate
- Adam: adaptive per-parameter rates; fast convergence, may overfit
- AdaGrad, RMSProp: historical context and limitations
- Learning rate schedulers: cosine annealing, warmup, one-cycle

---

## Module 5 — Deep Reinforcement Learning

> Learning sequential decision policies from reward signals — the foundation of modern RLHF and autonomous control systems.

**MDP Formalism**
- State space S, action space A, transition function P, reward function R, discount factor gamma
- Bellman equations for V(s) and Q(s, a)
- Policy iteration and value iteration to convergence

**Value-Based Methods**
- Q-Learning: off-policy TD learning for tabular MDPs
- Deep Q-Network (DQN): function approximation + experience replay + target network
- Double DQN, Dueling DQN: variance reduction and advantage decomposition

**Policy Gradient Methods**
- REINFORCE: Monte Carlo policy gradient with baseline
- Actor-Critic (A2C, A3C): value network as variance-reducing baseline
- PPO: clipped surrogate objective; stability via KL constraint
- DDPG, TD3, SAC: continuous action spaces; deterministic vs stochastic policies

**Safe and Constrained DRL**
- Constrained MDP (CMDP): reward maximization under cost constraints
- Lagrangian relaxation: dual variable as adaptive penalty coefficient
- Lyapunov-based safety constraints: stability guarantees via energy functions
- Applications: smart energy grids, autonomous systems, resource allocation

**RLHF**
- Human preference data → Bradley-Terry reward model
- PPO fine-tuning with KL penalty: r_theta(x, y) - beta * KL[pi_theta || pi_ref]
- Reward hacking: when the proxy reward diverges from true human preference
- Constitutional AI and scalable oversight as alternative alignment approaches

**Interview Focus:** Derive the PPO clipped objective. Explain why experience replay is needed in DQN. Implement Q-learning on a gridworld from scratch.

---

## Module 6 — Large Language Models

> Pretraining, alignment, and the machinery behind modern language systems.

**Transformer Architecture (Revisited at Scale)**
- Residual stream: every layer writes to and reads from a shared information channel
- Attention as selective information routing: which tokens influence which?
- MLP as key-value memory: neurons activate on specific input patterns
- Logit Lens: tracing how predictions form across layers

**Pretraining**
- Next-token prediction on large corpora: self-supervised signal
- Scaling laws: loss ~ (compute)^alpha; optimal allocation of compute to data vs. parameters
- Data quality: deduplication, filtering, and its effect on downstream performance

**Fine-Tuning and Adaptation**
- Full fine-tuning: catastrophic forgetting and mitigation
- LoRA: low-rank decomposition of weight updates; W' = W + A x B, rank r << d
- QLoRA: 4-bit quantized base model + LoRA adapters for memory efficiency
- Instruction tuning: teaching models to follow user intent

**RLHF (from the LLM Side)**
- SFT stage: supervised fine-tuning on high-quality demonstrations
- Reward model: trained on pairwise comparisons via Bradley-Terry model
- RL stage: PPO with KL divergence penalty against SFT reference

**Evaluation**
- Perplexity: a necessary but insufficient measure of language quality
- MMLU, HellaSwag, HumanEval: capability benchmarks and their limitations
- Evaluation-awareness: models that behave differently when they detect they are being evaluated

**Interview Focus:** Explain LoRA's efficiency advantage. Derive attention complexity. What is the KL penalty in RLHF and why is it necessary?

---

## Module 7 — Model Compression and Deployment

> Inference is where theory meets production constraints. Efficiency is not optional at scale.

**Compression Techniques**

| Technique | Core Idea | Typical Gain | Trade-Off |
|-----------|-----------|--------------|-----------|
| Pruning | Remove redundant weights (magnitude, structured) | 2-10x smaller | May require retraining |
| Quantization | Reduce precision: FP32 to INT8 or INT4 | 2-4x smaller, 2-4x faster | Precision loss; calibration required |
| Knowledge Distillation | Student learns from teacher logits (soft targets) | 2-10x faster inference | Student training overhead |
| Low-Rank Factorization | Decompose weight matrices: W = U x V | 3-5x fewer parameters | Moderate accuracy loss |
| LoRA / QLoRA | Parameter-efficient adaptation via low-rank updates | Minimal storage | Applies to adaptation, not base model |

**Inference Optimization**
- ONNX: hardware-agnostic interchange format for model graphs
- TensorRT: NVIDIA inference optimizer; layer fusion, kernel auto-tuning
- FlashAttention: IO-aware attention; recomputes rather than materializing large attention matrices
- KV cache: store past key/value projections; critical for autoregressive generation

**Deployment Patterns**
- Batch inference vs. streaming generation
- Latency vs. throughput trade-offs: beam size, temperature, early stopping
- Model serving: TorchServe, TFServe, vLLM for high-throughput LLM inference
- Profiling: identify bottlenecks (compute-bound vs. memory-bound operations)

---

## Module 8 — Research Design and Experimentation

> The difference between running experiments and doing research is the ability to draw valid causal conclusions.

**Ablation Studies**
- What to ablate: components, hyperparameters, data sources, architectural choices
- Isolating variables: change one thing at a time
- Reporting: always show full ablation table, not just the winning configuration

**Reproducibility**
- Random seeds, hardware determinism, environment specification
- Reporting: mean and standard deviation over at least 5 seeds
- Checkpointing: save optimizer state, not just model weights

**Uncertainty Quantification**
- Aleatoric uncertainty: irreducible noise in the data
- Epistemic uncertainty: model uncertainty; reducible with more data
- Calibration: does a model's confidence match its empirical accuracy?
- Techniques: MC Dropout, deep ensembles, conformal prediction

**Explainability**
- SHAP: game-theoretic attribution of feature contributions
- LIME: local linear approximation of black-box decisions
- Captum: gradient-based attribution for PyTorch models
- Attention as explanation: what it does and does not tell you

---

## Module 9 — Trustworthy and Robust AI

> A model that performs well on the test set but fails under distribution shift or adversarial perturbation is not production-ready.

**Adversarial Robustness**
- FGSM: single-step gradient attack; Fast Gradient Sign Method
- PGD: projected gradient descent; the standard robustness benchmark
- Adversarial training: include adversarial examples in the training loop
- Certified robustness: provable bounds via randomized smoothing or interval arithmetic

**Distribution Shift**
- Covariate shift: P(X) changes, P(Y|X) unchanged
- Label shift: P(Y) changes, P(X|Y) unchanged
- Domain adaptation and domain generalization techniques

**Fairness and Bias**
- Sources: data collection, labeling bias, historical discrimination encoded in features
- Metrics: demographic parity, equalized odds, calibration across groups
- Mitigation: pre-processing (reweighting), in-processing (constraint), post-processing (threshold adjustment)

**Differential Privacy**
- (epsilon, delta)-DP: formal privacy guarantee bounding information leakage
- DP-SGD: gradient clipping + Gaussian noise injection during training
- Privacy-utility trade-off: as epsilon decreases, model utility decreases

---

## Module 10 — Emerging Topics

> Research directions active in 2025-2026 that are increasingly relevant to interview discussions at frontier labs.

**Diffusion Models**
- Score matching and the denoising objective
- DDPM: forward noising process + learned reverse denoising
- Connections to flow matching and continuous normalizing flows

**Neural Operators**
- FNO (Fourier Neural Operator): learn mappings between function spaces
- Applications: PDE surrogate modeling, climate, fluid dynamics
- Physics-informed neural networks: encoding conservation laws as soft constraints

**Causal Learning**
- Structural causal models (SCMs) and the do-calculus
- Difference between correlation and causal effect
- Why causal reasoning matters for out-of-distribution generalization

**Multimodal Systems**
- Vision-language models: CLIP, Flamingo, LLaVA architectures
- Cross-attention for modality fusion
- Evaluation challenges for multimodal reasoning

**Mechanistic Interpretability**
- Linear probing: does layer L linearly encode concept C?
- Sparse autoencoders: decomposing polysemantic neurons into interpretable features
- Circuit analysis: reverse-engineering specific model behaviors end-to-end

---

## Interview Preparation Framework

| Phase | Weeks | Focus | Key Deliverable |
|-------|-------|-------|-----------------|
| Phase 1 — Core Concepts | 1-4 | ML/DL fundamentals, mathematical review | Implement linear models and neural nets from scratch |
| Phase 2 — Deep Reinforcement Learning | 5-7 | PPO, DDPG, constrained RL | End-to-end DRL implementation with ablation study |
| Phase 3 — LLMs and Transformers | 8-10 | Pretraining, fine-tuning, LoRA, RLHF | GPT-2 from scratch + LoRA fine-tuning |
| Phase 4 — Research Communication | 11 | Technical writing, experiment narration | Short technical blog or mini-paper |
| Phase 5 — Systems and Scalability | 12-13 | Compression, quantization, deployment | Inference profiling report |
| Phase 6 — Mock Interviews I | 14 | 1 technical + 1 research + 1 system design | Self-evaluation rubric |
| Phase 7 — Mock Interviews II | 15 | Full simulation under time constraint | Final preparation checklist |

---

## Interview Themes

**Mathematical and Statistical Foundations**
Gradient derivation, loss design, convergence proofs, bias-variance decomposition, regularization, cross-validation.

**Algorithm Design**
Model selection under data constraints, trade-offs between interpretability and expressivity, when to use which method and why.

**Research Reasoning**
Designing experiments for robustness, formulating ablation hypotheses, drawing valid conclusions from results, identifying confounds.

**System Efficiency**
Compression, quantization, deployment, distributed training, monitoring in production.

**Ethics and Trustworthiness**
Explainability, fairness, robustness to distribution shift, differential privacy, alignment.

---

## Tools and Frameworks

| Category | Tools |
|----------|-------|
| Core ML / DL | PyTorch, TensorFlow |
| Deep RL | Stable-Baselines3, RLlib, custom implementations |
| LLM / PEFT | HuggingFace Transformers, PEFT, TRL |
| Deployment | ONNX, TensorRT, TorchServe, vLLM |
| Experimentation | Weights and Biases, MLflow |
| Hyperparameter Optimization | Optuna, Ray Tune |
| Explainability | SHAP, LIME, Captum |
| Interpretability | TransformerLens, nnsight |

---

## Reference Papers

| Area | Paper |
|------|-------|
| Attention | Attention Is All You Need — Vaswani et al., 2017 |
| Scaling | Scaling Laws for Neural Language Models — Kaplan et al., 2020 |
| LoRA | LoRA: Low-Rank Adaptation of Large Language Models — Hu et al., 2021 |
| RLHF | Learning to summarize from human feedback — Stiennon et al., 2020 |
| PPO | Proximal Policy Optimization Algorithms — Schulman et al., 2017 |
| SAC | Soft Actor-Critic — Haarnoja et al., 2018 |
| Robustness | Towards Deep Learning Models Resistant to Adversarial Attacks — Madry et al., 2018 |
| Superposition | Toy Models of Superposition — Elhage et al., Anthropic, 2022 |
| Circuits | A Mathematical Framework for Transformer Circuits — Elhage et al., 2021 |
| Grokking | Progress measures for grokking via mechanistic interpretability — Nanda et al., 2023 |
| Safe RL | Constrained Policy Optimization — Achiam et al., 2017 |
| Physics-Informed | Physics-Informed Neural Networks — Raissi et al., 2019 |

---

## Repository Structure

```
research-scientist-guide/
│
├── 01_math_foundations/        # Linear algebra, probability, optimization
├── 02_classical_ml/            # Regression, SVM, trees, ensemble methods
├── 03_deep_learning/           # MLP, CNN, RNN, Transformer from scratch
├── 04_advanced_dl/             # Normalization, activations, optimizers
├── 05_deep_rl/                 # Q-learning, PPO, SAC, constrained RL
├── 06_llms/                    # Pretraining, LoRA, RLHF, evaluation
├── 07_compression/             # Pruning, quantization, distillation, deployment
├── 08_research_design/         # Ablations, reproducibility, uncertainty
├── 09_trustworthy_ai/          # Robustness, fairness, privacy
├── 10_emerging/                # Diffusion, neural operators, mech interp
│
├── appendices/
│   ├── A_derivations.md        # Key equation derivations
│   ├── B_code_templates.py     # Reusable PyTorch templates
│   └── C_paper_index.md        # Annotated reference list
│
└── interview_prep/
    ├── mock_technical.md
    ├── mock_research.md
    └── mock_system_design.md
```

---

## About

This guide consolidates preparation material from research experience across academic and industrial settings — spanning safe deep reinforcement learning for constrained energy systems, AI safety and mechanistic interpretability, and applied large language model research. It is written for candidates with a computer science or statistics background who are targeting research-oriented roles at AI labs, national laboratories, and university research groups.

Longer notes and weekly write-ups are published at:

- kundan-kumarr.github.io/blog
- neuravp.substack.com

---

*2026 Edition — Kundan Kumar — All implementations in PyTorch unless noted*
