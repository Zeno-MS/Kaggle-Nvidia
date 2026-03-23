# Implementation Plan for the NVIDIA Nemotron Model Reasoning Challenge

## Executive Summary

The NVIDIA Nemotron Model Reasoning Challenge is a featured Kaggle competition that asks participants to improve the reasoning accuracy of NVIDIA’s open Nemotron models on a novel benchmark of logical transformation puzzles, starting from a Nemotron‑3 Nano baseline and evaluated with a Nemotron‑based metric. The competition is positioned as a community testbed for advancing open reasoning techniques under realistic compute constraints on Google Cloud VMs with RTX‑class Blackwell GPUs.[^1][^2][^3][^4][^5]

This report proposes a multi‑phase implementation plan built around four pillars: (1) deep benchmark analysis and synthetic data generation, (2) targeted post‑training of Nemotron‑3 Nano using NeMo tooling and open reasoning corpora, (3) aggressive test‑time computation and generative selection strategies inspired by NVIDIA’s OpenMath‑Nemotron AIMO‑2 work, and (4) a self‑improving training loop that uses surrogate Nemotron‑3 Super evaluators and lightweight RL to optimize directly for competition‑style rewards. The plan is designed to be modular, allowing incremental progress while keeping a clear path toward a highly optimized, competition‑ready reasoning engine.[^6][^7][^8][^9][^10]

## One‑Page White Paper: Nemotron‑Driven Reasoning Engine for Logical Transformation Puzzles

### Problem

The NVIDIA Nemotron Model Reasoning Challenge introduces a novel benchmark of logical reasoning puzzles that require discovering and applying latent transformation rules, evaluated by a Nemotron‑3‑Nano‑based metric that scores both reasoning traces and final answers. Participants must start from a shared Nemotron‑3 Nano baseline and develop methods that substantially increase reasoning accuracy under constrained cloud resources, pushing the frontier of open, efficient reasoning models.[^2][^3][^4][^5][^1]

### Opportunity

Nemotron 3 models combine a hybrid Mamba‑Transformer Mixture‑of‑Experts architecture, Multi‑Token Prediction (MTP), long‑context windows up to 1M tokens, and multi‑environment reinforcement learning, yielding high‑throughput, agentic reasoning capabilities that are fully open‑weights and accompanied by reproducible training recipes. NVIDIA has already demonstrated that similar architectures can achieve state‑of‑the‑art performance on mathematical reasoning competitions (OpenMath‑Nemotron for AIMO‑2) using curated reasoning datasets, chain‑of‑thought supervision, trajectory‑induced regularization, and generative solution selection. Combined with NeMo toolchains and open post‑training datasets for controllable reasoning, there is a unique opportunity to build a specialized solver that both wins the competition and produces generally useful research artifacts.[^7][^8][^9][^10][^11][^6]

### Proposed Solution

The proposed solution is a Nemotron‑centric reasoning engine tailored to the competition’s transformation puzzles, structured around three layers:

1. **Reasoning‑aware data and curriculum layer**
   - Perform structural analysis of the benchmark to identify puzzle types (e.g., string transforms, symbolic pattern rules, arithmetic sequences, grid‑like transformations) and their difficulty distribution.[^3]
   - Build a synthetic puzzle generator that programmatically instantiates transformation‑rule families and composes them, mirroring the latent rules inferred from the official dataset while avoiding label leakage.
   - Augment with open reasoning corpora such as Nemotron post‑training data and OpenMathReasoning subsets that emphasize step‑by‑step logical derivations.[^8][^6]

2. **Targeted post‑training of Nemotron‑3 Nano**
   - Fine‑tune Nemotron‑3 Nano (and lightweight adapters) on the combined real and synthetic datasets using NeMo SFT/LoRA recipes, emphasizing explicit reasoning traces and consistent answer formats aligned with the Kaggle metric’s expectations.[^9][^10][^8]
   - Use curriculum scheduling: start with simple single‑rule puzzles, then progressively introduce compositions and edge cases, leveraging Nemotron’s long context to embed multiple exemplars and meta‑instructions per prompt.[^10][^7]
   - Align style and safety with the competition’s rules and Nemotron’s open‑model license.

3. **Inference‑time compute and self‑improvement layer**
   - Implement multi‑sample chain‑of‑thought decoding and **generative selection (GenSelect)**: generate several reasoning candidates per puzzle and select the best using a verifier model or scoring head, following the successful AIMO‑2 recipe.[^6]
   - Approximate the Kaggle metric offline with a Nemotron‑3 Super‑based surrogate evaluator (or fine‑tuned Nemotron‑3 Nano) to rank candidate traces and tune prompts/hyperparameters toward the leaderboard objective.[^5][^7][^9]
   - Close the loop with light‑weight RL (e.g., GRPO/DAPO in NeMo RL) that optimizes directly for the surrogate reward, periodically regenerating high‑quality trajectories and folding them back into the SFT dataset.[^7][^8][^9]

### Key Innovations

- **Surrogate Nemotron‑3‑Super judge**: Use the open Super model and evaluation recipes as a local proxy for the Kaggle Nemotron‑based metric, enabling thousands of offline experiments without burning submission quotas.[^9][^7]
- **Transformation‑rule curriculum and synthesizer**: Treat puzzles as samples from a latent rule system and explicitly model that system via a synthetic generator, giving the model a much broader and more systematic training distribution than the raw competition CSV alone.[^3]
- **GenSelect‑style test‑time compute scaling**: Allocate more inference compute per example by generating diverse reasoning trajectories and using a Nemotron‑based selector, leveraging evidence that generative selection significantly improves reasoning benchmarks under similar constraints.[^6]
- **Self‑aligning loop via NeMo RL**: Combine SFT, preference‑style selection, and RL using NeMo Gym and RL infrastructure to gradually move the model toward behaviors that maximize the competition’s metric while remaining generalizable.[^8][^7][^9]

### Expected Impact

This system is expected to deliver substantial gains over the Kaggle baseline Nemotron‑3 Nano model on the transformation‑puzzle benchmark by combining data‑centric augmentation, tailored post‑training, and test‑time computation scaling. Beyond the leaderboard, it yields reusable components: a transformation‑rule synthetic data generator, an open Nemotron‑based reasoning verifier, and a general recipe for rapidly building domain‑specialized reasoning engines using fully open infrastructure.[^1][^2][^3][^6]

## Competition Context and Constraints

### Task, Data, and Baseline

The competition is a featured prediction challenge on Kaggle, titled "NVIDIA Nemotron Model Reasoning Challenge," and is co‑run by NVIDIA AI and partners, with approximately three months of active duration and a substantial prize pool. Participants are provided with a CSV‑based dataset (around 3 MB) licensed under CC BY 4.0, described as a collection of logical reasoning puzzles that require learning and applying underlying transformation rules, suggestive of structured, synthetic tasks rather than natural language QA. The official baseline uses a Nemotron‑3 Nano variant exposed as a Kaggle Competition Metric, with public scores around the 0.6 range early in the competition, indicating significant headroom for improvement.[^12][^4][^2][^5][^1][^3]

All training and inference in the hosted environment run on Google Cloud VMs featuring RTX‑class Blackwell GPUs (RTX PRO 6000), enabling high‑throughput inference for multi‑sample decoding but constraining on‑platform large‑scale pre‑training. Offline training on external infrastructure is allowed by Kaggle’s usual rules, with model weights uploaded as datasets; this favors a workflow where the heaviest training steps occur off‑platform, and Kaggle submissions primarily perform inference and light adaptation.[^1]

### Evaluation and Metric Behavior

The competition uses a Nemotron‑based Kaggle Competition Metric, identified as `metric/nemotron-3-nano-30b-a3b-bf16`, that interacts with models by sending a prompt and receiving an answer that includes an explicit reasoning trace followed by a final response. Public snippets indicate that the metric is sensitive both to reasoning quality and answer correctness, and that evaluation parameters configured on the Kaggle Evaluation page (e.g., temperature, maximum tokens, top‑p) can override default code values. While the metric’s internals are not fully documented, its design is consistent with Nemotron’s broader philosophy of benchmarking structured, stepwise reasoning rather than just final tokens.[^13][^11][^2][^10][^5][^9]

This setup implies that improvements can come from several directions: producing more accurate final answers, generating higher‑quality and more interpretable reasoning traces, and aligning response formats tightly with what the Nemotron metric rewards. It also strongly motivates building an offline surrogate metric that approximates Kaggle’s scoring behavior, using open Nemotron weights and evaluation recipes.

## Background: Nemotron 3 and Open Reasoning Recipes

### Nemotron 3 Family and Architecture

The Nemotron 3 family (Nano, Super, Ultra) is designed for efficient, high‑accuracy reasoning and agentic workloads, using a hybrid Mamba‑Transformer Mixture‑of‑Experts architecture with long‑context support up to 1M tokens. Nemotron 3 Super, in particular, is a 120B total (about 12B active) parameter model that combines Mamba‑2 state‑space layers, interleaved Transformer attention blocks, and LatentMoE experts to achieve strong accuracy at significantly higher throughput than similarly sized dense Transformers. The model is trained natively in NVFP4 precision and incorporates Multi‑Token Prediction to accelerate inference via built‑in speculative decoding while improving long‑range reasoning quality.[^11][^10][^7][^9]

All Nemotron 3 models undergo extensive post‑training, including supervised fine‑tuning on tens of millions of instruction, reasoning, coding, and safety samples, followed by multi‑environment reinforcement learning in NeMo Gym that focuses on multi‑step tool use, planning, and verifiable tasks. This makes Nemotron 3 particularly well‑suited as both a base model and an evaluator for structured reasoning competitions.[^10][^7][^9]

### OpenMath‑Nemotron and AIMO‑2 Lessons

NVIDIA’s OpenMath‑Nemotron work, released as part of its winning submission to the AI Mathematical Olympiad Progress Prize 2 (AIMO‑2), provides a detailed recipe for building state‑of‑the‑art open mathematical reasoning models and competition strategies. The approach relies on three pillars: (1) constructing a curated high‑quality math dataset of hundreds of thousands of problems with chain‑of‑thought solutions, (2) leveraging test‑time computation with multi‑sample chain‑of‑thought generation and trajectory‑induced regularization (TIR), and (3) using generative solution selection (GenSelect) where a selection model chooses the best candidate among multiple generated solutions.[^6]

The AIMO‑2 pipeline iteratively improves models by training increasingly powerful TIR‑capable models that generate and filter additional training examples, creating a self‑reinforcing loop of better data and better models. Many of these ideas—especially GenSelect and iterative self‑improvement—are directly applicable to the Nemotron Model Reasoning Challenge, where the task is also to maximize structured reasoning quality under time and compute limits.[^6]

### NeMo Reasoning Model Tooling

NVIDIA’s NeMo ecosystem includes open datasets, curation code, and training scripts for building reasoning‑capable LLMs in short time frames on a single high‑end GPU, based on the Llama Nemotron Post‑Training dataset and associated recipes. The tutorials emphasize test‑time computation scaling (spending more tokens on internal reasoning before answering), curating reasoning‑oriented examples, and leveraging NeMo Curator and Evaluator to build reproducible pipelines.[^8][^9]

For Nemotron 3 Super, NVIDIA provides open weights, pretraining and post‑training recipes, RL environments via NeMo Gym, and deployment cookbooks for vLLM, SGLang, and TensorRT LLM, making it feasible to run Super as a local surrogate judge and to fine‑tune or extend it for domain‑specific evaluation tasks. These resources drastically lower the barrier to building custom evaluators and domain‑specific Nemotron variants targeted at the competition’s puzzles.[^7][^9]

## Routes to Explore for Competitive Advantage

This section outlines concrete research routes, each of which can be explored independently and then composed into a unified solution.

### 1. Benchmark Decomposition and Puzzle Taxonomy

- **Reverse‑engineering puzzle families**: Analyze the official CSV and create a taxonomy of puzzle types (e.g., string rewrites, arithmetic progressions, symbolic transformations, grid‑like rules), frequencies, and co‑occurrences.[^3]
- **Error archetypes**: Run the baseline Nemotron‑3 Nano metric on train/validation splits to identify common failure modes (e.g., partial rule discovery, off‑by‑one errors, misapplied composition), informing targeted augmentations.
- **Rule‑system hypothesis testing**: Formulate explicit hypotheses about underlying rule systems (e.g., finite sets of primitive transforms and composition operators) and test them by fitting small programmatic solvers to subsets of puzzles.

### 2. Synthetic Transformation‑Rule Generator

- **Programmatic puzzle engine**: Implement an engine that can generate puzzles by sampling from a library of primitive transformations (string operations, arithmetic transforms, permutations) and composing them to create multi‑step rules, closely mirroring inferred competition patterns.[^3]
- **Difficulty calibration**: Use metrics such as minimal program length, number of composition steps, and ambiguity to stratify synthetic puzzles into difficulty levels that approximate the competition’s distribution.
- **Domain randomization**: Randomize surface forms (tokens, variable names, problem narratives) while keeping underlying rules fixed, forcing the model to learn structural reasoning rather than memorize templates.

### 3. Nemotron‑3 Nano Post‑Training Specialization

- **LoRA‑based SFT**: Apply LoRA adapters on the official Nemotron‑3 Nano checkpoints using NeMo’s SFT recipes, training on a mixture of competition puzzles, synthetic puzzles, and selected reasoning tasks from open datasets like OpenMathReasoning.[^9][^8][^6]
- **Reasoning‑format alignment**: Standardize prompts and outputs so that the model always produces a clearly delimited reasoning trace followed by a final answer in the format expected by the Kaggle metric, improving score stability.[^5]
- **Curriculum learning**: Schedule training so that the model first masters single‑rule puzzles, then progressively more complex compositions and rare archetypes where the baseline struggles.

### 4. Test‑Time Computation and Generative Selection

- **Multi‑sample chain‑of‑thought (CoT)**: For each puzzle, generate multiple reasoning trajectories at modest temperature, using Nemotron‑3 Nano in a cost‑aware manner consistent with the competition’s inference budget.
- **GenSelect‑style selection**: Train or fine‑tune a verifier model—potentially a smaller Nemotron or a Super‑based classifier—to score each reasoning trajectory and select the candidate that maximizes predicted correctness, directly inspired by OpenMath‑Nemotron’s GenSelect.[^7][^6]
- **Adaptive compute allocation**: Use simple heuristics or a learned controller to allocate more samples to harder puzzles (e.g., those where candidates disagree or where early confidence is low) and fewer to easier ones, maximizing leaderboard gain per FLOP.

### 5. Surrogate Nemotron‑3 Super Judge

- **Super‑based evaluator**: Fine‑tune Nemotron‑3 Super (or a distilled variant) as an evaluator that, given a puzzle, reasoning trace, and answer, outputs a scalar reward approximating the Kaggle Nemotron metric’s score, trained on offline logs and/or manually labeled subsets.[^5][^9][^7]
- **Prompt‑space and hyperparameter search**: Use the surrogate judge to explore prompt templates, decoding parameters, and selection strategies at scale, only promoting the best‑performing configurations to actual Kaggle submissions.
- **Ensemble of evaluators**: Optionally combine multiple evaluators (e.g., Super, Nano, and simple rule‑based checks) to reduce overfitting to any single surrogate and to better capture both formal correctness and stylistic preferences.

### 6. RL and Preference Optimization Loop

- **Preference datasets**: Construct pairwise preference data where, for a given puzzle, one reasoning trajectory is judged better than another according to the surrogate evaluator or human annotators, mirroring RLHF setups.[^8][^7][^6]
- **GRPO/DAPO fine‑tuning**: Use NeMo RL to apply GRPO or DAPO to Nemotron‑3 Nano, optimizing for the surrogate reward while keeping models compact enough for competition deployment.[^9][^7]
- **Iterative self‑improvement**: Periodically regenerate trajectories using the improved model, re‑score them with the surrogate judge, and expand the preference dataset, recursively tightening the alignment between the model and the competition metric.

### 7. Infrastructure, Experimentation, and Safety

- **Offline evaluation harness**: Build a local harness that mirrors Kaggle’s prompt/response contract and evaluation interface, plugging in surrogate judges and sanity checks before each leaderboard submission.[^13][^5]
- **Reproducible pipelines**: Use NeMo Curator, Evaluator, and open Nemotron recipes to ensure that all steps—data curation, SFT, RL, and evaluation—are scripted and repeatable.[^7][^8][^9]
- **Safety and content constraints**: Ensure that synthetic data and generated reasoning traces respect Kaggle’s content rules and Nemotron model licenses, avoiding disallowed content even in intermediate traces.[^10][^9]

## Structural Schema of the Implementation Plan

This section presents a structured schema for implementing the proposed approach, organized by phases, components, and feedback loops.

### Phase 0: Environment and Baseline Reproduction

**Objectives**: Replicate the official baseline behavior locally and set up the technical environment.

- **Component 0.1 – Kaggle environment mirror**  
  - Containerize a local environment matching the Kaggle runtime (Python version, libraries, CUDA, transformers stack) and integrate the official Nemotron‑3 Nano baseline where possible.[^2][^5]
  - Implement a thin wrapper that exposes a `solve(puzzle)` API identical to the competition’s expected entry point.

- **Component 0.2 – Offline metric stub**  
  - Reconstruct the prompt/response contract used by the Kaggle Nemotron metric (reasoning trace + final answer) and implement a pluggable scoring interface.
  - Initially, use simple accuracy/regex checks on known puzzles, then later plug in Nemotron‑based surrogate evaluators.

### Phase 1: Benchmark Analysis and Synthetic Data Engine

**Objectives**: Understand the puzzle distribution and build a generator that expands it.

- **Component 1.1 – Dataset profiler**  
  - Parse the competition CSV into a structured schema: puzzle ID, input representation, target output, metadata (if any).[^3]
  - Compute statistics: token lengths, pattern families (via clustering/regex), and baseline error slices.

- **Component 1.2 – Puzzle taxonomy and rule hypotheses**  
  - Manually and programmatically cluster puzzles into families; document hypothesized underlying rules and typical transformations.
  - Validate hypotheses by implementing small symbolic solvers for representative subsets.

- **Component 1.3 – Synthetic puzzle generator**  
  - Design a domain‑specific language (DSL) for primitive transformations and their composition.
  - Implement a generator that samples random DSL programs and produces input/output pairs, with difficulty labels derived from program complexity.

- **Component 1.4 – Data curation pipeline**  
  - Combine original competition data, synthetic puzzles, and selected external reasoning datasets (e.g., OpenMathReasoning subsets) into a unified training corpus, with tags indicating source and difficulty.[^8][^6]

### Phase 2: Nemotron‑3 Nano SFT and Style Alignment

**Objectives**: Train specialized Nemotron‑3 Nano variants that excel at producing clear reasoning traces and correct final answers for transformation puzzles.

- **Component 2.1 – Prompt and output templates**  
  - Standardize prompts to include a short description of the task, a few in‑context examples, and explicit instructions to show reasoning and then output the final answer in a specified format aligned with the Kaggle metric.[^5]

- **Component 2.2 – LoRA adapter training**  
  - Use NeMo SFT recipes to train LoRA adapters on Nemotron‑3 Nano, feeding the curated corpus and emphasizing faithful reproduction of step‑by‑step derivations.[^10][^9][^8]
  - Experiment with different adapter ranks and parameter‑efficient fine‑tuning schemes to stay within deployment constraints.

- **Component 2.3 – Curriculum scheduler**  
  - Implement a curriculum that starts SFT on simpler synthetic puzzles, then gradually mixes in real competition puzzles and harder synthetic compositions, monitoring held‑out validation performance.

### Phase 3: Test‑Time Compute Scaling and Generative Selection

**Objectives**: Exploit inference‑time compute to boost accuracy via multi‑sample generation and selection.

- **Component 3.1 – Multi‑sample CoT decoding**  
  - Extend the `solve(puzzle)` API to sample multiple reasoning traces per puzzle under a budget (e.g., K samples or token budget), using temperature and top‑p tuned on offline validation data.

- **Component 3.2 – Verifier and selector models**  
  - Train a selector model (e.g., smaller Nemotron or a dedicated head on Nemotron‑3 Nano) on labeled or pseudo‑labeled data to predict which candidate reasoning trace is most likely correct, following GenSelect principles.[^6][^7]
  - Integrate the selector into inference: generate K candidates, score them, and emit the best‑scoring trajectory and its final answer.

- **Component 3.3 – Adaptive allocation policy**  
  - Implement heuristics (or a small learned policy) that decide K per puzzle based on early candidate diversity or confidence scores, ensuring the solution remains within Kaggle’s runtime limits.

### Phase 4: Surrogate Nemotron‑3 Super Judge and RL Alignment

**Objectives**: Approximate the Kaggle metric offline and use it to drive RL‑style improvement.

- **Component 4.1 – Super‑based surrogate reward model**  
  - Fine‑tune Nemotron‑3 Super (using provided recipes) as an evaluator that takes a puzzle, reasoning trace, and answer, and outputs a scalar reward approximating leaderboard scores, trained on offline logs and synthetic preference data.[^9][^7]

- **Component 4.2 – Preference dataset construction**  
  - For each puzzle, generate multiple trajectories with variant models/prompts; use the surrogate evaluator (and simple correctness checks) to label pairwise preferences.

- **Component 4.3 – GRPO/DAPO fine‑tuning loop**  
  - Use NeMo RL to apply GRPO or DAPO to Nemotron‑3 Nano (with adapters), optimizing for surrogate rewards while constraining divergence from the SFT baseline to avoid overfitting.[^7][^8][^9]
  - Periodically refresh the preference dataset with trajectories from the updated model, creating a recursive improvement loop.

### Phase 5: Integration, Evaluation, and Submission Strategy

**Objectives**: Integrate components into a robust Kaggle submission and manage experimentation.

- **Component 5.1 – Unified solver pipeline**  
  - Compose all components into a single submission script: load Nemotron‑3 Nano with LoRA adapters, apply standardized prompts, run multi‑sample decoding, apply the selector, and emit formatted reasoning + answers.

- **Component 5.2 – Offline leaderboard emulator**  
  - Integrate the surrogate judge (and any auxiliary checks) into an evaluation harness that can estimate leaderboard performance on held‑out splits before making real submissions.[^5][^9][^7]

- **Component 5.3 – Submission scheduling and ablation**  
  - Plan a submission cadence that uses the limited daily submissions to test major architectural changes and hyperparameter regimes, while smaller variations are resolved offline.

- **Component 5.4 – Monitoring and diagnostics**  
  - Track per‑family performance, error types, and calibration of surrogate scores versus actual leaderboard scores, updating the surrogate and selection mechanisms as needed.

## Forward‑Looking Extensions

Beyond winning the competition, this implementation plan lays groundwork for broader research in transformation‑rule reasoning and agentic AI.

- **Generalized rule‑reasoning benchmarks**: Extend the synthetic generator and evaluation stack to new domains (e.g., program synthesis, data transformation, automated grading) and release them as open benchmarks.
- **Modular agent integration**: Embed the specialized Nemotron‑based solver as a module inside larger agent systems (e.g., OpenClaw/OpenHands) that already use Nemotron 3 Super for planning, leveraging its 1M‑token context for long‑horizon tasks.[^9][^7]
- **Cross‑model distillation**: Distill insights from Super‑based evaluators and Ultra‑scale Nemotron models into smaller, edge‑deployable reasoning models using the same SFT+GenSelect+RL recipe.[^11][^10][^7]

This combination of data‑centric design, open‑model post‑training, and test‑time computation scaling offers a principled, extensible path not only to strong competition performance but also to a reproducible, research‑grade reasoning system built entirely on open Nemotron infrastructure.

---

## References

1. [Kaggle - NVIDIA Nemotron Model Reasoning Challenge - LinkedIn](https://www.linkedin.com/posts/kaggle_nvidia-nemotron-model-reasoning-challenge-activity-7440068632056983554-V6NA) - ... Nemotron-3 Nano baseline and a novel reasoning benchmark from NVIDIA Research. The goal is to de...

2. [NVIDIA Nemotron Model Reasoning Challenge - Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models) - NVIDIA Nemotron Model Reasoning Challenge. Advance reasoning techniques using NVIDIA Nemotron open m...

3. [NVIDIA Nemotron Model Reasoning Challenge - Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) - This dataset comprises a collection of logical reasoning puzzles requiring the identification and ap...

4. [NVIDIA Nemotron 3 Reasoning Challenge | Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) - In this competition, participants will work from a shared Nemotron 3 Nano baseline and a novel reaso...

5. [Kaggle Competition Metrics | Nemotron-3-Nano-30B-A3B-BF16](https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default) - It responds to user queries and tasks by first generating a reasoning trace and then concluding with...

6. [[PDF] arXiv:2504.16891v1 [cs.AI] 23 Apr 2025](https://arxiv.org/pdf/2504.16891.pdf) - OpenMath-Nemotron models. In this section we present the training and evaluation details of our Open...

7. [[PDF] Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba ...](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) - The report is organized into 3 broad sections: Pre-training (§2), Post-training (§3), and Quantiza- ...

8. [Train a Reasoning-Capable LLM in One Weekend with NVIDIA NeMo](https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/) - We leverage the Llama Nemotron Post-Training dataset, enabling your model to learn controllable reas...

9. [Introducing Nemotron 3 Super: An Open Hybrid Mamba-Transformer ...](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/) - For the full technical details, read the Nemotron 3 Super technical report. Stay up-to-date on NVIDI...

10. [[PDF] NVIDIA Nemotron 3: Efficient and Open Intelligence - arXiv](https://arxiv.org/pdf/2512.20856.pdf) - The Nemotron 3 family uses a Mixture-of-Experts hybrid Mamba–Transformer architecture to provide bes...

11. [NVIDIA Nemotron 3: Efficient and Open Intelligence - arXiv](https://arxiv.org/abs/2512.20856) - We introduce the Nemotron 3 family of models - Nano, Super, and Ultra. These models deliver strong a...

12. [NVIDIA Nemotron Model Reasoning Challenge | Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion) - NVIDIA Nemotron Model Reasoning Challenge. Advance reasoning techniques using NVIDIA Nemotron open m...

13. [NVIDIA Nemotron Model Reasoning Challenge - Kaggle](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/681745) - The parameters on the Evaluation page are what the metric is currently running with. They override t...

