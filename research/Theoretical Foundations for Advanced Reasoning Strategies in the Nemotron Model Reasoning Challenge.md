# Theoretical Foundations for Advanced Reasoning Strategies in the Nemotron Model Reasoning Challenge

## 1. Introduction and Scope

The NVIDIA Nemotron Model Reasoning Challenge centers on learning and applying latent transformation rules from examples, a problem class that aligns closely with the Abstraction and Reasoning Corpus (ARC-AGI) and broader work on structured reasoning in large language models (LLMs). This paper surveys and synthesizes the main theoretical frameworks underlying the research directions previously identified as high‑leverage for this competition, without focusing on concrete implementation details.[^1][^2][^3]

The goal is to articulate the *theories* and core conceptual tools behind twelve lines of work: ARC‑style abstraction and induction, Test‑Time Training, Process Reward Models, Monte Carlo Tree Search for reasoning, neurosymbolic and program synthesis approaches, inference scaling laws, advanced self‑consistency aggregation, Minimum Description Length, dual‑process cognition, RL with verifiable rewards, and contrastive/preference‑based training.[^4][^5][^6][^7][^8][^2][^9][^10][^11][^12][^13][^14][^15][^16][^1]

***

## 2. ARC‑Style Abstraction and the Induction–Transduction Complementarity

### 2.1 ARC-AGI as a Theory of Intelligence Test

The ARC-AGI benchmark, introduced by François Chollet, is designed explicitly to measure abstraction and reasoning rather than pattern matching, using grid‑based tasks where the underlying rule is not stated and must be inferred from input–output examples. The 2024 and 2025 ARC Prize competitions built on this, treating each instance as a mini scientific‑discovery problem: infer hidden regularities, represent them as compositional rules, and apply them to new inputs.[^17][^18][^19][^1]

This framing leads to a working theory: solving ARC‑like tasks requires **inductive program synthesis** over a latent space of symbolic transformations, supported by **transductive pattern completion** to interpolate among examples.[^2][^1]

### 2.2 Induction vs. Transduction

The 2024 ARC Prize winning solution explicitly decomposed solving into two complementary parts: *inductive* reasoning that discovers explicit symbolic rules consistent with all training examples, and *transductive* reasoning that directly maps inputs to outputs using powerful but opaque neural models. Empirical analysis showed that the best performance arises when these two modes are combined, leading to the statement that “induction and transduction are strongly complementary” for abstraction tasks.[^2]

From a theoretical perspective, induction operates over a **hypothesis space of programs** (or rules) and searches for the simplest hypothesis that explains the observations, while transduction performs **function approximation** from examples to outputs without explicit modeling of the underlying rule. The ARC results suggest that the hypothesis class accessible to current program synthesizers is too narrow to solve all tasks alone, while pure transduction struggles with extreme out‑of‑distribution generalization and systematic rule reuse.[^1][^2]

### 2.3 Object-Centric and Relational Representations

Many ARC approaches leverage **object‑centric representations**, explicitly segmenting the grid into entities and relations before induction. This connects to a general theory that reasoning over **structured factorized representations** (objects and relations) is more robust and sample‑efficient than reasoning over raw pixels or token sequences. It also mirrors work in neurosymbolic program synthesis where symbolic logic is built over extracted relational predicates.[^10][^20][^21][^1][^2]

***

## 3. Test‑Time Training (TTT) as On‑the‑Fly Specialization

### 3.1 Conceptual Definition

Test‑Time Training (TTT) is a paradigm where model parameters are adapted at inference time using unsupervised or self‑supervised losses derived from the input distribution of the *current* task instance. Rather than relying solely on pretraining and offline fine‑tuning, TTT treats each test example (or batch of related examples) as an opportunity for “micro‑learning,” nudging the model toward representations specific to that instance.[^22][^23][^24][^25]

### 3.2 Theoretical Motivation

From a representation‑learning perspective, TTT can be viewed as minimizing a local loss that enforces **consistency constraints** or **inductive biases** tailored to the test distribution, subject to a prior induced by the pretrained model. For abstract reasoning tasks, MIT’s work shows that in‑context learning (ICL) alone only modestly improves performance, while combining ICL with TTT on the same support examples leads to much larger gains, indicating that *updating parameters* is more powerful than conditioning alone.[^24][^25][^22]

Formally, if the base model implements a mapping parameterized by θ, TTT performs a few gradient steps θ′ = θ − η∇L_TTT(θ; x_test), where L_TTT is a self‑supervised loss defined on x_test (e.g., prediction consistency, reconstruction, or pseudo‑label agreement), then uses θ′ for prediction. Under certain smoothness assumptions, this can be seen as local Bayesian updating around θ, effectively specializing the model to a narrow region of the input space.[^23][^22]

### 3.3 Relevance to Latent Rule Tasks

For latent transformation puzzles, each puzzle (or family of related puzzles) defines a small local manifold within the broader distribution. TTT enables **per‑puzzle specialization**: the model moves in parameter space toward one that better encodes that puzzle’s rule system, without permanently altering global parameters. This is tightly aligned with ARC‑style refinement loops, where each task triggers its own small optimization or search process.[^25][^23][^24][^1][^2]

***

## 4. Process Reward Models (PRMs) and Fine‑Grained Supervision

### 4.1 From Outcome Rewards to Process Rewards

Conventional reward models for LLMs assign scalar rewards based on **final outputs** (e.g., the answer or full completion), which conflates many possible error sources—misinterpretation, local arithmetic mistakes, or formatting errors—into a single signal. **Process Reward Models (PRMs)** instead assign rewards at intermediate steps of a chain‑of‑thought, labeling each reasoning step as correct, incorrect, or partially aligned with a gold solution.[^9][^11]

This aligns with a long‑standing idea in reinforcement learning that **dense rewards** accelerate learning by providing more frequent feedback. PRMs can be seen as supplying a reward function over the trajectory of thoughts rather than just terminal states.[^11][^9]

### 4.2 Distributional Mismatch and Reflective Reasoning

Recent work on PRMs for reflective mathematical reasoning argues that standard PRMs trained on short, simple chains do not generalize to long, complex reasoning sequences because of a distributional gap. The authors propose constructing training data from **extended, self‑reflective chains of thought**, where the model critiques or revises its own reasoning, and training PRMs specifically on these trajectories.[^9]

The theoretical implication is that the reward function must be **support‑matched** to the policy’s behavior distribution; otherwise, importance weighting or off‑policy correction is required. For LLM reasoning, this translates to: PRMs need to be trained on the same style and length of reasoning traces that the model will produce after alignment.[^11][^9]

### 4.3 PRMs as Surrogate Value Functions

PRMs can also be interpreted as **value functions** over partial trajectories: given a prefix of a reasoning chain, predict the expected correctness of the final answer if the trajectory continues in the same style. This perspective is crucial for integrating PRMs with search algorithms like MCTS, where a value estimate for a partial state determines exploration priorities.[^9][^11]

***

## 5. Monte Carlo Tree Search (MCTS) for Reasoning Trajectories

### 5.1 MCTS Basics

MCTS is a simulation‑based planning algorithm widely used in game AI (e.g., AlphaGo) that builds a search tree by iteratively sampling trajectories, updating estimates of action values, and balancing exploration and exploitation. Each iteration consists of selection, expansion, simulation (rollout), and backpropagation steps guided by an upper confidence bound (UCB) or similar rule.[^26][^27]

In the context of LLM reasoning, **nodes** represent partial reasoning states (or prefixes of chain‑of‑thought), **edges** represent the generation of new tokens or logical steps, and **rewards** come from PRMs or verifiable correctness checks.[^28][^29][^26]

### 5.2 Preference‑Based and Constrained MCTS

Recent work such as ReKG‑MCTS and SRA‑MCTS adapts MCTS to contexts where rewards come from **preference comparisons** or where the action space must be constrained to maintain validity. CMCTS explicitly combines MCTS with PRMs and structural constraints, demonstrating that guiding search with step‑level rewards and domain‑specific constraints yields substantial gains on math and graph reasoning tasks even at modest model sizes.[^30][^29][^31][^27][^28]

Theoretically, this can be understood as approximating optimal planning in a large Markov decision process (MDP) where states are text prefixes, actions are token emissions, and the reward is verifiable correctness or PRM score. MCTS provides an approximate solution that concentrates exploration in regions of the trajectory space with high expected reward.[^29][^27][^26]

### 5.3 Trajectory‑Induced Regularization and Self‑Improvement

MCTS‑based approaches often incorporate **trajectory‑induced regularization**: trajectories discovered via search are fed back into the training process as high‑quality examples, gradually sculpting the policy’s behavior. This yields a self‑improving loop analogous to AlphaZero: policy generates data with search, data improves policy, improved policy powers better search.[^31][^26]

***

## 6. Neurosymbolic Reasoning and Program Synthesis

### 6.1 Neurosymbolic Architectures

Neurosymbolic systems combine differentiable neural networks with discrete symbolic structures and algorithms. The theoretical motivation is that neural networks excel at perception and approximate inference in high‑dimensional spaces, while symbolic methods excel at compositionality, systematic generalization, and formal verification.[^20][^10]

A 2025 study on improving rule‑based reasoning in LLMs via neurosymbolic methods encodes internal representations into a **Vector Symbolic Architecture (VSA)**, operates over this space with symbolic algorithms, and then decodes back to natural language answers. The key theoretical claim is that VSA enables binding and superposition operations analogous to symbolic compositionality while remaining differentiable.[^32][^10]

### 6.2 Inductive Logic Programming (ILP)

Inductive Logic Programming seeks to infer logical rules (often in Prolog form) from positive and negative examples, under background knowledge. ILP for ARC‑style tasks operates over object‑centric encodings of grids, learning rules that transform one configuration into another.[^21][^20]

From a learning theory standpoint, ILP searches a **hypothesis space of logic programs** constrained by mode declarations and bias, aiming to find programs that entail all positive examples and none of the negatives. This is a form of **structural risk minimization** in a symbolic space, where model complexity is controlled by program size and allowed predicates.[^20][^21]

### 6.3 Program Synthesis Benchmarks (e.g., IPARC)

The IPARC benchmark extends ARC concepts into a structured program synthesis setting, where tasks require synthesizing small programs from examples. LLM‑guided approaches decompose tasks into subprograms and assemble them via data‑flow diagrams, reflecting a compositional view of problem solving: complex behaviors emerge from the composition of simpler learned modules.[^33]

The theoretical thread across neurosymbolic and ILP work is that explicit program representations serve as **interpretable, verifiable hypotheses** about the environment’s rules, facilitating both strong generalization and robust verification.[^10][^33][^21][^20]

***

## 7. Inference Scaling Laws and Compute‑Optimal Reasoning

### 7.1 Empirical Scaling Laws at Inference Time

Scaling laws traditionally describe how performance improves with model size and training compute. Recent work instead focuses on **inference‑time scaling laws**, examining how accuracy depends on model size, number of samples, and verification cost for a fixed inference budget.[^12][^13]

An ICLR 2025 study shows empirically that for complex tasks, **smaller models with more sampled trajectories** can surpass larger models with fewer samples when total FLOPs are held constant. They derive an approximate compute‑optimal relationship between model size and inference budget, formalizing the tradeoff between “thinking depth” (number of samples and verification steps) and raw capacity.[^13][^12]

### 7.2 Best‑of‑N and Beyond

Under a simple best‑of‑N scheme, the probability of success increases as one minus the probability that all N samples fail, assuming independent errors. However, independence rarely holds; thus advanced aggregation (PRM‑guided selection, ranked voting, MCTS) exploits dependency structure to extract more from each sample.[^34][^14][^15][^12][^13]

The theoretical implication is that reasoning systems should be designed **jointly** over model architecture and inference strategy, rather than treating decoding as a fixed post‑processing step. In particular, sampling and verification become central algorithmic components, not mere heuristics.

***

## 8. Self‑Consistency, Ranked Voting, and Aggregation Theory

### 8.1 Self‑Consistency as Stochastic Search

Self‑consistency sampling runs the model multiple times with varied decoding randomness, then aggregates answers via majority vote. This can be seen as a **stochastic search** in reasoning‑trajectory space, where each trajectory corresponds to a sample from the policy, and the voting mechanism approximates a mode of the underlying answer distribution.[^34]

### 8.2 Ranked Voting Self‑Consistency

Recent work shows that standard majority voting discards useful information, because each sample typically prefers not just a single answer but an *ordering* over plausible alternatives. Ranked Voting Self‑Consistency generalizes voting to exploit these preferences, using social‑choice‑theoretic methods (e.g., Borda count, Condorcet methods) to aggregate rankings into a final answer.[^14][^15]

Theoretically, this can be understood as moving from estimating the marginal distribution over answers to estimating a **preference order**, which is richer and more sample‑efficient under certain noise models. It also connects LLM aggregation to well‑studied results in voting theory about robustness and manipulability.[^15][^14]

### 8.3 Confidence‑Weighted Aggregation

Another line of work introduces confidence‑weighted self‑consistency, where each sampled trajectory includes an internal confidence estimate. Under a Bayesian interpretation, these confidences approximate posterior probabilities over answers, and weighted voting approximates **Bayes‑optimal decision rules** under 0–1 loss.[^16]

***

## 9. Minimum Description Length (MDL) and Kolmogorov Complexity

### 9.1 MDL Principle

The **Minimum Description Length (MDL)** principle states that the best explanation for observed data is the one that minimizes the sum of:

1. The description length of the hypothesis (model), and  
2. The description length of the data given the hypothesis.[^5][^35][^36][^8]

This provides a formalization of Occam’s Razor: simpler models that compress the data better are preferred. MDL’s central quantity, **stochastic complexity**, generalizes maximum likelihood by incorporating model complexity penalties derived from information theory.[^36][^8][^5]

### 9.2 Relation to Kolmogorov Complexity

Kolmogorov complexity defines the information content of a string as the length of the shortest program that generates it on a universal Turing machine, but is uncomputable in general. MDL provides a computable surrogate by restricting the hypothesis class to a parametric family and using coding arguments to approximate description lengths.[^8][^5][^36]

### 9.3 Application to Rule Learning

In transformation puzzles, each candidate rule (or program) can be assigned an MDL score based on its complexity (program length, number of primitives) and its ability to reproduce the examples. The rule with minimal description length is theoretically the most plausible. MDL therefore underpins a principled prior for ranking symbolic hypotheses and can be integrated with ARC‑style induction or program synthesis.[^35][^5][^36]

***

## 10. Dual‑Process Cognition: System 1 / System 2

### 10.1 Kahneman’s Model

Kahneman’s dual‑process theory posits two interacting systems: **System 1**, which operates automatically and quickly with little effort, and **System 2**, which is slower, deliberative, and effortful. System 1 handles routine judgments and pattern recognition; System 2 is recruited when tasks are novel, complex, or when System 1’s outputs are flagged as potentially erroneous.[^37][^6][^38][^39][^40]

This model provides a psychological explanation for why hybrid architectures—fast neural transduction plus slow symbolic induction—are effective: they mimic the division of labor between intuitive and analytical cognition.[^40][^19][^2]

### 10.2 Implications for LLM Architectures

Dual‑process theory suggests designing AI systems with:

- A **fast, low‑cost pathway** that leverages learned heuristics and pattern matching for easy or familiar instances (System 1 analog).[^3][^24][^25]
- A **slow, high‑cost pathway** that employs explicit search, TTT, program synthesis, or MCTS when the fast pathway’s confidence is low or when internal monitors detect conflict (System 2 analog).[^40][^3][^2]

Recent analyses of design thinking also map creative workflows onto System 1 generating candidate ideas and System 2 evaluating and refining them, reinforcing the view that effective problem solvers interleave fast generative and slow evaluative processes.[^40]

***

## 11. Reinforcement Learning with Verifiable Rewards (RLVR)

### 11.1 RLVR Paradigm

Reinforcement Learning with Verifiable Rewards (RLVR) is a post‑training paradigm where rewards are provided by **deterministic verifiers** or rule‑based checks rather than human preference models. In RLVR, an LLM acts as a policy generating an output (often a chain of thought and answer), which is then passed through a verifier that returns a reward only if certain objective criteria are met (e.g., exact correctness, passing test cases, satisfying formal constraints).[^41][^42][^7][^43][^4]

The theoretical appeal is twofold:

1. Rewards are **noise‑free and unambiguous** in domains where verification is straightforward (math, code, structured puzzles).  
2. RLVR avoids many failure modes of RLHF, such as reward hacking or annotator bias, because the reward is defined by formal rules.[^42][^7][^43][^4]

### 11.2 RLVR and Reasoning Quality

A 2025 study on RLVR shows that training with verifiable rewards not only improves pass@K metrics but also leads to qualitative improvements in chain‑of‑thought reasoning, as measured by a new CoT‑Pass@K metric that accounts for both final answers and intermediate reasoning steps. The authors provide a theoretical analysis of the **incentive structure** of RLVR, arguing that even when rewards depend only on final correctness, exploration in policy space tends to favor trajectories that embody correct reasoning, because these trajectories occupy a larger volume of successful policy configurations.[^7][^43]

### 11.3 Relation to Verifiable Environments

Transformation puzzles with executable rule programs are quintessential **verifiable environments**: given a candidate rule, one can run it on inputs and check outputs exactly. This makes them ideal for RLVR, where the DSL executor serves as the environment and correctness forms the reward signal.[^4][^42][^7]

***

## 12. Contrastive, Preference‑Based, and Negative Training

### 12.1 Contrastive Learning for Reasoning

Contrastive learning constructs pairs (or sets) of examples where some are labeled as more desirable than others, and the model is trained to assign higher scores or probabilities to preferred items. For reasoning, this often means providing both a correct reasoning trace and a subtly incorrect one for the same problem and training a discriminator or PRM to favor the correct trace.[^14][^15][^16][^11][^9]

Theoretically, contrastive methods maximize a lower bound on mutual information between inputs and desired outputs or behaviors, and can be framed as **noise‑contrastive estimation** in trajectory space.[^11]

### 12.2 Direct Preference Optimization (DPO) and Variants

DPO and related methods bypass explicit reward modeling by directly optimizing the log‑odds that the policy prefers chosen outputs over rejected ones, derived from fixed preference datasets. For step‑level reasoning, DPO‑style objectives can be defined over sequences of micro‑decisions, aligning the model with human or automated judgments about which intermediate steps constitute good reasoning.[^14][^11]

### 12.3 Negative Examples and Error Taxonomies

Training on curated **negative examples**—incorrect solutions with annotated error types (off‑by‑one, misapplied rule, incomplete reasoning)—helps the model internalize a taxonomy of failure modes. This connects with cognitive theories of learning from **counterexamples**, where exposure to specific near‑miss cases sharpens category boundaries and rule representations.[^9][^11]

***

## 13. Synthesis: A Unified Theoretical Picture

The research directions surveyed here can be viewed as different lenses on the same underlying problem: **discovering, representing, and exploiting latent rule systems from sparse examples under compute constraints**.

- ARC‑style induction/transduction and neurosymbolic program synthesis prioritize *explicit rule representations* and MDL‑like simplicity priors.[^5][^36][^8][^33][^21][^1][^2][^10][^20]
- TTT, PRMs, and contrastive training focus on *how the model learns from each instance*, providing dense, structure‑aware feedback during both training and inference.[^22][^23][^24][^25][^11][^9]
- MCTS, self‑consistency, and inference scaling laws treat reasoning as *search over trajectories*, where sampling, aggregation, and value estimation are first‑class algorithmic components.[^12][^13][^26][^29][^15][^16][^34][^14]
- RLVR leverages the special status of verifiable domains to run RL with near‑perfect reward signals, closing the loop between model proposals and formal correctness.[^43][^42][^7][^4]
- Dual‑process cognition provides a high‑level organizing metaphor: System 1 corresponds to fast, heuristic, neural transduction; System 2 corresponds to slow, deliberate, symbolic or search‑based reasoning, with meta‑control deciding when to escalate from one to the other.[^6][^39][^37][^2][^40]

For the Nemotron Model Reasoning Challenge, these theories jointly motivate a design space where a Nemotron base model is augmented by per‑task specialization (TTT), symbolic hypothesis spaces (program synthesis + MDL priors), structured search (MCTS + PRMs + advanced aggregation), and verifiable feedback loops (RLVR), all orchestrated within a dual‑process control framework.

---

## References

1. [ARC Prize 2025: Technical Report - arXiv](https://arxiv.org/html/2601.10904v1) - The defining theme of 2025 is the emergence of the refinement loop – a per-task iterative program op...

2. [Multimodal Reasoning to Solve the ARC-AGI Challenge](https://omseeth.github.io/blog/2025/MLLM_for_ARC/) - By comparison, the winning approach [18] of the ARC Prize from 2024 achieved 71.6% accuracy on the p...

3. [The State Of LLMs 2025: Progress, Problems, and Predictions](https://magazine.sebastianraschka.com/p/state-of-llms-2025) - ​All that being said, the takeaway is that LLM development this year was essentially dominated by re...

4. [Reinforcement Learning with Verified Reward (RLVR)](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-reward-rlvr) - RLVR uses rule-based rewards to post-train LLMs, significantly improving accuracy and robustness for...

5. [[PDF] Model Selection Based on Minimum Description Length - IRI](https://iri.columbia.edu/~tippett/cv_papers/Grunwald.pdf) - We introduce the minimum description length (MDL) principle, a general principle for inductive infer...

6. [System 1 and System 2 Thinking - The Decision Lab](https://thedecisionlab.com/reference-guide/philosophy/system-1-and-system-2-thinking) - System 1 and System 2 thinking describes two distinct modes of cognitive processing introduced by Da...

7. [Reinforcement Learning with Verifiable Rewards Implicitly ...](https://arxiv.org/html/2506.14245v2)

8. [[PDF] A Tutorial Introduction to the Minimum Description Length Principle](https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf) - In this chapter, we introduce the MDL Principle in an entirely non-technical way, concentrating on i...

9. [[PDF] Process Reward Models for Reflective Mathematical Reasoning](https://aclanthology.org/2025.findings-emnlp.253.pdf) - One piv- otal method to improve the reasoning ability of. LLMs is Chain-of-Thought (CoT) (Wei et al....

10. [Improving Rule-based Reasoning in LLMs via Neurosymbolic ...](https://arxiv.org/html/2502.01657v2) - In this paper, we introduce a novel method that extends the capabilities of LLMs by encoding their h...

11. [[2501.07301] The Lessons of Developing Process Reward Models ...](https://arxiv.org/abs/2501.07301) - Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical ...

12. [Inference Scaling Laws: An Empirical Analysis of Compute-Optimal ...](https://arxiv.org/html/2408.00724v3) - We explore new inference scaling laws and compute-optimal inference by evaluating the performance of...

13. [[PDF] inference scaling laws: an empirical analysis of compute-optimal ...](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c3caae2f725c8e2a55ecd600563d172-Paper-Conference.pdf) - best-of-n. This strategy, also known as rejection sampling, generates a set of candidates and choose...

14. [[PDF] Ranked Voting based Self-Consistency of Large Language Models](https://ylwangy.github.io/pub/ACL_2025_RVSC.pdf) - Majority voting is considered an effective method to enhance chain-of-thought reason- ing, as it sel...

15. [Ranked Voting based Self-Consistency of Large Language Models](https://arxiv.org/abs/2505.10772) - Majority voting is considered an effective method to enhance chain-of-thought reasoning, as it selec...

16. [[PDF] Confidence Improves Self-Consistency in LLMs - ACL Anthology](https://aclanthology.org/2025.findings-acl.1030.pdf) - Self-consistency decoding enhances LLMs' performance on reasoning tasks by sampling diverse reasonin...

17. [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis) - First, the Kaggle competition results. In total, 1,455 teams submitted 15,154 entries for ARC Prize ...

18. [2025 Competition Details - ARC Prize](https://arcprize.org/competitions/2025/) - 2025 Paper Award Winners · 1st Place - $50k · 2nd Place - $20k · 3rd Place - $5k · Runners Up - $2.5...

19. [On the “ARC-AGI” $1 Million Reasoning Challenge](https://aiguide.substack.com/p/on-the-arc-agi-1-million-reasoning) - ARC is a set of analogy puzzles in which the solver must infer the abstract rule underlying a small ...

20. [[PDF] Neurosymbolic Program Synthesis - UT Austin Computer Science](https://www.cs.utexas.edu/~swarat/pubs/ns-handbook-2025.pdf) - Abstract. We survey neurosymbolic program synthesis, an emerging research area at the interface of d...

21. [Program Synthesis Using Inductive Logic Programming for the ...](https://journals.sagepub.com/doi/10.1177/17248035251363178) - More recently and after our initial work, another ILP based approach to ARC appeared, using POPPER (...

22. [[PDF] Test-Time Training for Long-Context LLMs - arXiv](https://arxiv.org/pdf/2512.13898.pdf) - Our results call for reallocating inference-time budget from thousands of “thinking” tokens to a sma...

23. [[PDF] discover.pdf - Test-Time Training](https://test-time-training.github.io/discover.pdf) - TTT-Discover continues to train an LLM on a single problem at test time. πθi denotes the policy with...

24. [Study could lead to LLMs that are better at complex reasoning](https://news.mit.edu/2025/study-could-lead-llms-better-complex-reasoning-0708) - MIT researchers have shown how strategically applying a method known as test-time training with task...

25. [Test-time training could lead to LLMs that are better at complex ...](https://techxplore.com/news/2025-07-llms-complex.html) - Test-time training, a method that involves temporarily updating some of a model's inner workings dur...

26. [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference ...](https://neurips.cc/virtual/2024/109438) - We introduce an approach aimed at enhancing the reasoning capabilities of Large Language Models (LLM...

27. [Monte Carlo Tree Search for Graph Reasoning in Large Language ...](https://dl.acm.org/doi/pdf/10.1145/3746252.3760854) - In response, we propose Graph-MCTS, a novel framework that enables LLMs to perform stepwise, interac...

28. [ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs ...](https://aclanthology.org/2025.findings-acl.484/) - We propose ReKG-MCTS, a novel training-free framework that synergizes Monte Carlo Tree Search (MCTS)...

29. [CMCTS: A Constrained Monte Carlo Tree Search Framework for ...](https://ui.adsabs.harvard.edu/abs/2025arXiv250211169L/abstract) - This paper introduces the Constrained Monte Carlo Tree Search (CMCTS) framework to enhance the mathe...

30. [Monte Carlo Tree Search for Comprehensive Exploration in LLM ...](https://icml.cc/virtual/2025/poster/45984) - To more comprehensively explore the space of heuristics, this paper proposes to use Monte Carlo Tree...

31. [[PDF] SRA-MCTS: Self-driven Reasoning Augmentation with Monte Carlo ...](https://www.ijcai.org/proceedings/2025/0965.pdf) - To tackle this, we propose a self-driven reasoning augmentation pro- cess, SRA-MCTS, which incorpora...

32. [Improving Rule-based Reasoning in LLMs via Neurosymbolic ...](https://arxiv.org/html/2502.01657v1) - This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden st...

33. [Structured Program Synthesis using LLMs: Results and Insights from ...](https://arxiv.org/html/2506.13820v1) - This paper presents a structured inductive programming approach with LLMs that successfully solves t...

34. [LLM Fan-Out 101: Self-Consistency, Consensus, and Voting Patterns](https://kinde.com/learn/ai-for-software-engineering/workflows/llm-fan-out-101-self-consistency-consensus-and-voting-patterns/) - ## What is the LLM fan-out pattern?Link to this section

The LLM fan-out pattern is a technique for ...

35. [22 Minimum Description Length - Oracle Help Center](https://docs.oracle.com/en/database/oracle/machine-learning/oml4sql/21/dmcon/minimum-description-length.html) - A Minimum Description Length (MDL) model takes into consideration the size of the model as well as t...

36. [A Tutorial Introduction to the Minimum Description Length ...](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf)

37. [Of 2 Minds: How Fast and Slow Thinking Shape Perception and ...](https://www.scientificamerican.com/article/kahneman-excerpt-thinking-fast-and-slow/) - System 1 operates automatically and quickly, with little or no effort and no sense of voluntary cont...

38. [Daniel Kahneman-Thinking, Fast and Slow .pdf](https://dn790002.ca.archive.org/0/items/DanielKahnemanThinkingFastAndSlow/Daniel%20Kahneman-Thinking,%20Fast%20and%20Slow%20%20.pdf)

39. [Daniel Kahneman Explains The Machinery of Thought - Farnam Street](https://fs.blog/daniel-kahneman-the-two-systems/) - Daniel Kahneman dissects the machinery of thought into two agents, system 1 and system two, which re...

40. [Design thinking, fast and slow: A framework for Kahneman's dual ...](https://www.cambridge.org/core/journals/design-science/article/design-thinking-fast-and-slow-a-framework-for-kahnemans-dualsystem-theory-in-design/A200DC637BBDC982D288FC4F8A112DE7) - In his book Thinking, Fast and Slow, Daniel Kahneman presented a model of human cognition based on t...

41. [Reinforcement Learning with Verified Rewards (RLVR)](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr) - Reinforcement Learning with Verified Rewards (RLVR) employs rule-based signals (e.g. answer accuracy...

42. [RLVR: Verifiable Rewards for Reliable Enterprise LLMs | Appenwww.appen.com › blog › rlvr](https://www.appen.com/blog/rlvr) - Understand RLVR and how it differs from RLHF: where each fits, and how enterprises can apply them

43. [Reinforcement Learning with Verifiable Rewards Implicitly ...](https://arxiv.org/abs/2506.14245) - Recent advancements in long chain-of-thought (CoT) reasoning, particularly through the Group Relativ...

