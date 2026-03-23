<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# proceed

Here’s a concrete, forward‑looking roadmap that “operationalizes” the cognitive‑science paper you just had me write, oriented toward the three big themes: predictive/active inference, resource‑rational cognition, and LLMs as tools/models of mind.[^1][^2][^3][^4][^5]

I’ll structure it as research tracks with concrete project ideas.

***

## 1. High‑Level Aims

Over the next 12–24 months, a coherent cognitive‑science program could aim to:

- Build and test **computational models** that unify predictive processing/active inference with resource‑rational constraints.[^2][^6][^5]
- Use **LLMs as both tools and objects** of study: probing their “cognition” with psych methods, and using them to generate and fit cognitive models.[^4][^7][^8]
- Tie all of this back to **behavioral and (when feasible) neural data**, even if you start with public datasets and simulations.[^9][^1]

***

## 2. Track A – Predictive Processing \& Active Inference

### A1. Minimal active‑inference agents in toy environments

Goal: Implement small, fully transparent agents that do perception + action under an active‑inference objective, then compare their behavior to human data.

Concrete steps:

- Start from an accessible active‑inference tutorial (e.g., Sprevak \& Smith’s intro piece and Friston’s process theory).[^10][^2]
- Implement a **grid‑world** or bandit environment where an agent must trade off exploration and exploitation.
- Fit/compare two agents:
    - A standard RL agent (e.g., Q‑learning).
    - An active‑inference agent minimizing expected free energy (with both epistemic and extrinsic value).
- Compare policies and learning curves to human data from existing exploration/exploitation tasks (public datasets, or quick online experiments).

Deliverables:

- A codebase with a clean active‑inference implementation and comparison plots.
- A short paper/tech report: “Active Inference vs RL in Simple Decision Tasks.”


### A2. Predictive‑processing accounts of specific cognitive phenomena

Pick one phenomenon where PP/active inference make clear predictions—e.g.:

- **Mismatch negativity / surprise responses** in oddball paradigms (using existing EEG/MEG datasets).[^11][^9]
- **Perceptual illusions** where priors vs likelihoods trade off (visual or auditory).

Plan:

- Build a simple hierarchical generative model for the stimulus.
- Derive trial‑by‑trial prediction‑error or free‑energy traces.
- Show that these traces align with behavioral or neural signatures better than simpler accounts.

***

## 3. Track B – Resource‑Rational and Computational Rationality

### B1. Resource‑rational analysis of a classic bias

Goal: Show that a canonical bias (e.g., limited depth search in planning, base‑rate neglect, or anchoring) is **near‑optimal** given computation limits.

Steps:

- Choose one well‑studied task with public data (e.g., multi‑step planning, Bayesian reasoning problem).[^6][^5][^12]
- Specify:
    - The **computational problem** (e.g., exact Bayesian update or optimal plan).
    - A **cost model** for mental operations (node expansions, samples, memory writes).
- Derive or numerically search for the algorithm that maximizes expected utility minus cost under that model.
- Fit that algorithm to human behavior, and compare fit vs standard heuristics and ideal‑Bayes models.

Outcome:

- A resource‑rational model that explains both accuracy and RT/error patterns better than purely normative or descriptive competitors.


### B2. Algorithm‑level modeling of “System 1 / System 2”

Using dual‑process tasks (e.g., heuristics \& biases, cognitive reflection tests), build:

- A **fast heuristic module** (e.g., associative retrieval, one‑shot approximate inference).
- A **slow deliberative module** (limited tree search, Monte Carlo sampling).

Then use resource‑rational analysis to decide when the system should invoke the slow module (metareasoning): invest more computation when stakes and uncertainty are high, otherwise default to heuristics.[^5][^13][^14][^6]

***

## 4. Track C – LLMs and Cognitive Science

### C1. LLMs as “participants” in classic cognitive tasks

Goal: Run LLMs through the same paradigms used on humans and compare patterns, not just accuracy.

Projects:

- Reproduce a suite of **classic experiments** (categorization, similarity judgments, framing effects, sequence prediction) with one or more open LLMs.[^8][^15][^4]
- Match:
    - Human **error patterns** and bias signatures.
    - Human **RT proxies** (e.g., number of decoding steps, log‑prob distributions).

You can:

- Use Representational Similarity Analysis (RSA) or embedding comparisons to see if LLM internal representations track human semantic spaces and neural data where available.[^16][^1][^4]
- Explicitly report where LLMs diverge from human behavior in systematic ways.


### C2. LLM‑assisted generation of cognitive models (GeCCo‑style)

Build a pipeline like GeCCo:[^7][^4]

- Input: a task description + human dataset (choices/RTs).
- Use an LLM to:
    - Propose candidate **computational cognitive models** (code, generative story).
    - Fit them (e.g., Bayesian or maximum‑likelihood estimation).
    - Evaluate on held‑out data and iteratively refine via feedback.

Research questions:

- How often does the LLM discover models that match or beat hand‑designed ones?
- What prompt/feedback structures produce interpretable, theoretically meaningful models rather than just overfit ones?


### C3. Active‑inference framing of LLMs

Following work on “LLMs as atypical active‑inference agents,” treat an LLM as a generative model missing a tight action loop.[^3][^4]

Projects:

- Formalize how **next‑token prediction** is a special case of prediction‑error minimization.
- Simulate “closing the loop” by embedding an LLM in a simple interactive environment (e.g., tool‑calling or text‑based gridworld) and analyze its behavior in active‑inference terms.
- Explore predictions about emergent self‑awareness and control when feedback loops tighten.[^3]

***

## 5. Track D – Human–LLM Interaction and Cognitive Impacts

### D1. Measuring cognitive load and strategy shifts during LLM use

Use existing EEG/behavioral work as a template:[^1]

- Design tasks where people solve problems with and without LLM assistance.
- Measure:
    - Performance (accuracy, speed).
    - Self‑reports of effort and confidence.
    - When possible, simple physiological proxies (e.g., pupillometry, HRV) or re‑analyze public EEG datasets.

Questions:

- Do users offload certain reasoning steps to LLMs while retaining others?
- When does LLM support reduce cognitive load vs induce overreliance or deskilling?


### D2. Human‑in‑the‑loop active‑inference with LLMs

Model the human + LLM pair as a **joint active‑inference system**:

- Human policy: when to consult the model, when to override it.
- LLM policy: how much explanation to provide, how to present uncertainty.

Use this to design interaction patterns that minimize **joint** expected free energy—balancing user understanding, error rates, and cognitive effort.[^17][^18][^4][^1]

***

## 6. Infrastructure and Workflow

To make this tractable:

- Standardize on a **small set of core environments/tasks** reused across projects (e.g., a bandit, a simple planning task, a reference cognitive bias task).
- Maintain a **shared modeling stack**:
    - Probabilistic programming (Pyro, NumPyro, or Stan) for Bayesian and resource‑rational models.
    - A small active‑inference library (your own or an existing one).
    - LLM access wrappers for running experiments and GeCCo‑style pipelines.
- Keep all datasets, models, and code in a common repo with Jupyter notebooks that go from **theory → simulation → fit to human/LLM data → figures**.

***

If you tell me which of these tracks best matches what you want to build next (e.g., more theory‑heavy vs more engineering vs more human‑subjects‑oriented), I can turn that track into a step‑by‑step project plan with specific experiments, model classes, and evaluation metrics.

<div align="center">⁂</div>

[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12307350/

[^2]: https://onlinelibrary.wiley.com/doi/abs/10.1111/tops.12704

[^3]: https://arxiv.org/pdf/2311.10215.pdf

[^4]: https://arxiv.org/pdf/2409.02387.pdf

[^5]: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A

[^6]: https://cocosci.princeton.edu/papers/liederresource.pdf

[^7]: https://neurips.cc/virtual/2025/poster/118148

[^8]: https://cogscillm.com

[^9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8602880/

[^10]: https://activeinference.github.io/papers/process_theory.pdf

[^11]: https://www.jneurophilosophy.com/index.php/jnp/article/view/225

[^12]: https://cocosci.princeton.edu/papers/Resource-rational_analysis.pdf

[^13]: https://cocosci.princeton.edu/papers/lieder_resource.pdf

[^14]: https://people.eecs.berkeley.edu/~russell/classes/cs294/f21/papers/Gershman-et-al-Science2015-Computational-Rationality.pdf

[^15]: https://coursetable.com/catalog?course-modal=202401-31582

[^16]: https://arxiv.org/abs/2409.02387v7

[^17]: https://www.youtube.com/watch?v=6F8YtinDWKc

[^18]: https://neuronline.sfn.org/scientific-research/what-does-a-large-language-model-know

