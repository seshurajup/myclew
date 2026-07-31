## MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution

Zefeng Wang *,1 , Minxi Yan *,2 , Jinhe Bi 1 , Sikuan Yan 1 , Volker Tresp 1 , Yunpu Ma 1,3,4

1 LMU Munich, 2 The Chinese University of Hong Kong, 3 MCML, 4 MemAgents Lab

## Abstract

Recent LLM agents tackle increasingly longhorizon, open-ended tasks, and external skills, reusable procedural knowledge supplied to the agent, further extend this capability. However, a fixed, hand-authored skill is rarely optimal, and cannot adapt to the diversity of tasks an agent encounters. Self-improving agents address this by rewriting their own skill files from execution traces, yielding meaningful gains on challenging benchmarks. Yet such self-evolution remains non-recursive : it improves only the task skill ( what the agent does) while the improvement procedure ( how it improves) is authored once and held fixed. We introduce MetaSkill-Evolve , a two-timescale framework that makes agentic skill improvement recursive : every branch carries both a task skill s and a branch-local meta-skill m = ( ψ, σ, α, π, ε ) whose five components parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver agents of the improvement pipeline. Task skills evolve on a fast loop while the meta-skill evolves on a slower one under the same pipeline applied to itself, with no additional model or objective. With all five pipeline agents sharing a single frozen backbone, MetaSkill-Evolve outperforms no-skill, static-skill, and single-level evolution baselines on three agentic benchmarks (OfficeQA, SealQA, ALFWorld), improving held-out test accuracy over the raw backbone by +23.54, +16.09, and +1.92 points respectively.

## 1 Introduction

Language model agents now tackle increasingly long-horizon, open-ended tasks, from document understanding and multi-step reasoning to tool use, yet they rarely succeed out of the box (Yao et al., 2023). A productive remedy is to equip the agent

* Equal contribution.

with a skill : a curated, editable Markdown specification of reusable procedures, now a portable file-system artifact in widely deployed agent harnesses (Wang et al., 2023; Zheng et al., 2025). But a fixed, hand-authored skill is rarely optimal, and cannot anticipate the diversity of tasks an agent encounters. Self-improvement systems such as EvoSkill (Alzubi et al., 2026), GEPA (Agrawal et al., 2026), and SkillWeaver (Zheng et al., 2025) address this by closing the loop with an analyzepropose-evolve pipeline that rewrites the skill after each failure trace, so that iteration by iteration the skill grows more capable.

These systems, however, evolve only what the agent does, not how it evolves : the artifact under optimization changes while the operator that optimizes it stays fixed. In the vocabulary of selfimproving machines (Good, 1965; Schmidhuber, 2006), they are self-improving but stop short of being recursively self-improving. The meta-level logic is hardcoded in advance and shared by every branch throughout the run: how failures are diagnosed, which edits are proposed, how much search effort is allocated, whether cross-branch experience is reused, and how an approved edit is applied to disk (Fig. 1, third panel). A branch therefore cannot improve the way it diagnoses failures: it applies the same procedure to every error, whether a misread table or a faulty calculation, and when that procedure yields the wrong fix, nothing in the loop can revise it.

A closer look at this rigidity suggests that two quantities govern evolutionary skill search. The first is the current skill utility U ( s ) , the score of the present skill on a validation batch. The second is the meta-productivity P ( m | s ) , the rate at which a branch generates stronger descendants under its current improvement policy m . These are not the same: a skill may score well today yet sit in a branch whose meta-level policy produces weak children, while a moderately-performing skill may

Figure 1: Four regimes of agent skill improvement. No-Skill : no reusable skill memory. Static Skill : a handauthored s 0 held fixed (padlocked). Single Level Evolve : the task skill evolves s 0 → s 1 → s 2 , but the driving meta-process stays padlocked. MetaSkill-Evolve (ours): a branch-level meta-skill m = ( ψ, σ, α, π, ε ) co-evolves on a slower outer ring via the same fi ve-agent pipeline that rewrites s , with no extra model and no extra framework.

<!-- image -->

reside in a branch whose policy reliably moves scores up, making it the more promising line to extend even though its present score lags behind. Optimizing only U ( s ) ignores the second quantity entirely, and we hypothesize that this omission is a primary reason fixed-meta evolution stalls once repeated failures share a diagnosis style the metaprocess cannot revise. This motivates the central question of our work:

## RESEARCH QUESTION

Can the improvement procedure itself be evolved as a first-class object alongside the task skills it produces, using the same agentic pipeline?

To this end, we introduce MetaSkill-Evolve (Fig. 1, rightmost panel), a two-timescale evolutionary framework that lifts the improvement procedure into a learnable object, yielding a practical, bounded form of recursive self-improvement in which the improvement operator is reflexively applied to itself. Every branch carries a state b = ( s, m, h ) : a task skill s , a branch-level metaskill m = ( ψ, σ, α, π, ε ) , and an iteration history h . The task skill evolves at every iteration on the fast timescale; the meta-skill evolves every H iterations on the slow timescale, driven by how much the branch's last H descendants improved, i.e., a running measure of whether its current improvement policy is still productive. The five components of m jointly parameterise the improvement loop: ψ diagnoses and tags a failure trace, σ con- trols cross-branch retrieval, α sets the per-iteration child budget, π turns a diagnosis into a concrete edit proposal, and ε applies an approved proposal to the on-disk skill files and verifies that the result is coherent.

Crucially, this adds no architectural component: each component of m is a Markdown skill file identical in format to a task skill, so the same fi veagent pipeline (Analyzer, Retriever, Allocator, Proposer, Evolver) that rewrites s is applied recursively to refine m itself, closing the self-improvement loop on the very operator that performs it. The meta-skill in turn sharpens frontier selection beyond pure utility: each candidate parent is scored by η 1 U ( s ) + η 2 P ( m | s ) + η 3 N ( b ) , where N ( b ) discounts branches already selected many times, steering the search toward branches that are at once productive and underexplored.

We evaluate the resulting system on three agentic benchmarks that stress complementary capabilities: OfficeQA (Opsahl-Ong et al., 2026), SealQA (Pham et al., 2026), and ALFWorld (Shridhar et al., 2021). All five pipeline agents share a single frozen Gemma-4 31B (Google, 2026) backbone, so any improvement is attributable to the evolved skills and meta-skills rather than to added model capacity or training. Against No-Skill, Static-Skill, and a Single-LevelEvolution baseline that ablates meta-skill updates, MetaSkill-Evolve improves held-out test accuracy by +23.54 / +16.09 / +1.92 points on OfficeQA /

SealQA / ALFWorld over No-Skill, and by +6.38 / +8.05 / +1.92 points over Single-Level Evolution. On the two QA benchmarks the progression NoSkill → Static → Single-Level → Ours is monotonic; on ALFWorld the backbone is already near ceiling on the held-out split, so the margins are small and the static skill is roughly neutral.

In summary, our contributions are:

1. Two-timescale framework separating fast taskskill from slow meta-skill evolution, with metaproductivity P ( m | s ) as the slow objective alongside task utility U ( s ) .
2. Five-agent evolution pipeline that extends the fixed analyze → propose → evolve loop with two typed stages they lack: a Retriever ( σ ) for crossbranch sharing and an Allocator ( α ) for an adaptive per-parent child budget, giving Analyzer, Retriever, Allocator, Proposer, Evolver.
3. Recursive self-improvement via typed metaskills. m =( ψ, σ, α, π, ε ) as meta-skills, evolved by the same fi ve-agent pipeline as task skills, a bounded, one-level recursion that needs no new model or objective.
4. Meta-aware frontier selection that scores each candidate parent by utility U ( s ) , metaproductivity P ( m | s ) , and branch novelty N ( b ) , steering search toward branches that are at once productive and underexplored.

## 2 Related Work

## Skill-based agents and skill self-improvement.

A skill is a reusable, named procedure that augments a frozen LLM agent and is now a portable artifact across agent harnesses (Wang et al., 2023; Zheng et al., 2025; Opsahl-Ong et al., 2026; Li et al., 2026; Jiang et al., 2026; Zhao et al., 2026). A growing line closes the loop with execution feedback (reflection, distillation, analyze → propose → evolve and RL-driven rewrites, trajectory and memory-based methods) (Shinn et al., 2023; Zhao et al., 2024; Alzubi et al., 2026; Xia et al., 2026; Shi et al., 2026; Lin et al., 2025; Zhang et al., 2026b; Fang et al., 2026; Ni et al., 2026; Tian et al., 2026; Si et al., 2026; Huang et al., 2025; Yang et al., 2026a; Ma et al., 2026a; Wan et al., 2025b), while a parallel sub-line grows a skill library rather than a single artifact (Yang et al., 2026b; Liu et al., 2026; Shen et al., 2026; Wang et al., 2026; Ma et al., 2026b; Zhang et al., 2026a). Throughout, the improvement procedure

(how failures are diagnosed, edits scoped, search allocated, experience reused) is authored once and held fixed. MetaSkill-Evolve makes that procedure a first-class, branch-local object, co-evolved by the same pipeline.

Recursive self-improvement. Improving a system's own capacity to improve traces to ultraintelligent machines (Good, 1965; Wan et al., 2025a; Tian et al., 2025) and the Gödel machine, which rewrites its own code under a proof of improvement (Schmidhuber, 2006). LLM-era instances drop the proof, improving a code-improvement scaffold (Zelikman et al., 2024), co-evolving prompts and their mutation-prompts (Fernando et al., 2023), or rewriting their own designs (Hu et al., 2025; Zhang et al., 2026c). These systems recurse on code or prompts under one global policy. MetaSkill-Evolve instead recurses on skill files : the operator is a five-agent pipeline parameterised by a branch-local meta-skill that the same pipeline refines: a bounded, one-level recursion that adds no model and keeps a per-lineage rather than global policy.

## Prompt, textual, and evolutionary optimisation.

Another body of work treats the agent's prompt or context as the object of optimisation. Selffeedback, LLM optimisers, prompt compilation, textual gradients, evolutionary prompt search, and in-context search all improve the instructions seen by a fixed agent (Madaan et al., 2023; Bi et al., 2025a; Yang et al., 2024; Khattab et al., 2023; Pryzant et al., 2023; Zhou et al., 2023; Yuksekgonul et al., 2024; Guo et al., 2025; Bi et al., 2025c; Agrawal et al., 2026; Lee et al., 2025; Ouyang et al., 2026; Ye et al., 2026; Bi et al., 2025b). Populationbased program search makes the search object more executable by pairing behavioural archives with verifiers and recent extensions to coding agents and open-ended evolution (Novikov et al., 2025; Lange et al., 2025; Wang et al., 2025; He et al., 2026; Bi et al., 2026). All rewrite a single artifact under one fixed rule. We instead let the rule diverge across branches and evolve on its own timescale; our frontier score η 1 U v + η 2 ˆ P v + η 3 N v uses ˆ P v as a qualitydiversity descriptor preserving improvement-policy diversity.

## 3 MetaSkill-Evolve

This section presents MetaSkill-Evolve. We first formalise task-skill evolution and its utility objec-

tive (§3.1), then introduce the per-branch meta-skill that parameterises the search (§3.2). We next describe the persistent evolution graph and the score that decides which branch to expand (§3.3). The final two subsections detail the algorithm's two timescales: a fast loop that evolves task skills on the selected parent (§3.4), and a slow loop that, every H iterations, evolves the meta-skill itself by reapplying the same five-agent pipeline to the hidden meta-skill files (§3.5).

## 3.1 Problem Formulation

Let T be a task with inputs x and expected outputs y . A task skill s is a Markdown-format LLM-agent program specifying procedures, tools, and heuristics for T . Writing A s for the agent that executes skill s , its utility is the expected task reward

$$U ( s ) \, = \, \mathbb { E } _ { ( x , y ) \sim \mathcal { T } } [ r ( A _ { s } ( x ) , y ) ] \, , \quad ( 1 ) \quad ,$$

where r ( · , · ) ∈ [0 , 1] scores a prediction against the reference output. Because T is accessible only through samples, U ( s ) is estimated as accuracy on a held-out validation batch.

## 3.2 Branch State and Meta-Skill

Each task-skill iteration turns a failure into a skill edit through a fixed five-step procedure, i.e., diagnose, retrieve, allocate, propose, execute. MetaSkill-Evolve makes that procedure adaptive by attaching to each branch a meta-skill m that parameterises all five steps. A branch state is b = ( s, m, h ) , where h is the branch's iteration history, and

$$m = ( \psi , \, \sigma , \, \alpha , \, \pi , \, \varepsilon ) .$$

Each component is itself a Markdown-format LLMagent program (a SKILL.md fi le) consumed by exactly one specialist agent:

- ψ -diagnosis policy (Analyzer): maps failures to a tag ϕ and free-form analysis a .
- σ -sharing policy (Retriever): selects samebranch and cross-branch inspirations matching ϕ .
- α -allocation policy (Allocator): sets the child budget K ∈ [1 , K max ] per step.
- π -edit-proposal policy (Proposer): emits an edit δ conditioned on the worst case, analysis, and retrieved inspirations, i.e., ( f, a, I ) .
- ε -edit-executor policy (Evolver): writes δ to disk and verifies the result.

Since each meta-skill file uses the same Markdown representation as the task-skill files the pipeline already consumes, the same five agents that improve s also improve m when applied recursively.

A meta-skill is good insofar as it converts iterations into utility gains. We make this precise through the meta-productivity of m at skill s , the expected per-child improvement over K proposals,

$$P ( m \, | \, s ) = \mathbb { E } \left [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left ( U ( s ^ { \prime } _ { k } ) - U ( s ) \right ) \right ] , \ ( 3 )$$

estimated per node by the empirical mean ˆ P v = ∆ U children of v (zero for nodes with no children). MetaSkill-Evolve jointly maximises task utility U ( s ) and meta-productivity P ( m | s ) across all branches: the fast loop improves the task skill s , and the slow loop improves the meta-skill m that produces those improvements.

## 3.3 Evolution Graph and Frontier Selection

We record the entire search history as a directed acyclic graph (DAG) G = ( V , E ) persisted in SQLite. Each node v ∈ V is one evaluated branch state and stores ( s v , m v , U v , ∆ U v , ϕ v ) together with its branch path and selection counter. Edges are directed and of two kinds: a lineage edge u → v marks v as the child produced by evolving its parent u , and an inspiration edge records a cross-branch node that σ retrieved when proposing v . Both edge types point from an earlier node to a later one, so G is acyclic by construction: each node is created once, from already-existing nodes, and is never revised in place. Persisting G rather than keeping a fixed-size beam lets us revisit previously deprioritised lineages, supports cross-branch retrieval, and preserves full provenance for any final node.

A child enters the archive (the pool of deployable states that also serves as the candidate set for future parents) only when it strictly improves on its parent, ∆ U v &gt; 0 . Accuracy-neutral or regressing children ( ∆ U v ≤ 0 ) are not eligible to be selected as parents, but are still persisted in G : they preserve provenance and remain available to σ as inspiration, so a neutral or failed edit can still inform a later proposal.

From the archive we refresh the frontier F each iteration to the topK nodes by

$$v ^ { * } = \arg \max _ { v \in \mathcal { F } } [ \eta _ { 1 } U _ { v } + \eta _ { 2 } \hat { P } _ { v } + \eta _ { 3 } N _ { v } ] , \ \ ( 4 )$$

where ˆ P v is the meta-productivity estimate (Eq. 3)

Figure 2: System overview. The branch state b = ( s, m, h ) (left) feeds the five-agent pipeline (centre), whose output is appended to the SQLite node graph (right). Frontier selection (Eq. 4) draws the next parent from the graph.

<!-- image -->

and N v = 1 / (1 + times\_selected v ) . Each term targets a distinct failure mode of greedy search:

- U v - exploitation: prevents chasing volatile gain estimates on weak parents.
- ˆ P v - trajectory quality: redirects effort from plateaued high points to nodes still generating useful descendants.
- N v -visitation cooling: a node selected k times must outperform an unselected sibling by η 3 k/ ( k +1) to be re-picked, preventing budget monopolies.

Setting any η i =0 exposes the corresponding mode: the frontier locks on stagnated high-utility nodes ( η 2 =0 ), collapses to one lineage ( η 3 =0 ), or trusts noisy single-child gains as parent quality ( η 1 =0 ). Crucially, we do not fi lter by lineage: diversity is a property of the score, not a structural constraint.

## 3.4 Fast Timescale: Task-Skill Evolution

The fast loop (Algorithm 1) runs one task-skill iteration on a frontier parent v . Before invoking any agent, the runtime restores the selected branch's task and meta snapshots ( s v , m v ) to disk. Thus the SQLite DAG, rather than whatever files remain in the working tree, defines branch state: each branch starts from its recorded snapshot and is evaluated in isolation, preventing leakage between lineages. We then score s v on the training batch and take its worst-scoring example as the diagnostic target: a deliberately high-signal choice, hedged against outliers because each resulting edit is judged by its validation gain ∆ U v rather than by that single training case. This example drives the five-agent pipeline:

- Analyzer ( ψ ) : emits a tag ϕ and a free-form analysis a ; the tag vocabulary is itself maintained by ψ and revised by the slow loop.
- Retriever ( σ ) : ranks a ϕ -matched candidate pool by tag similarity, over-fetching to 3 × the inspiration budget, then LLM-re-ranks this wider pool down to the inspirations I handed to the Proposer; the breadth/depth balance is itself a learned object.
- Allocator ( α ) : chooses K ∈ [1 , K max ] , widening search after stagnation ( ˆ P ≈ 0 ) and contracting after a productive edit.
- Proposer ( π ) : for each of the K children emits an edit δ ; when K&gt; 1 a diversity hint steers the k -th proposer toward a distinct intervention angle, reducing near-duplicate children.
- Evolver ( ε ) : translates δ into file writes via skill\_tools and verifies the result with a before/after hash check that flags edits leaving the target files unchanged.

Each child s ′ k is evaluated on D val to obtain its gain ∆ U k . Every H iterations the slow loop (§3.5) then refreshes the meta-skill m v ; the K children are committed to G carrying this refreshed meta-skill, and the frontier is re-synchronised before the next iteration.

## 3.5 Slow Timescale: Meta-Skill Evolution

Updating m at every iteration would expose the meta-skill to the same single-example noise that drives task-skill evolution. The slow loop (Algorithm 2) instead fires once every H fast iterations and aggregates over that horizon, trading reactivity for stability. Its driving signal is the empirical meta-

## Algorithm 1 Fast timescale: one task-skill iteration

Input: Frontier parent v ; train batch D train, val batch D val Output: K committed child nodes (those with ∆ U &gt; 0 enter the archive)

```
Restore and evaluate 1: Restore snapshots s v (task), m v (meta) to disk 2: E ← Eval ( s v , D train ) ▷ collect failures 3: if E has no failures then return all_passed 4: end if 5: f ← arg min e ∈E score ( e ) ▷ worst case Diagnose and plan (3 agents) 6: ϕ, a ← Analyzer ( f, m v .ψ ) ▷ tag, analysis 7: I ← Retriever ( ϕ, b v , m v .σ ) ▷ inspiring nodes 8: K ← Allocator ( h v , a, I , m v .α ) ▷ child budget Propose and evolve ( K children) 9: for k = 1 . . . K do 10: Restore s v 11: δ ← Proposer ( f, a, I , m v .π ) 12: s ′ k ← Evolver ( s v , δ, m v .ε ) 13: U ′ k ← Eval ( s ′ k , D val ) ; ∆ U k ← U ′ k -U v 14: end for Interleave slow loop, commit and sync 15: if t mod H = 0 then m v ← update meta-skill (Alg. 2) 16: end if 17: Commit children {⟨ s ′ k , ∆ U k , m v ⟩} K k =1 to G 18: F ← SyncFrontier ( F ) ▷ next parent drawn by Eq. 4
```

productivity ˆ P ( m | s ) = 1 |H| ∑ u ∈H ∆ U u over the last H descendants H . We fold ˆ P together with the tags, diagnoses, and outcomes of that window into a synthetic meta-failure trace f m : the improvement history reshaped to look like one failing training example, so that a single Analyzer prompt serves both timescales.

We then re-run the same five-agent pipeline on f m , switching its target object from task-skill files to the hidden meta-skill files. The Analyzer names the single most-implicated component of { ψ, σ, α, π, ε } ; this diagnosis fixes the failure tag ϕ m that steers retrieval, but it does not narrow the edit scope. The Retriever surfaces cross-branch lineages whose meta-failure tags match ϕ m , the Allocator sets the round budget K m , and the Proposer and Evolver then co-edit all five fi les per round. The resulting snapshot of all five files becomes the child's meta\_state\_json . Each branch therefore carries its own lineage-local m , and the sole channel by which one lineage's improvement policy reaches another is this meta-level retrieval, so escape strategies propagate between lineages without any shared global state. Three details separate this from a plain fast-loop run:

- Constrained Analyzer. A null or task-skill diagnosis triggers a round-robin fallback over

Algorithm 2 Slow timescale: meta-skill update, at

```
t mod H = 0
```

```
Input: Branch history H (last H children); meta-skill m ; parent v ; names M = { ψ, σ, α, π, ε } Output: Snapshot of all five meta-skill files Build meta-failure trace 1: ˆ P ← 1 |H| ∑ u ∈H ∆ U u ▷ meta-productivity 2: f m ← trace of tags, diagnoses, outcomes over H , ˆ P Diagnose and plan (target ∈ M ) 3: ϕ m , a m ← Analyzer ( f m , m ) ▷ round-robin fallback 4: I m ← Retriever ( ϕ m , b v , m.σ ) 5: K m ← Allocator ( H , a m , I m , m.α ) Wholem rewrite ( K m accumulating children) 6: for k = 1 . . . K m do 7: Render meta-files from disk ▷ reflects child k -1 8: for j ∈ M do ▷ Proposer: sequential 9: δ ( j ) m ← Proposer ( f m , a m | tgt = j, I m , m.π ) 10: end for 11: { m ( j ) } ← ParallelEvolver ( { δ ( j ) m } j ∈M , m.ε ) ▷ one worker/file 12: end for 13: return snapshot of { m ( j ) } j ∈M
```

{ ψ, σ, α, π, ε } , so the slow loop never aborts on an unusable target nor silently degrades into a redundant fast-loop run.

- Wholem rewrite. Each child edits all five meta-skill files in one step (Proposer sequential, Evolvers parallel), preserving cross-component coherence; for instance, a π edit assuming a finer tag vocabulary is co-applied with the matching ψ edit.
- Accumulating children. Child k +1 reads the files as written by child k , not by the parent; this moving target drives incremental refinement rather than K m independent overwrites that average back to the parent.

## 4 Experiments

## 4.1 Setup

Benchmarks and backbone. We evaluate on three agentic benchmarks chosen to span complementary capabilities: OfficeQA (Opsahl-Ong et al., 2026), SealQA (Pham et al., 2026), and ALFWorld (Shridhar et al., 2021).

Within the evolution loop, each benchmark file is split by stratified sampling over its category column into three disjoint partitions: a training partition (failure mining), a validation partition (child scoring and best-skill selection), and a held-out test partition that the loop never observes. We then report accuracy of the selected skill on the held-out

test partition through a separate benchmark -mode pass.

A single frozen base model, Gemma-4 31B (Google, 2026), serves all five pipeline agents (Analyzer, Retriever, Allocator, Proposer, Evolver); no agent is fine-tuned, so all gains are attributable to evolved skills and meta-skills.

Baselines. We compare four configurations, all sharing the same backbone:

- No-Skill : the base agent with no skill loaded and reflection disabled. Quantifies the raw backbone.
- Static Skill : the same agent loaded with our hand-authored initial skill, held fixed for the entire run. Isolates the value of a skill artifact per se .
- Single-Level Evolution : our fast loop with the slow loop frozen ( K max =1 , no cross-branch sharing, no meta-skill updates). Isolates the contribution of task-skill evolution alone.
- MetaSkill-Evolve (ours): the full two-timescale system.

## 4.2 Main Results

Table 1 and Fig. 3 report held-out test accuracy across the four conditions on all three benchmarks. The two QA benchmarks, where the backbone has the most headroom, tell the cleanest story. First , the static skill is worth several points over the raw backbone on OfficeQA (+4.31) and is roughly neutral on SealQA (+0.24). Second , replacing the fixed skill by single-level evolution (our fast loop with the slow loop frozen) adds a further +12.85 / +7.80 points, isolating the contribution of the inner evolution loop and the SQLite-backed evolution graph. Third , switching on the slow loop lifts performance by another +6.38 / +8.05 points, a gain attributable to meta-skill adaptation, since the only difference from the Single-Level baseline is whether the meta-skill files { ψ, σ, α, π, ε } are themselves evolved. The progression No-Skill → Static → Single-Level → MetaSkill-Evolve is monotonic on both QA benchmarks, so each design choice (adding a skill, evolving it, then evolving the procedure that evolves it) pays off independently: end-toend, MetaSkill-Evolve improves held-out accuracy over the raw backbone by +23.54 points on OfficeQA and +16.09 on SealQA.

ALFWorld stresses the opposite regime: the backbone already solves most episodes (92.31%), leaving little room to improve. The static skill

Figure 3: End-to-end held-out test accuracy on the three benchmarks. All conditions share the Gemma-4 31B backbone; only the skill-evolution strategy differs. Red annotations mark MetaSkill-Evolve's gain over NoSkill; the No-Skill → Ours ordering improves on every benchmark, with the largest margins on the two QA tasks where the backbone has the most headroom.

<!-- image -->

slightly regresses ( -1 . 93 ), and single-level evolution only recovers to the No-Skill baseline (92.31%), so neither non-meta step yields a net gain over the raw backbone. The slow loop alone supplies the entire end-to-end improvement of +1.92 points (to 94.23%): small in absolute terms, but it indicates that meta-skill adaptation remains the operative ingredient even once task-skill evolution has saturated.

## 4.3 Component Ablations

To attribute the gains to specific design choices we disable one component at a time and re-run on all three benchmarks (Table 3 in App. E; QA benchmarks visualised in Fig. 4). The configurations -ψ , -σ , -α , -π ablate the corresponding meta-skill component. In particular, -σ removes the Retriever's inspiration policy entirely; the separate no cross-branch condition (denoted -σ x in Fig. 4) removes only cross-branch candidates while same-branch inspirations remain available. The no meta-updates condition freezes the slow loop entirely. The Evolver ε always executes (with -π it consumes the raw analysis instead of a structured proposal), so ε does not carry its own ablation row.

Fig. 4 shows the component ablations on the two QA benchmarks. There are three findings. First, every typed component contributes : no singlecomponent ablation matches the full system, although which component matters most differs by

Table 1: Main results. Held-out test accuracy (%) on OfficeQA, SealQA, and ALFWorld; the test partition is never seen during evolution (App. F). ALFWorld reports the aggregate task success rate. All rows share the Gemma-4 31B backbone; only the skill-evolution strategy differs.

| Method                     | OfficeQA                   | SealQA   | ALFWorld   |
|----------------------------|----------------------------|----------|------------|
| Non-evolutionary baselines | Non-evolutionary baselines |          |            |
| No-Skill                   | 31.78                      | 29.17    | 92.31      |
| Static skill               | 36.09                      | 29.41    | 90.38      |
| Self evolution             |                            |          |            |
| Single-Level               | 48.94                      | 37.21    | 92.31      |
| MetaSkill-Evolve           | 55.32                      | 45.26    | 94.23      |
| ∆ vs. No-Skill             | +23.54                     | +16.09   | +1.92      |

-

Figure 4: Component ablations on the two QA benchmarks. Blue polygon: accuracy with one meta-skill component removed; dashed rings: full MetaSkillEvolve and Static-Skill references. Dominant component differs by domain: α on OfficeQA, π on SealQA. ALFWorld in Table 3.

<!-- image -->

domain. Second, on OfficeQA the allocation policy α is the single most important component ( 55 . 32 → 35 . 58 , -19 . 7 pts): the OfficeQA failure landscape contains pockets of related arithmetic errors where α 's adaptive widening of the child budget after stagnation is what produces a successful child at all, and π ( -17 . 7 pts) is a close second. Third, on SealQA the edit-proposal policy π is instead dominant ( 45 . 26 → 36 . 84 ), where the gain hinges on the precise content of each edit rather than on how widely the search fans out.

## 4.4 Meta-Update Horizon

The horizon H governs the coupling between the two timescales: it sets how many fast task-skill iterations elapse between consecutive meta-skill evolutions (§3.5). The choice cuts two ways. Firing the slow loop too often exposes the meta-skill

σ

Figure 5: Meta-update horizon sweep. Held-out test accuracy as the horizon H (fast iterations between consecutive meta-skill evolutions) widens from 2 to 8, with the meta-update count held fixed at three, so total iterations = 3 H (shown in parentheses). The broken y -axis separates near-ceiling ALFWorld (top) from the two QA accuracies (bottom). The default H =2 (shaded band) is best on every benchmark; OfficeQA is by far the most sensitive to a stale meta-skill.

<!-- image -->

to the single-example noise that aggregating over H is meant to filter; firing it too rarely lets the meta-skill go stale relative to the drifting task skill, so its broader rewrites land on edits the fast loop has already moved past and overwrite productive changes. To probe this trade-off while holding total compute fixed, we keep the number of metaupdates at three and scale the iteration budget with H , sweeping H ∈ { 2 , 4 , 8 } (equivalently 6, 12, and 24 fast iterations). Fixing the meta-update count makes the H =2 point here (6 iterations, three meta-updates) a slightly different operating point from the five-iteration default (two meta-updates at H =2 ) behind Tables 1 and 3: although all three are scored on the same held-out test partition, the extra iteration and meta-update mean its accuracies are not expected to coincide cell-for-cell with the full-system rows there.

Figure 5 (exact values in Table 2, App. D) reports the sweep on the held-out test set. The tightest spacing H =2 is best on every benchmark, and accuracy falls as the meta-skill is refreshed less often, but the magnitude is strongly benchmark-dependent. OfficeQA is the most sensitive, shedding 9 . 1 points from H =2 to H =8 ( 48 . 94 → 41 . 35 → 39 . 84 ); SealQA and ALFWorld are nearly flat between H =2 and H =4 and slip

only at H =8 (a 0 . 9 - and 1 . 9 -point drop overall). All three horizons already aggregate over multiple iterations, escaping the per-iteration noise that motivates H&gt; 1 ; among them the most reactive schedule wins, which fixes our default at H =2 .

## 5 Conclusion

We introduced MetaSkill-Evolve, a two-timescale framework in which every branch carries a task skill s and a meta-skill m = ( ψ, σ, α, π, ε ) whose five components parameterise the improvement pipeline. Because each component of m is itself a Markdown LLM-agent program, the same pipeline that rewrites s on the fast loop refines m on the slow loop, a bounded instance of recursive selfimprovement that needs no extra model or training, while frontier selection η 1 U v + η 2 ˆ P v + η 3 N v redirects search from plateaued branches to ones whose improvement policy is still productive. The result is a +23.54 / +16.09 / +1.92 point gain in held-out test accuracy over the raw Gemma-4 31B backbone on OfficeQA / SealQA / ALFWorld, of which the slow loop contributes +6.38 / +8.05 / +1.92, evidence that an agent's improvement policy admits the same search machinery as its task behaviour, and that separating what to do from how to improve keeps each loop's signal interpretable.

## Limitations

MetaSkill-Evolve is evaluated on three curated benchmarks; transfer to open-ended, long-horizon real-world tasks with noisier feedback is untested. The five-agent pipeline is itself fixed: we evolve the skills it produces but not its roles or wiring. Meta-updates fire at a fixed horizon H .

## References

Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. 2026. Gepa: Reflective prompt evolution can outperform reinforcement learning. Preprint , arXiv:2507.19457.

- Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. 2026. Evoskill: Automated skill discovery for multi-agent systems. Preprint , arXiv:2603.02766.
- Jinhe Bi, Aniri, Minglai Yang, Xingcheng Zhou, Wenke Huang, Sikuan Yan, Yujun Wang, Zixuan Cao,

Michael Färber, Xun Xiao, Volker Tresp, and Yunpu Ma. 2026. EchoRL: Reinforcement learning via rollout echoing. In Forty-third International Conference on Machine Learning .

Jinhe Bi, Yifan Wang, Danqi Yan, Xun Xiao, Artur Hecker, Volker Tresp, and Yunpu Ma. 2025a. Prism: Self-pruning intrinsic selection method for training-free multimodal data selection. ArXiv , abs/2502.12119.

Jinhe Bi, Yujun Wang, Haokun Chen, Xun Xiao, Artur Hecker, Volker Tresp, and Yunpu Ma. 2025b. LLaVA steering: Visual instruction tuning with 500x fewer parameters through modality linear representationsteering. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 15230-15250, Vienna, Austria. Association for Computational Linguistics.

- Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hin rich Schuetze, Volker Tresp, and Yunpu Ma. 2025c. Cot-kinetics: A theoretical modeling assessing lrm reasoning process. ArXiv , abs/2505.13408.
- Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2026. Memp: Exploring agent procedural memory. Preprint , arXiv:2508.06433.
- Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. 2023. Promptbreeder: Self-referential self-improvement via prompt evolution. Preprint , arXiv:2309.16797.
- I. [J. Good. 1965. Speculations concerning the first ultraintelligent machine. Adv. Comput. , 6:31-88.](https://api.semanticscholar.org/CorpusID:17886872)
- Google. 2026. google/gemma-4-31b. Hugging Face model repository.
- Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. 2025. Evoprompt: Connecting llms with evolutionary algorithms yields powerful prompt optimizers. Preprint , arXiv:2309.08532.
- Yufei He, Juncheng Liu, Yue Liu, Yibo Li, Tri Cao, Zhiyuan Hu, Xinxing Xu, and Bryan Hooi. 2026. Evotest: Evolutionary test-time learning for self-improving agentic systems. Preprint , arXiv:2510.13220.
- Shengran Hu, Cong Lu, and Jeff Clune. 2025. Automated design of agentic systems. Preprint , arXiv:2408.08435.
- Xingyue Huang, Rishabh, Gregor Franke, Ziyi Yang, Jiamu Bai, Weijie Bai, Jinhe Bi, Zifeng Ding, Yiqun Duan, Chengyu Fan, Wendong Fan, Xin Gao, Ruohao Guo, Yuan He, Zhuangzhuang He, Xianglong Hu, Neil Johnson, Bowen Li, Fangru Lin, and 27 others. 2025. Loong: Synthesize long chain-of-thoughts at scale through verifiers. Preprint , arXiv:2509.03059.

- Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. 2026. Sok: Agentic skills - beyond tool use in llm agents. Preprint , arXiv:2602.20867.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. Preprint , arXiv:2310.03714.

Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. 2025. Shinkaevolve: Towards open-ended and sample-efficient program evolution. Preprint , arXiv:2509.19349.

Yoonho Lee, Joseph Boen, and Chelsea Finn. 2025. Feedback descent: Open-ended text optimization via pairwise comparison. Preprint , arXiv:2511.07919.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, Chujun Tao, Binxu Li, Xuandong Zhao, Hejia Geng, Xiaojun Wu, Junwei Zhou, Xiaokun Chen, Hanwen Xing, Yubo Li, and 59 others. 2026. Skillsbench: Benchmarking how well agent skills work across diverse tasks. Preprint , arXiv:2602.12670.

Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Hongzhang Liu, Ronghao Chen, Yangfan He, Daxin Jiang, Binxing Jiao, Chen Hu, and Huacan Wang. 2025. Seagent: Self-evolution trajectory optimization in multistep reasoning with llm-based agents. Preprint , arXiv:2508.02085.

Xingyan Liu, Xiyue Luo, Linyu Li, Gang Huang, Jianfeng Liu, and Hongli Qiao. 2026. Skillforge: Forging domain-specific, self-evolving agent skills in cloud technical support. ArXiv , abs/2604.08618.

Xiaowen Ma, Yunpu Ma, Chenyang Lin, Sikuan Yan, Jinhe Bi, Zixuan Cao, Yijun Tian, Volker Tresp, and Hinrich Schuetze. 2026a. Self-evolving multi-agent systems via textual backpropagation. In Findings of the Association for Computational Linguistics: ACL 2026 , pages 9918-9951, San Diego, California, United States. Association for Computational Linguistics.

Ziyu Ma, Shidong Yang, Yuxiang Ji, Xucong Wang, Yong Wang, Yiming Hu, Tongwen Huang, and Xiangxiang Chu. 2026b. Skillclaw: Let skills evolve collectively with agentic evolver. Preprint , arXiv:2604.08377.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint , arXiv:2303.17651.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2skill: Distill trajectory-local lessons into transferable agent skills. Preprint , arXiv:2603.25158.

Alexander Novikov, Ngân V˜ u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. 2025. Alphaevolve: A coding agent for scientific and algorithmic discovery. Preprint , arXiv:2506.13131.

- Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, and Xing Chen. 2026. Officeqa pro: An enterprise benchmark for end-to-end grounded reasoning. Preprint , arXiv:2603.08655.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2026. Reasoningbank: Scaling agent self-evolving with reasoning memory. Preprint , arXiv:2509.25140.

Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, and Tu Vu. 2026. Sealqa: Raising the bar for reasoning in search-augmented language models. Preprint , arXiv:2506.01062.

- Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with "gradient descent" and beam search. Preprint , arXiv:2305.03495.

[Juergen Schmidhuber. 2006. Goedel machines: Self-referential universal problem solvers making provably optimal self-improvements. Preprint , arXiv:cs/0309048.](https://arxiv.org/abs/cs/0309048)

- Shuaike Shen, Wenduo Cheng, Mingqian Ma, Alistair Turcan, Martin Jinye Zhang, and Jian Ma. 2026. Skillfoundry: Building self-evolving agent skill libraries from heterogeneous scientific resources. Preprint , arXiv:2604.03964.

Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi GU, Xunliang Cai, Xiang Wang, and An Zhang. 2026. Skill1: Unified evolution of skill-augmented agents via reinforcement learning. Preprint , arXiv:2605.06130.

- Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Preprint , arXiv:2303.11366.

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. Alfworld: Aligning text and embodied environments for interactive learning. Preprint , arXiv:2010.03768.
- Shuzheng Si, Haozhe Zhao, Yu Lei, Qingyi Wang, Dingwei Chen, Zhitong Wang, Zhenhailong Wang, Kangyang Luo, Zheng Wang, Gang Chen, Fanchao Qi, Minjia Zhang, and Maosong Sun. 2026. From context to skills: Can language models learn from context skillfully? Preprint , arXiv:2604.27660.
- Yijun Tian, Shaoyu Chen, Zhichao Xu, Yawei Wang, Jinhe Bi, Peng Han, and Wei Wang. 2025. Reinforcement mid-training. Preprint , arXiv:2509.24375.
- Yu Tian, Jiawei Chen, Lifan Zheng, Mingxiang Tao, Xinyi Zeng, Zhaoxia Yin, Hang Su, and Xian Sun. 2026. Skills-coach: A self-evolving skill optimizer via training-free grpo. Preprint , arXiv:2604.27488.
- Guancheng Wan, Lucheng Fu, Haoxin Liu, Yiqiao Jin, Hui Yi Leong, Eric Hanchen Jiang, Hejia Geng, Jinhe Bi, Yunpu Ma, Xiangru Tang, B. Aditya Prakash, Yizhou Sun, and Wei Wang. 2025a. Beyond magic words: Sharpness-aware prompt evolving for robust large language models with tare. Preprint , arXiv:2509.24130.
- Guancheng Wan, Xiaoran Shang, Yuxin Wu, Guibin Zhang, Jinhe Bi, Liangtao Zheng, Xin Lin, Yue Liu, Yanbiao Ma, Wenke Huang, and Bo Du. 2025b. HYPERION: Fine-grained hypersphere alignment for robust federated graph learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems .
- Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, and Shumin Deng. 2026. Skillx: Automatically constructing skill knowledge bases for agents. Preprint , arXiv:2604.04804.
- Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An openended embodied agent with large language models. Preprint , arXiv:2305.16291.
- Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. 2025. Thetaevolve: Test-time learning on open problems. Preprint , arXiv:2511.23473.
- Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. 2026. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. Preprint , arXiv:2602.08234.
- Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. Preprint , arXiv:2309.03409.
- Minglai Yang, Xinyu Guo, Zhengliang Shi, Jinhe Bi, Steven Bethard, Mihai Surdeanu, and Liangming Pan. 2026a. Alignsae: Concept-aligned sparse autoencoders. Preprint , arXiv:2512.02004.
- Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, and Liang He. 2026b. Autoskill: Experience-driven lifelong learning via skill selfevolution. Preprint , arXiv:2603.01145.
- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. Preprint , arXiv:2210.03629.
- Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, and Guojie Song. 2026. Meta context engineering via agentic skill evolution. Preprint , arXiv:2601.21557.
- Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. 2024. Textgrad: Automatic "differentiation" via text. Preprint , arXiv:2406.07496.
- Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. 2024. Self-taught optimizer (stop): Recursively self-improving code generation. Preprint , arXiv:2310.02304.
- Hanrong Zhang, Shicheng Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayu Zhou, Chengze Li, WeiChieh Huang, Yifei Yao, Kening Zheng, Xue Liu, Xiaoxiao Li, and Philip S. Yu. 2026a. Coevoskills: Self-evolving agent skills via co-evolutionary verification. Preprint , arXiv:2604.01687.
- Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. 2026b. Memskill: Learning and evolving memory skills for self-evolving agents. Preprint , arXiv:2602.02474.
- Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. 2026c. Darwin godel machine: Openended evolution of self-improving agents. Preprint , arXiv:2505.22954.
- Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. Preprint , arXiv:2308.10144.
- Xuanle Zhao, Qiushi Sun, Jingyu Xiao, Xuexin Liu, Haoyue Yang, Qiaosheng Chen, Xianzhen Luo, Jing Huang, Yufeng Zhong, Lei Chen, Shuai Fu, Zhenlin Wei, Jinhe Bi, Lei Jiang, Haibo Qiu, Siqi Yang, Peng Shi, Jian Hu, and Zhixiong Zeng. 2026. Beyond nl2code: A structured survey of multimodal code intelligence. Preprint , arXiv:2606.15932.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. 2025. Skillweaver: Web agents can self-improve by discovering and honing skills. Preprint , arXiv:2504.07079.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2023. Large language models are human-level prompt engineers. Preprint , arXiv:2211.01910.

## A Five-Agent Pipeline Details

All five agents are LLM-backed ToolCallingAgent instances sharing the same frozen base model; they load the skill catalog (task skills or meta-skills, depending on context) as additional context.

Analyzer ( ψ ). Given the execution trace and failure reason of the worst-scoring training example, the Analyzer produces (i) a structured root-cause analysis, (ii) a short failure tag ϕ (at most 15 words), and (iii) the target skill file to edit. It uses a three-layer recovery chain: primary parse of final\_answer , step-level output scanning across all agent steps, and a repair call with constrained response\_format . The target\_skill fi eld is auto-derived from relevant\_sections when not explicitly provided, improving robustness with smaller models.

Retriever ( σ ). Given ϕ and the current branch path, the Retriever fetches same-branch and crossbranch candidates from the SQLite graph store, over-fetched to 3 L same and 3 L cross by tag similarity, and LLM-re-ranks them by relevance to the present failure, returning the subset it judges relevant.

Allocator ( α ). Given recent ∆ U values along the branch and the analysis, the Allocator chooses a child budget K ∈ [1 , K max ] , allocating more search effort on stagnation and less when recent edits have been productive.

Proposer ( π ). Given the failure analysis, the used skill materials, and the inspiring nodes, the Proposer produces a concrete edit proposal: target section, change, and rationale. When K&gt; 1 , a diversity hint instructs it to take a distinct intervention angle from prior child proposals.

Evolver ( ε ). The Evolver reads the current skill file, writes the edited version, and verifies that the mutations are consistent with the proposal summary. After each apply, the skill registry is refreshed so subsequent evaluations use the updated skill.

## B Meta-Skill Representation and Skill-Catalog Disclosure

Each meta-skill component ψ, σ, α, π, ε is stored as a SKILL.md fi le under the project skills directory (e.g., skills/meta-analyzer/SKILL.md ) and snapshotted into the SQLite node graph alongside the task-skill snapshot, so each node carries a complete, self-contained record of both task-level and meta-level state at creation. When a branch is selected for expansion, its meta-skill snapshot is restored from the node record before the five-agent pipeline runs, ensuring each branch applies its own lineage-specific improvement policy. Branches can thus diverge in their meta-level heuristics: one may have learned an aggressive edit policy for tableextraction failures while another developed conservative, incremental edits for arithmetic reasoning failures.

Skill-catalog progressive disclosure. Agents receive a compact catalog first (skill names and oneline summaries), then load the full SKILL.md only for the skill they identify as relevant, and load resource files on demand. This keeps context length manageable while preserving full expressiveness.

## C Meta-Skill Prompt Templates

Each meta-skill component is initialized as a brief Markdown document that describes its role in the evolutionary process. For example, the initial diagnosis policy ψ instructs the Analyzer to: (i) identify the primary failure class from the execution trace, (ii) distinguish between skill-addressable failures and base-model capability limits, and (iii) assign a short, specific failure tag.

## D Hyperparameter Sensitivity

Defaults. Frontier weights η 1 =1 . 0 , η 2 =0 . 5 , η 3 =0 . 25 ; meta-update horizon H =2 ; iteration budget 5 fast iterations (two meta-updates at H =2 ); child budget K ∈ [1 , K max ] with K max =3 and an initial K =2 that the Allocator adapts per step (this initial value is what the -α ablation freezes to); frontier size K F =3 , with early stopping after 5 iterations without a frontier improvement. Category-aware round-robin training draws 6 categories per batch and 3 samples per category.

Cross-branch sharing uses retrieval probability p cross =0 . 2 and same-/cross-branch inspiration limits L same =3 , L cross =2 (over-fetched at 3 × before LLMre-ranking, §3.4). Evaluation concurrency defaults to 16, raised to 128 for the QA benchmarks under the vLLM backend; it affects only throughput, not accuracy.

Sensitivity to cross-branch sharing. Sweeping the Retriever's cross-branch candidate limit confirms that moderate cross-branch retrieval consistently improves performance on both OfficeQA and SealQA; very aggressive cross-branch retrieval introduces noise by supplying irrelevant inspirations from branches working on structurally different failures.

Table 2: Meta-update horizon sweep (held-out test accuracy, %), the exact values plotted in Fig. 5. H is the number of fast task-skill iterations between consecutive meta-skill evolutions; the number of meta-updates is held fixed at three, so the iteration budget scales as 3 H . The tightest spacing H =2 is best on every benchmark; wider gaps let stale meta-rewrites overwrite productive edits.

| Benchmark   |   H =2 (6 it.) |   H =4 (12 it.) |   H =8 (24 it.) |
|-------------|----------------|-----------------|-----------------|
| OfficeQA    |          48.94 |           41.35 |           39.84 |
| SealQA      |          44.14 |           44.14 |           43.24 |
| ALFWorld    |          90.38 |           90.38 |           88.46 |

## E Full Ablation Tables

Table 3 expands the component-ablation summary of §4.3.

Table 3: Component ablation across the three benchmarks (held-out test accuracy, %), scored on the same partition as Table 1. The Evolver ( ε ) always executes; with -π it consumes the raw analysis instead of a structured proposal. -σ removes the Retriever's inspiration policy entirely; no cross-branch removes only crossbranch retrieval candidates while same-branch candidates remain available. No meta-updates freezes the slow loop, exactly reproducing the Single-Level row of Table 1.

| Configuration                            |   OfficeQA |   SealQA |   ALFWorld |
|------------------------------------------|------------|----------|------------|
| Full MetaSkill-Evolve                    |      55.32 |    45.26 |      94.23 |
| - ψ ( disable_psi )                      |      39.09 |    39.63 |      88.46 |
| - σ (no inspirations)                    |      39.09 |    40.54 |      88.46 |
| - α ( disable_alpha , K =2 )             |      35.58 |    40.54 |      88.46 |
| - π ( disable_pi )                       |      37.59 |    36.84 |      86.54 |
| No cross-branch retrieval                |      39.84 |    41.44 |      92.31 |
| No meta-updates ( disable_meta_updates ) |      48.94 |    37.21 |      92.31 |

On ALFWorld the edit-proposal policy π is again the dominant component ( 94 . 23 → 86 . 54 ), echoing SealQA: the gain hinges on the precise content of each edit rather than on how widely the search fans out. Cross-branch sharing carries the remainder of the meta-gain. Removing only cross-branch retrieval ( no cross-branch , 92 . 31 ) returns accuracy exactly to the Single-Level baseline ( 92 . 31 ), so cross-branch transfer of reusable subroutines (e.g. object-disambiguation and deferredplacement sub-skills) accounts for the whole +1 . 92 improvement there. As on the QA benchmarks, freezing the slow loop ( no meta-updates , 92 . 31 ) reproduces the Single-Level row of Table 1 exactly.

## F Evaluation Protocol: Native Held-Out Test

The accuracy figures in Table 1 are computed on a held-out test partition that the evolution loop never observes, so they measure generalisation to unseen rows rather than recall of evolved ones.

Before the loop runs, each benchmark file is stratified by its category column into three disjoint partitions: train (failure mining), val (child scoring and frontier selection), and test (the per-category remainder).

## G Train/Validation Split Ratio Sensitivity

We study how the loop's train/validation split affects the quality of the evolved task skill. All numbers in this section are recomputed on a common held-out test set per benchmark: because the stratified split is seeded, a larger-holdout run's test set is a subset of a smaller one's, so we intersect the test items across the swept runs and re-score every configuration on that shared subset (no re-running). These subsets are smaller than the benchmark sets used in the main results, so absolute numbers differ slightly.

Training-ratio sweep. Table 4 sweeps the training fraction with validation fixed at 0 . 10 (dualevolve, 5 outer iterations). More training data yields only modest gains: ALFWorld rises monotonically ( 87 . 7 → 88 . 7 → 90 . 6 ), while SealQA and OfficeQA are essentially flat within noise ( ± 1 ∼ 4 items on n ≈ 87 ∼ 107 ).

Validation-ratio sweep on ALFWorld. Table 5 enlarges the validation split ( 0 . 10 → 0 . 25 ) at each training ratio. A larger validation set consistently improves the evolved skill, by +2 . 0 , +4 . 3 , and +2 . 5 points at tr =0 . 05 / 0 . 10 / 0 . 15 respectively. This effect is specific to high-baseline benchmarks

Table 4: Training-ratio sweep (val = 0 . 10 ), accuracy (%) on the common held-out test subset.

| Benchmark   |   tr =0.05 |   tr =0.10 |   tr =0.15 |
|-------------|------------|------------|------------|
| SealQA      |      45.98 |      43.68 |      48.28 |
| OfficeQA    |      44.86 |      42.99 |      48.25 |
| ALFWorld    |      87.74 |      88.68 |      90.57 |

such as ALFWorld, where the base agent already solves most episodes out of the box: a small validation set then contains too few failures to drive evolution. The analyzer sees almost no failing trajectories, so the diagnosis → proposal → selection cycle has essentially no signal to act on and the slow loop degenerates toward a no-op. Enlarging the validation split surfaces enough failure cases to make skill proposal and frontier selection informative, which in turn yields a measurably better skill on the held-out test set. The practical implication is that on saturated benchmarks the validation split must be large enough to expose failures; otherwise the evolution loop has nothing to learn from and cannot improve.

Table 5: ALFWorld validation-ratio sweep, accuracy (%) on the common held-out test subset. Enlarging the validation split exposes more failure cases and produces a better evolved skill.

| train ratio   |   val =0.10 |   val =0.25 |
|---------------|-------------|-------------|
| tr =0.05      |       87.74 |       89.69 |
| tr =0.10      |       88.68 |       92.96 |
| tr =0.15      |       90.57 |       93.02 |