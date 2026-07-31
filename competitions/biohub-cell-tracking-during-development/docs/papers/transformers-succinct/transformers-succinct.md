> **Source:** `arx_2510.19315.pdf` · 21 pages · 0 figures · 350 display equations · 0 tables · converted by fleet `paper-md` (backend=**docling**)

## TRANSFORMERS ARE INHERENTLY SUCCINCT

Pascal Bergstr¨ aßer RPTU Kaiserslautern-Landau Kaiserslautern, Germany bergstraesser@cs.uni-kl.de

Anthony W. Lin RPTU Kaiserslautern-Landau and MPI-SWS Kaiserslautern, Germany lin@cs.uni-kl.de

Ryan Cotterell ETH Z¨ urich Zurich, Switzerland ryan.cotterell@inf.ethz.ch

## ABSTRACT

We study succinctness as a measure of the expressive power of transformers. Succinctness-how compactly a formalism can describe a language relative to other formalisms-is a classical notion in logic and automata theory. We prove that fixed-precision transformers are remarkably succinct: they can be exponentially more succinct than both linear temporal logic (LTL) and recurrent neural networks, and, by extension, state-space models, and doubly exponentially more succinct than finite automata. In other words, there exist families of languages describable by polynomial-size transformers whose smallest equivalent LTL formula or recurrent neural network is exponentially large, and whose smallest equivalent automaton is doubly exponentially large. We also establish matching upper bounds, showing that any fixed-precision transformer can be converted to an LTL formula with at most an exponential blow-up-improving a prior doubly exponential translation. As a consequence of this succinctness, we show that basic verification problems for transformers, such as emptiness and equivalence, are provably intractable: specifically, EXPSPACE -complete.

## 1 INTRODUCTION

Transformers (Vaswani et al., 2017) are the dominant architecture underlying most modern large language models. A substantial body of recent theoretical work has investigated their expressive power (Strobl et al., 2024; Barcel´ o et al., 2024; Yang et al., 2024; Hahn, 2020; P´ erez et al., 2021; Chiang and Cholak, 2022; Jerad et al., 2025), their trainability and ability to generalize to unseen strings of longer lengths (Zhou et al., 2024; Huang et al., 2025; Chiang and Cholak, 2022), and the extent to which their behavior can be formally verified (S¨ alzer et al., 2025). A key finding of this line of work is that transformers with finite precision-the setting most faithful to real-world hardware-recognize various classes of subregular languages depending on the exact assumptions made (Yang et al., 2024; Barcel´ o et al., 2024; Jerad et al., 2025; Li and Cotterell, 2025).

Subregular languages constitute strict subclasses of the regular languages. For instance, the subregular class of star-free languages are precisely those definable by regular expressions that replace the Kleene star with intersection and complementation. The language a ∗ b ∗ is star-free because it can be written as ∅ · b · a · ∅ , whereas ( aa ) ∗ is not star-free (Straubing, 1994). By contrast, recurrent neural networks (RNNs) can recognize all regular languages under a fixed precision assumption (Minsky, 1967; Siegelmann and Sontag, 1995; Merrill et al., 2020; Svete and Cotterell, 2023), making them strictly more expressive than transformers as language recognizers. However, the strong empirical performance of transformers invites the question as to whether expressive capacity is the most revealing lens through which to compare architectures.

In this paper, we propose succinctness as an alternative lens for understanding the expressivity of transformers. The succinctness of a language L with respect to a class C of language recognizers (e.g., transformers, finite automata, and formulas in FO [ &lt; ] ) measures the size of the smallest

C ∈ C that recognizes L . In other words, succinctness tells us how many symbols are needed to describe L with respect to the class C . Succinctness is a classical notion in logic and computer science (Stockmeyer, 1974; Grohe and Schweikardt, 2004), where it sharpens expressive power into a complexity-theoretic refinement: rather than asking only which languages a formalism can recognize, succinctness asks how compactly each such language can be described within it. Greater succinctness comes at a price-more succinct formalisms typically have correspondingly harder decision problems, since their compact descriptions force any decision procedure to unfold a larger amount of underlying structure. A well-known example concerns linear temporal logic (LTL; Pnueli, 1977), which is expressively equivalent to the star-free languages (Libkin, 2004), and, hence, also to the counter-free automata of McNaughton and Papert (1971). Despite this equivalence in expressive power, LTL can be exponentially more succinct than finite automata (Sistla and Clarke, 1985), i.e., certain languages admit polynomial-size LTL formulas but require exponentially larger automata. A direct consequence is that decision problems for LTL, such as checking whether a formula recognizes a trivial language, are provably harder than the corresponding problems for automata (Sistla and Clarke, 1985).

This paper offers a formal result, which can be summarized as follows: transformers can describe certain languages extremely succinctly. Specifically, we show that transformers can be exponentially more succinct than LTL and RNNs, and hence also state-space models (SSMs; Gu and Dao, 2023; Merrill et al., 2024). Moreover, they are doubly exponentially more succinct than finite automata. In concrete terms, there exist families of languages describable by polynomial-size transformers that require exponentially larger LTL formulas or RNNs, and doubly exponentially larger automata. We also establish matching upper bounds: we give a translation from finite-precision transformers to LTL formulas of exponential size, significantly improving the doubly exponential translation of Yang et al. (2024). It follows that for any fixed-precision transformer, there is an equivalent LTL formula of exponential size and an equivalent finite automaton of doubly exponential size. 1 The key technical ingredient behind these results is showing that transformers can count from 0 to 2 2 N -that is, implement doubly exponentially large counters-via a subtle encoding using attention. We then prove that the resulting languages require exponentially larger descriptions as LTL formulas or RNNs, and doubly exponentially larger descriptions as finite automata. A natural consequence of this succinctness is that analyzing transformers should be computationally challenging. And, indeed, we show that checking whether a given transformer recognizes a trivial language, is EXPSPACE -complete. Under standard complexity-theoretic assumptions, this means that no algorithm can solve the problem in less than double exponential time.

The specific transformer model we study is the unique-hard attention transformer (UHAT), a simple and widely used abstraction of self-attention (Yang et al., 2024; Jerad et al., 2025; Strobl et al., 2024; Hao et al., 2022; Li and Cotterell, 2025; Hahn, 2020; Barcel´ o et al., 2024; Bergstr¨ aßer et al., 2024). In particular, Jerad et al. (2025) show that expressivity bounds on UHATs entail corresponding bounds on softmax transformers with fixed precision. Different results in this paper hold under different precision assumptions: the UHAT upper bounds are stated for arbitrary rational weights, whereas the corresponding RNN results assume fixed (finite) precision. Importantly, this means our conclusions are valid in the setting that most faithfully mirrors real-world implementations-fixed precision arithmetic.

## 2 PRELIMINARIES

We adopt the following notational conventions in this paper. We write N def = { 1 , 2 , 3 , ... } for the natural numbers and N 0 def = { 0 , 1 , 2 , 3 , ... } for the natural numbers including zero. Given N ∈ N , we define [ N ] def = { 1 , ..., N } . Furthermore, we write Q for the rational numbers. We denote scalars by lowercase italicized Latin letters, vectors by boldface lowercase italicized Latin letters, and matrices by boldface uppercase italicized Latin letters. For a vector v = ( v 1 , ..., v D ) , we write v i : j def = ( v i , ..., v j ) for all 1 ≤ i ≤ j ≤ D , and v i for its i th component. An alphabet is a finite, nonempty set Σ of symbols . A word (also called a string ) is a finite sequence of symbols a = a 1 ... a N .

1 This holds without exception: the LTL bound is the constructive translation of Prop. 13, and the automaton bound is its composition with the standard exponential LTL-to-automaton conversion (Sistla and Clarke, 1985; Vardi and Wolper, 1994); both are unconditional upper bounds on every fixed-precision transformer.

We denote symbols using lowercase Latin letters and words as boldfaced lowercase Latin letters. We write | a | = | a 1 ... a N | = N for the length of a word a . We write Σ ∗ for the set of all words -including the empty word ε -and Σ + def = Σ ∗ \ { ε } . A language is a subset L ⊆ Σ ∗ .

We assume familiarity with basic formal language theory and complexity theory; see Kozen (1997) and Sipser (1997) for standard references. In particular, we work with finite automata and the following complexity classes (Sipser, 1997):

$$P \subseteq N P \subseteq P S P A C E \subseteq E X P \subseteq N E X P \subseteq E X P S P A C E .$$

P and NP are problems solvable by a Turing machine in polynomial and nondeterministic polynomial time, respectively, and EXP and NEXP are their exponential-time counterparts. PSPACE and EXPSPACE are problems solvable by a Turing machine in polynomial and exponential space, respectively.

## 2.1 LINEAR TEMPORAL LOGIC

A formula in linear temporal logic (LTL) over an alphabet Σ is defined by the grammar

$$\varphi \colon = \top \, | \, \bot \, | \, Q _ { a } \left ( a \in \Sigma \right ) | \, \varphi \wedge \varphi \, | \, \varphi \vee \varphi \, | \, \neg \varphi \, | \, \varphi \, \mathbf S \, \varphi \, | \, \varphi \, \mathbf U \, \varphi .$$

Satisfaction of an LTL formula φ on a word a = a 1 ... a N ∈ Σ + at position n ∈ [ N ] , written a , n | = φ , is defined inductively (omitting the trivial cases for ⊤ and ⊥ ):

$$\begin{array} { r l } { a , n \models Q _ { a } } & { i f f a _ { n } = a } & { ( a \in \Sigma ) } \\ { a , n \models \varphi _ { 1 } \wedge \varphi _ { 2 } } & { i f f a , n \models \varphi _ { 1 } \text { and } a , n \models \varphi _ { 2 } } \\ { a , n \models \varphi _ { 1 } \vee \varphi _ { 2 } } & { i f f a , n \models \varphi _ { 1 } \text { or } a , n \models \varphi _ { 2 } } \\ { a , n \models \neg \varphi _ { 1 } } & { i f f a , n \not = \varphi _ { 1 } } \\ { a , n \models \varphi _ { 1 } \subset \varphi _ { 2 } } & { i f f for some j \text { with } 1 \leq j < n \colon a , j \models \varphi _ { 2 } \text { and } } \\ & { \quad \text {for all } k \text { with } j < k < n \colon a , k \models \varphi _ { 1 } } \\ { a , n \models \varphi _ { 1 } \cup \varphi _ { 2 } } & { i f f for some j \text { with } n < j \leq N \colon a , j \models \varphi _ { 2 } \text { and } } \\ & { \quad \text {for all } k \text { with } n < k < j \colon a , k \models \varphi _ { 1 } } \end{array}$$

We also use the standard abbreviations

$$P \varphi \coloneqq \top S \varphi \quad F \varphi \coloneqq \top U \varphi \quad Y \varphi \coloneqq \bot S \varphi \quad \text {H} \varphi \coloneqq \varphi \wedge \neg P \neg \varphi .$$

An LTL formula φ recognizes the language L ( φ ) consisting of all words a ∈ Σ + where a , N | = φ . 2

Example 1. The star-free language (ab) + can be defined in LTL as

$$Q _ { b } \wedge \mathbf H ( Q _ { b } \to \mathbf Y Q _ { a } ) \wedge \mathbf H ( ( Q _ { a } \wedge \mathbf Y \top ) \to \mathbf Y Q _ { b } ) .$$

Eq. (1) asserts that the last letter is b , every b is preceded by a , and every a that has a predecessor is preceded by b .

## 2.2 UNIQUE-HARD ATTENTION TRANSFORMERS

Symbol Embedding. Let Σ be an alphabet. A symbol embedding is a function emb : Σ → Q D for some D &gt; 0 . 3 Asymbol embedding naturally extends to a homomorphism Σ ∗ → ( Q D ) ∗ , where emb (a 1 ... a N ) = emb (a 1 ) , ..., emb (a N ) for a 1 , ..., a N ∈ Σ .

2 For fragments that only allow U or F , we use a , 1 | = φ instead.

3 We define transformers over arbitrary rational numbers, as this is the most general setting in which our upper bounds hold. All results, however, carry over to fixed-precision arithmetic, i.e., a constant number of bits per value, independent of input length. The lower bounds hold under the even stronger restriction to fixedprecision integers. The precise statement of this carry-over depends on the formalization of fixed-precision arithmetic adopted: standard floating-point representations (Goldberg, 1991) are not associative, so the value of, for example, a dot product in an attention head can depend on the order in which its summands are evaluated. Our claim should therefore be understood with respect to a fixed, deterministic evaluation order; algebraic identities used in the proofs that rely on associativity (such as the order of summation) need to be re-checked under any other choice.

Attention layer. A unique hard-attention (UHA) layer of width R &gt; 0 is specified by:

- Three affine transformations: A , B : Q R → Q R and C : ( Q R × Q R ) → Q S ;
- A mask predicate M : N × N → { 0 , 1 } , defined as one of M ( n, m ) def = 1 (no masking), M ( n, m ) def = 1 [ m&lt;n ] (strict future masking), or M ( n, m ) def = 1 [ m&gt;n ] (strict past masking);
- A tie-breaking function τ that selects an element of a finite, non-empty subset of N , defined as either min (leftmost) or max (rightmost).

Given a sequence of N vectors v 1 , ..., v N ∈ Q R with N ≥ 1 , the layer operates as follows. The score function is defined as the dot product

$$S ( v _ { n } , v _ { m } ) \stackrel { \text {def} } { = } \langle A ( v _ { n } ) , B ( v _ { m } ) \rangle$$

for all n, m ∈ [ N ] . For each position n ∈ [ N ] , let

$$U _ { n } \stackrel { \text {def} } { = } \{ m \in [ N ] \ | \ M ( n , m ) = 1 \} & & ( 3 a )$$

$$B _ { n } \stackrel { \text {def} } { = } \{ m \in U _ { n } \, | \, \forall m ^ { \prime } \in U _ { n } \colon S ( v _ { n } , v _ { m } ) \geq S ( v _ { n } , v _ { m ^ { \prime } } ) \}$$

̸

be the set of unmasked positions and the subset of those that maximize the score, respectively. The attention vector at position n is defined as a n def = v τ ( B n ) if U n = ∅ and a n def = 0 otherwise. The layer outputs the sequence C ( v 1 , a 1 ) , ..., C ( v N , a N ) .

ReLU layer. A ReLU layer of width R &gt; 0 applies, for a designated coordinate r ∈ [ R ] , the ReLU function to the r th component of each input vector. Formally, define ρ r : Q R → Q R by

$$\rho _ { r } ( v ) \stackrel { \text {def} } { = } ( v _ { 1 \colon r - 1 } , \max ( 0 , v _ { r } ) , v _ { r + 1 \colon R } ) .$$

Given a sequence of N input vectors v 1 , ..., v N ∈ Q R with N ≥ 1 , the layer outputs the sequence ρ r ( v 1 ) , ..., ρ r ( v N ) obtained by applying ρ r position-wise. Equivalently, one could place a feedforward network at the end of each encoder layer (Hao et al., 2022; Barcel´ o et al., 2024).

Transformer. A unique hard-attention transformer (UHAT) is a length-preserving function T : Σ + → ( Q S ) + obtained by composing a symbol embedding with a finite sequence of UHA and ReLU layers of conformable width. To use a UHAT T : Σ + → ( Q S ) + as a language recognizer, we equip it with an acceptance vector t ∈ Q S . The language recognized by T , denoted L ( T ) , consists of all words a ∈ Σ + such that ⟨ t , v N ⟩ &gt; 0 with T ( a ) = v 1 , ..., v N ∈ Q S . 4

## 2.3 BOOLEAN RASP

As an intermediate step in proving EXPSPACE -hardness for UHATs, we use Boolean RASP (B-RASP; Yang et al., 2024), a programming language shown to be expressively equivalent to UHATs. A B-RASP program P is a finite sequence of predicates P 1 , ..., P Π ∈ { 0 , 1 } [ N ] . The program operates on an input word a = a 1 ... a N ∈ Σ + . The first | Σ | predicates are defined as follows. For each a ∈ Σ , there is a lookup function Q a ∈ { 0 , 1 } [ N ] defined by Q a ( n ) = 1 iff a n = a . We label these predicates P 1 , ..., P | Σ | . Each remaining predicate P t +1 , for t ≥ | Σ | , is built from P 1 , ..., P t by one of two operations.

- A position-wise operation sets P t +1 ( i ) := R ( i ) , where R ( i ) is a Boolean combination of P 1 ( i ) , ..., P t ( i ) .
- An attention operation sets

$$P _ { t + 1 } ( i ) \coloneqq \bullet \, _ { j } \, \left [ M ( i , j ) , S ( i , j ) \right ] V ( i , j ) \colon D ( i )$$

where ◀▶ ∈ { ◀ , ▶ } and we define the following operations

- -◀ and ▶ indicate leftmost and rightmost tie-breaking, respectively;
- -M ( i, j ) is a mask predicate as in the definition of a UHAT;
- -S( i, j ) and V ( i, j ) are Boolean combinations of P 1 ( i ) , ..., P t ( i ) and P 1 ( j ) , ..., P t ( j ) , called the score predicate and value predicate , respectively;

4 For fragments that only allow strict past masking, we use ⟨ t , v 1 ⟩ instead.

- -D ( i ) is a Boolean combination of P 1 ( i ) , ..., P t ( i ) .

The semantics of the attention operation are as follows. For each i ∈ [ N ] , let

$$o ( i ) \stackrel { \text {def} } { = } \begin{cases} \min \{ j \in [ N ] \ | \ M ( i , j ) = 1 \text { and } S ( i , j ) = 1 \} , & \text {for} \ \blacktriangleleft \\ \max \{ j \in [ N ] \ | \ M ( i , j ) = 1 \text { and } S ( i , j ) = 1 \} , & \text {for} \ \blacktriangleleft \end{cases} .$$

Then P t +1 ( i ) def = V ( i, o ( i )) if o ( i ) exists, and P t +1 ( i ) def = D ( i ) otherwise.

We can view a B-RASP program as a language recognizer by asking whether P Π ( N ) = 1 . 5

## 2.4 RECURRENT NEURAL NETWORKS

As with transformers, we treat recurrent neural networks as language acceptors, following Merrill et al. (2020) and Weiss et al. (2018; 2024). We define a recurrent neural network ( RNN ) as a quadruple (Σ , g, h 0 , f ) where Σ is an alphabet, g : ( Q D × Σ) → Q D is a transition function, h 0 ∈ Q D is an initial hidden state, and f : Q D → {⊥ , ⊤} is an acceptance function. Consider string a = a 1 ... a N . For n ≥ 1 , we define the n th hidden state h n def = g ( h n -1 , a n ) inductively. We say a is accepted iff f ( h N ) = ⊤ . As a computational model, it is natural to assume RNNs operate over a fixed precision, i.e., computation is always performed over rational numbers that can be represented with a constant k number of bits. The details of the actual representation are not important for our analysis. Therefore, the state space of the above RNN can be mapped to D -vectors over { 0 , 1 } k (instead of Q ). The following proposition is now immediate.

Proposition 1. An RNN (Σ , g, h 0 , f ) with g : ( Q D × Σ) → Q D with fixed precision k can be represented by a finite automaton with 2 kD many states.

## 2.5 SIZE MEASURES AND SUCCINCTNESS

Let R be a finite representation of a language, i.e., in our case a UHAT, LTL formula, finite automaton, RNN, or B-RASP program. We define the size of R , denoted by |R| , as the length of its minimal binary encoding. In measuring succinctness of RNN, we put the precision k in unary also as part of the size measure; since we do not want to compare a transformer that uses a fixed precision k and allow an RNN that uses a fixed precision 2 k .

Definition 2 ( f -more succinct) . Let C (1) and C (2) be classes of finite representations of languages, and let f : N → N be a function. We say C (1) is f -more succinct than C (2) if there is a family of languages { L n } ∞ n =1 together with representations R (1) n ∈ C (1) of L n such that every R (2) n ∈ C (2) representing L n satisfies |R (2) n | ≥ f ( |R (1) n | ) .

We say C (1) is exponentially more succinct than C (2) if it is f -more succinct for some f ( n ) ∈ Ω(2 cn d ) with c, d &gt; 0 , and doubly exponentially more succinct if for some f ( n ) ∈ Ω(2 2 cn d ) with c, d &gt; 0 .

Definition 3 ( g -bounded expansion) . Let C (1) and C (2) be classes of finite representations of languages, and let g : N → N be a function. We say C (1) has g -bounded expansion over C (2) if for every language L and every choice of representation R (2) ∈ C (2) of L , there is a representation R (1) ∈ C (1) of L with |R (1) | ≤ g ( |R (2) | ) .

We say C (1) has polynomially bounded expansion over C (2) if it has g -bounded expansion for some polynomial g , and exponentially bounded expansion if for some g ( n ) ∈ O (2 cn d ) with c, d &gt; 0 .

Def. 2 and Def. 3 are duals. On one hand, Def. 2 is an existential lower-bound: it asks for a witness family on which C (2) is forced to be at least f times bigger than C (1) . On the other, Def. 3 is a universal upper-bound: it asks that on every language, every C (2) representation has a C (1) translation of size at most g . Neither definition alone is antisymmetric, but together they pin the gap: C (1) is f -more succinct and has g -bounded expansion over C (2) exactly when the size gap is at least f on a witness family and at most g uniformly.

5 As for UHATs, we use P Π (1) = 1 if only strict past masking is allowed.

## 3 THE SIZE OF SMALLEST WITNESS VIA NON-EMPTINESS PROBLEM

We now consider the problem of checking whether the language recognized by a UHAT or B-RASP program is non-empty. In particular, the technique is essentially a simulation of a Turing machine with an 2 O ( N ) -sized tape for a given N . As we will see later, there are Turing machines such that the shortest accepted word by the constructed UHAT is of length at least 2 2 Ω( N ) .

Example 2. We consider an example with N = 4 . Let Σ = { 0 , 1 , # , a , b , c } , and let H def = { (a , b) , (b , c) , (b , a) , (c , b) } be a set of constraints specifying which symbols can appear in adjacent positions. We now describe a B-RASP program that accepts words of the form

$$0 0 0 a _ { 1 } \# 0 0 1 a _ { 2 } \# 0 0 1 0 a _ { 3 } \# \cdots \# 1 1 1 1 a _ { 2 ^ { 4 } } \#$$

such that (a n , a n +1 ) ∈ H for all 1 ≤ n &lt; 2 4 . We show how to construct a B-RASP programs that (i) check that the bit counter is incremented, and (ii) check that the successive symbols are in H . 6 To check (i), we use the following attention operation:

$$C _ { + 1 } ( i ) & \coloneqq \vartriangleright _ { j } [ j < i , Q _ { \# } ( j ) ] \ \bigvee _ { k = 1 } ^ { 4 } \left ( \bigwedge _ { r = 1 } ^ { k - 1 } ( - C _ { r } ( i ) \wedge C _ { r } ( j ) ) \wedge C _ { k } ( i ) \wedge - C _ { k } ( j ) \\ & \bigwedge _ { r = k + 1 } ^ { 4 } \bigwedge _ { ( C _ { r } ( i ) \leftrightarrow C _ { r } ( j ) ) } ( C _ { r } ( i ) \leftrightarrow C _ { r } ( j ) ) \right ) \colon 1 \\ \intertext { A n c h i s g a n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C _ { + 1 } ( i ) \coloneqq \vartriangleright _ { j } [ j < i , Q _ { \# } ( j ) ] } \quad \bigtriangledown \left ( \bigwedge _ { k = 1 } ^ { 4 } ( - C _ { r } ( i ) \wedge C _ { r } ( j ) ) \wedge C _ { k } ( i ) \wedge - C _ { k } ( j ) \\ & \bigwedge _ { r = k + 1 } ^ { 4 } \bigwedge _ { ( C _ { r } ( i ) \leftrightarrow C _ { r } ( j ) ) } ( C _ { r } ( i ) \leftrightarrow C _ { r } ( j ) ) \right ) \colon 1 \\ \intertext { A n c h i s g a n t i o n } \quad \intertext { A n c h i s g a n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n t i o n } \quad \intertext { C o n$$

Assume i is a # -position. Attention selects the rightmost # -position j left of position i . Let b i 1 ... b i 4 and b j 1 ... b j 4 be the bit words directly left of position i and j , respectively. We assume that we already defined C k ( i ) = b i k and C k ( j ) = b j k for all k ∈ [4] . Then, the above value predicate checks that the bit word b i 1 ... b i 4 is the bit word b j 1 ... b j 4 incremented by 1. To check (ii), we can use the attention operation

$$M _ { \leftarrow } ( i ) \coloneqq \blacktriangleright _ { j } \left [ j < i , Q _ { a } ( j ) \vee Q _ { b } ( j ) \vee Q _ { c } ( j ) \right ] \bigvee _ { ( h , h ^ { \prime } ) \in H } Q _ { h } ( j ) \wedge Q _ { h ^ { \prime } } ( i ) \colon 1 .$$

If i is a position of a symbol a i , attention picks the rightmost position j of a symbol a j to the left of i and checks with the value predicate that (a j , a i ) ∈ H . Two boundary conditions remain: the input must begin with the counter 0000 and end with the counter 1111 . The first is a position-wise check at the leftmost # , requiring C 1 ( i ) = ... = C 4 ( i ) = 0 at that position; the second is the analogous check at the rightmost # , requiring all four bits to be 1 . We omit the construction of these gadgets here, since they follow the same pattern as C +1 and M ← .

The construction given in Ex. 2 allows us to succinctly recognize a language whose shortest word has length exponential in the number of bits of the binary counter. In the following, we describe how to extend this idea such that we can reduce an EXPSPACE -complete problem to non-emptiness of a certain B-RASP program. Intuitively, we place multiple such words as above on top of each other, creating multiple rows and columns (separated by # ). Moreover, we introduce vertical constraints, i.e., between rows, in addition to the horizontal constraints H . Using this technique, we will see in Thm. 15 how B-RASP programs can even succinctly recognize languages whose shortest word has doubly exponential length.

Throughout the rest of this section, we build up to the proof of the following complexity bound.

Theorem 4. The non-emptiness problem for UHATs and B-RASP programs is EXPSPACE -complete.

To prove Thm. 4, we start with the lower bound for B-RASP programs.

Proposition 5. The non-emptiness problem for B-RASP programs is EXPSPACE -hard.

For the proof, we use the construction sketched in Ex. 2 and reduce from the tiling problem.

Problem 6. We now describe the 2 N -tiling problem . A tile is a quadruple t = ⟨ a, b, c, d ⟩ ∈ N 4 0 . We write left ( t ) = a , up ( t ) = b , right ( t ) = c , and down ( t ) = d .

6 Filling in the remainder of the B-RASP program to enforce the remaining constraints is straightforward.

Given: An instance I = ( N,T, tfi n ) , where N &gt; 0 is an integer in unary, T is a finite set of tiles, and tfi n ∈ T is a designated final tile.

Question: Do there exist a natural M ∈ N and a function τ : [2 N ] × [ M ] → T such that

1. τ (2 N , M ) = tfi n ,
2. down ( τ ( i, 1)) = up ( τ ( i, M )) = 0 for all 1 ≤ i ≤ 2 N ,
3. left ( τ (1 , j )) = right ( τ (2 N , j )) = 0 for all 1 ≤ j ≤ M ,
4. right ( τ ( i, j )) = left ( τ ( i +1 , j )) for all 1 ≤ i &lt; 2 N and 1 ≤ j ≤ M , and
5. up ( τ ( i, j )) = down ( τ ( i, j +1)) for all 1 ≤ i ≤ 2 N and 1 ≤ j &lt; M ?

A configuration of tiles, i.e., a candidate for the function τ , places tiles in 2 N columns and an arbitrary number ( M ) of rows.

Proposition 7. The 2 N -tiling problem is EXPSPACE -complete.

Proof. The result follows from Theorem 5 in Schwarzentruber (2019) by choosing k = 1 .

■

To prove Prop. 5, we construct a B-RASP program of size polynomial in N that accepts an encoding of a configuration of tiles as a sequence of words, similar to those displayed in Ex. 2, if and only if the configuration is a solution of the given 2 N -tiling problem instance. The key observation is that strict future masking with rightmost tie-breaking enables us to check conditions between successive tiles in a row (Item 4) but also between the current tile and the tile at the most recent past occurrence of the same counter value, i.e., in the same column of the previous row (Item 5). The proof of the next lemma can be found in App. A.2.

Lemma 8. Given a 2 N -tiling problem instance, one can construct in time polynomial in N a BRASP program, whose language is non-empty iff the 2 N -tiling problem instance has a solution.

Lem. 8 reduces the 2 N -tiling problem to the non-emptiness problem for B-RASP programs. Thus, together with Prop. 7, it implies Prop. 5.

We observe that the B-RASP program constructed in Lem. 8 is of a special form, which allows for a polynomial-time translation to UHAT.

Lemma 9. Given a B-RASP program P 1 , ..., P Π where every attention operation is of the form

$$P _ { t + 1 } ( i ) \coloneqq \diamondsuit \left [ M ( i , j ) , S ( j ) \wedge \bigwedge _ { k \in K } P _ { k } ( i ) \leftrightarrow P _ { k } ( j ) \right ] V ( i , j ) \colon D ( i ) ,$$

where | Σ | ≤ t &lt; Π , ◀▶ ∈ { ◀ , ▶ } , S( j ) is a Boolean combinations of P 1 ( j ) , ..., P t ( j ) , and K ⊆ { 1 , ..., t } , one can construct in polynomial time a UHAT that recognizes the same language.

Proof sketch. Any Boolean combination of position-wise predicates can be simulated by a sequence of attention layers (each layer simply applies its affine map C to the current vector and disregards the attention vector) and ReLU layers. For each B-RASP attention operation, we use a single UHA layer (§ 2.2) whose mask predicate is M , whose tie-breaker matches ◀▶ , whose affine maps A , B compute the relevant components of the score, and whose affine map C combines the position's input v n with the attention-selected a n . The value predicate V ( i, j ) is simulated as follows: the layer's affine map C copies the relevant components of a i -the layerℓ vector that attention selected from position o ( i ) (the unique position whose layerℓ vector wins the argmax in the UHA layer; equivalently, o ( i ) = τ ( B i ) in the UHAT notation of § 2.2, with B i the argmax set among the unmasked positions and τ the tie-breaker)-into the layer's output at position i , and a small ReLU sub-network applies the Boolean combination V to those copied components. The part S( j ) of the score predicate that only depends on j can be simulated using an additional preliminary layer that already computes the result of S( j ) at every position j . For the part ∧ k ∈ K P k ( i ) ↔ P k ( j ) that checks equality of two binary numbers, we provide a score function that maximizes the attention score if the two binary numbers are equal. The full proof can be found in App. A.3. ■

Proposition 10. The non-emptiness problem for UHAT is EXPSPACE -hard.

Proof. Together, Prop. 7, Lem. 8 and Lem. 9 imply the EXPSPACE lower bound for UHAT.

■

Corollary 11. The non-emptiness problem for UHATs in which every layer uses strict future masking and rightmost tie-breaking (or, dually, strict past masking and leftmost tie-breaking) is EXPSPACE -hard.

Proof. The B-RASP program constructed in Lem. 8 uses only strict future masking and rightmost tie-breaking, and it can be adapted to use only strict past masking and leftmost tie-breaking. 7 The UHAT translation in Lem. 9 preserves the mask predicate and tie-breaking. Therefore the EXPSPACE lower bound established by Prop. 7, Lem. 8, and Lem. 9 transfers to UHATs in either of the two restricted classes. ■

We now prove the upper bounds in Thm. 4. To this end, we first note that any B-RASP program can be converted in exponential time into an LTL formula using the construction given by Yang et al. (2024). In Prop. 13 we prove that the same holds true for UHATs, which improves the doubly exponential construction given by Yang et al. (2024) that translates UHATs into B-RASP programs first. These constructions suffice for the exponential-space upper bounds in Thm. 4 since nonemptiness of languages given by LTL formulas is in polynomial space (Sistla and Clarke, 1985).

To perform the translation from UHAT to LTL, we first have to make the crucial observation that the values occurring during the computation of a UHAT are not too large. The proof of the following proposition can be found in App. A.4.

Proposition 12. For every UHAT T , the precision required to evaluate T on any input is polynomial in |T | , i.e., every rational value arising in the computation of T can be represented with at most poly( |T | ) bits.

By Prop. 12, the set of rationals that can arise in the computation of T is finite, and each member has bit-length polynomial in |T | . The set therefore has cardinality at most 2 poly( |T | ) , i.e., exponential in |T | , and can be enumerated in exponential time. This is precisely what makes the layer-by-layer LTL construction in the proof of Prop. 13 feasible: at each layer we have a finite, enumerable, polynomial-bit-length set of vectors to range over, which implies that the LTL formula only has to simulate the position-wise behavior of attention layers, i.e., masking and selecting the position of the attention vector, but not the actual computation of values.

The proof of the following proposition can be found in App. A.5.

Proposition 13. Given a UHAT T recognizing a language L ⊆ Σ + , one can construct in exponential time an LTL formula φ that recognizes L .

Note, if we start with a UHAT, where every attention layer uses strict future masking and leftmost tie-breaking (resp. strict past masking and rightmost tie-breaking), then the LTL formula constructed in the proof of Prop. 13 only uses the P (resp. F ) operator. It was shown by Sistla and Clarke (1985) that the non-emptiness problem for the fragments of LTL that only allow P or F is NP -complete. Thus, we obtain an improved complexity upper bound for such restricted UHATs.

Corollary 14. The non-emptiness problem for UHATs, where each attention layer uses strict future masking/leftmost tie-breaking (resp. strict past masking/rightmost tie-breaking), is in NEXP .

Note that it has been shown by Jerad et al. (2025) that such restricted UHATs are equally expressive as the LTL fragment with only P (respectively, F ). 8 However, the construction by Jerad et al. (2025) from UHAT to the LTL fragments incurs a doubly exponential blow-up, as opposed to our singly exponential translation. The full proof of Thm. 4, combining the preceding lemmas, propositions, and corollaries into the two directions of EXPSPACE -completeness, can be found in App. A.1.

## 4 SUCCINCTNESS ACROSS REPRESENTATIONS

We now study how succinctly transformers can represent languages compared to standard models from formal language theory. We first compare transformers to LTL. One suggestion that trans-

7 In case of strict past masking, we use the first coordinate in the acceptance condition.

8 Note that LTL with a subset of the operators is often defined over EOS-padded strings; this choice affects its expressive capacity when using LTL with a subset of the operators. For instance, Li and Cotterell's (2026) demonstration that LTL[ P ] can not accept { a, b } ∗ a with EOS-padding, but can without the padding.

formers may be more succinct than LTL comes from Thm. 4, which shows that the non-emptiness problem for UHATs is EXPSPACE -complete, whereas for LTL the corresponding problem is known to be PSPACE -complete. The following result shows that this exponential gap is also manifested in terms of the formalisms' respective succinctness.

Theorem 15. UHATs are exponentially more succinct than LTL.

Proof. It suffices to exhibit a witness family { L n } ∞ n =1 together with UHATs T n of size poly( n ) recognizing L n such that every LTL formula recognizing L n has size at least c 1 2 c 2 n for constants c 1 , c 2 &gt; 0 . Such a family is a witness in the sense of Def. 2: from |T n | = poly( n ) , say |T n | ≤ c 0 n d , we get n ≥ ( |T n | /c 0 ) 1 /d , and substituting into | φ n | ≥ c 1 2 c 2 n gives | φ n | ≥ c 1 2 c ′ |T n | 1 /d for c ′ 2 = c 2 /c 1 /d 0 . This witnesses f -more succinctness for f ( m ) = c 1 2 c ′ 2 m 1 /d as required.

Polynomial-size UHAT. This direction proceeds in 3 steps.

1. Let M n be a (deterministic) Turing machine that implements a binary counter with 2 n bits, i.e., it writes 0 2 n on its tape and increments the binary number until it has written 1 2 n on its tape and accepts. In particular, M n fi rst checks that it was started with the empty tape and then writes 2 n many 0 's on its tape using an additional n -bit counter. To increment the 2 n -bit counter, M n traverses the counter from left to right while flipping every 1 to 0 until it encounters the first 0 , which is then flipped to 1 . To initialize the counter with 0 's, M n uses a linear number of states in n . Incrementing can be done with a constant-sized Turing machine. Moreover, M n uses an exponential number of tape cells in n and the unique accepting run has length at least 2 2 n .
2. Van Emde Boas (1997) gives a reduction from Turing machines to tiling problem instances that encodes configurations of Turing machines in its rows and a correct tiling corresponds to a valid execution of the Turing machine. We observe that the 2 p ( n ) -tiling problem instance I n , for some polynomial p , constructed from M n has size polynomial in n and it has the property that the smallest correct tiling has at least 2 2 n many rows.
3. Lem. 8 and Lem. 9 show that there is a UHAT T n of size polynomial in the size of I n that recognizes encodings of correct tilings of I n . Thus, T n is of size polynomial in n and the smallest accepted word has length at least 2 2 n . We let L n be the language recognized by T n .

Exponential LTL lower bound. Let φ n be an LTL formula that recognizes L n . Because the smallest accepted word by any LTL formula has length at most exponential in the formula size, using an exponential conversion from LTL to finite automata similar to Vardi and Wolper (1994), it follows that the size of φ n is at least exponential in n . ■

Conversely, the next result shows that UHATs have polynomially bounded expansion over LTL: every LTL formula has an at most polynomially larger UHAT for the same language. Combined with Thm. 15, this pins the gap between UHATs and LTL: at most polynomial universally, and at least exponential on a witness family. The proof can be found in App. A.6.

Proposition 16. UHATs have polynomially bounded expansion over LTL. In particular, given an LTL formula φ , one can construct in polynomial time a UHAT T such that L ( T ) = L ( φ ) .

We show next that UHATs are doubly exponentially more succinct than finite automata.

Theorem 17. UHATs are doubly exponentially more succinct than finite automata.

Proof. We reuse the witness family from Thm. 15: L n is the language recognized by the UHAT T n constructed there. T n is of size polynomial in n and the smallest accepted word has length at least 2 2 n . Because any automaton recognizing a non-empty language accepts a word of length at most linear in the automaton size, the smallest automaton that recognizes L n has size at least doubly exponential in n . Combined with |T n | = poly( n ) , this exhibits the witness family required by Def. 2 with f doubly exponential. ■

Conversely, the best known translation from counter-free automata-the class of finite automata equivalent to LTL-to LTL incurs an exponential blow-up (Maler and Pnueli, 1990). Composing this with Prop. 16 yields an at-most exponential-time translation from counter-free automata to

UHATs, which shows that UHATs have exponentially bounded expansion over finite automata when restricted to the star-free languages. The translation in Yang et al. (2024) also incurs an exponential blow-up via Maler and Pnueli (1990).

Combining Thm. 17 with Prop. 1 yields the following succinctness gap between UHATs and RNNs.

Corollary 18. UHATs are exponentially more succinct than RNNs.

## 5 APPLICATIONS

As a consequence of our results, we can show that reasoning about the language accepted by a UHAT, e.g., checking equivalence and emptiness, is intractable. Contrast this fact with deterministic finite automata, where these problems can be done in polynomial time (Kozen, 1997). As an example, we give a precise statement about the complexity of equivalence problem , i.e., the problem of checking whether two UHATs recognize the same language. The proof can be found in App. A.7.

Theorem 19. Deciding the equivalence between two UHATs is EXPSPACE -complete.

## 6 CONCLUDING REMARKS

Related work. Our work directly draws upon a number of recent results (Yang et al., 2024; Barcel´ o et al., 2024; Jerad et al., 2025; Li and Cotterell, 2025), which demonstrate the close connection between unique-hard attention transformers and LTL and, thus, the star-free regular languages. However, none of these results concerns succinctness and computational complexity of verification. Closer to our complexity-theoretic angle, S¨ alzer et al. (2025) studied the verification problem for transformers of various precisions and showed that fixed-precision transformers are at least NEXP -hard (i.e., hard for the class of problems solvable by nondeterministic algorithms that run in exponential time). Their technique implies that transformers can be (singly) exponentially more succinct than finite automata, but yields no conclusion about their succinctness relative to representations like LTL or RNNs. Our results substantially improve on this by showing that transformers can be doubly exponentially more succinct than automata, and exponentially more succinct than LTL and RNNs. Our model is also simpler: we use unique-hard attention, whereas S¨ alzer et al. (2025) employs a combination of soft and hard attention. Our setting also restricts positional information to positional masking-a simple class of positional embeddings also considered by Yang et al. (2024), Jerad et al. (2025) and Li and Cotterell (2025)-in contrast to the arbitrary fixed-precision positional encodings admitted by S¨ alzer et al. (2025). Without position encodings, S¨ alzer et al. (2026) recently showed that verification is undecidable for average-hard-attention and softmax-attention transformers with finite but unbounded precision.

Formal Verification of Transformers. We close by situating our findings within the broader program of formal verification-the automated analysis, verification, and explanation of transformerswhich is a central concern of explainable AI (Huang et al., 2020). Substantial practical progress has been made on verifying feed-forward neural networks, with tools developed over the last decade and benchmarked at the annual VNN competition (Brix et al., 2024); transformers, by contrast, remain largely out of reach. Despite the high worst-case complexity ( EXPSPACE -complete), we pose as a challenge bringing techniques from automated verification (Clarke et al., 2018)-symbolic methods, simulation, inter alia -to bear on transformer verification in practice. Because our EXPSPACE -hardness proof requires transformers that encode large counters, a complementary direction is to identify subclasses that cannot encode such counters and thus admit lower-complexity verification. A related open question is the learnability of succinct transformers, on which the empirical evidence remains mixed (Garg et al., 2022; Naim et al., 2025; Huang et al., 2025). Finally, our results are a first step toward understanding how succinct transformers can be relative to other languageacceptor models, e.g., fixed-precision softmax transformers (Li and Cotterell, 2025), which UHATs overapproximate. We leave the succinctness of fixed-precision softmax and average-hard attention transformers as future work; see Yang et al. (2026) for an initial attempt.

## ACKNOWLEDGMENTS

We thank David Chiang, Marco S¨ alzer, Andy Yang and anonymous reviewers for their helpful feedback. Pascal Bergstr¨ aßer and Anthony W. Lin are supported by Deutsche Forschungsgemeinschaft (grant number 522843867) and European Union 9 (ERC, LASD, 101089343).

## REFERENCES

- Pablo Barcel´ o, Alexander Kozachinskiy, Anthony Widjaja Lin, and Vladimir V. Podolskii. 2024. Logical languages accepted by transformer encoders with hard attention. In International Conference on Learning Representations .
- Pascal Bergstr¨ aßer, Chris K¨ ocher, Anthony Widjaja Lin, and Georg Zetzsche. 2024. The power of hard attention transformers on data sequences: A formal language theoretic perspective. In Advances in Neural Information Processing Systems .
- [Christopher Brix, Stanley Bak, Taylor T. Johnson, and Haoze Wu. 2024. The fifth international verification of neural networks competition (VNN-COMP 2024): Summary and results.](https://arxiv.org/abs/2412.19985)
- David Chiang and Peter Cholak. 2022. Overcoming a theoretical limitation of self-attention. In Annual Meeting of the Association for Computational Linguistics .
- E.M. Clarke, O. Grumberg, D. Kroening, D. Peled, and H. Veith. 2018. Model Checking , 2 edition. MIT Press.
- Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn in-context? A case study of simple function classes. In Advances in Neural Information Processing Systems .
- David Goldberg. 1991. What every computer scientist should know about floating-point arithmetic. ACM Computing Surveys , 23(1).
- Martin Grohe and Nicole Schweikardt. 2004. The succinctness of first-order logic on linear orders. In IEEE Symposium on Logic in Computer Science .
- [Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces.](https://arxiv.org/abs/2312.00752)
- Michael Hahn. 2020. Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics , 8.
- Yiding Hao, Dana Angluin, and Robert Frank. 2022. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics , 10.
- Xiaowei Huang, Daniel Kroening, Wenjie Ruan, James Sharp, Youcheng Sun, Emese Thamo, Min Wu, and Xinping Yi. 2020. A survey of safety and trustworthiness of deep neural networks: Verification, testing, adversarial attack and defence, and interpretability. Computer Science Review , 37.
- Xinting Huang, Andy Yang, Satwik Bhattamishra, Yash Raj Sarrof, Andreas Krebs, Hattie Zhou, Preetum Nakkiran, and Michael Hahn. 2025. A formal framework for understanding length generalization in transformers. In International Conference on Learning Representations .
- Selim Jerad, Anej Svete, Jiaoda Li, and Ryan Cotterell. 2025. Unique hard attention: A tale of two sides. In Annual Meeting of the Association for Computational Linguistics .
- Dexter Kozen. 1997. Automata and Computability . Springer.
- Jiaoda Li and Ryan Cotterell. 2025. Characterizing the expressivity of fixed-precision transformer language models. In Advances in Neural Information Processing Systems .

9 Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

- Jiaoda Li and Ryan Cotterell. 2026. Characterizing the expressivity of local attention in transformers. In Annual Meeting of the Association for Computational Linguistics .
- [Leonid Libkin. 2004. Elements of Finite Model Theory . Springer.](https://doi.org/10.1007/978-3-662-07003-1)
- Oded Maler and Amir Pnueli. 1990. Tight bounds on the complexity of cascaded decomposition of automata. In Symposium on Foundations of Computer Science .
- Robert McNaughton and Seymour Papert. 1971. Counter-Free Automata . MIT Press.
- William Merrill, Jackson Petty, and Ashish Sabharwal. 2024. The illusion of state in state-space models. In International Conference on Machine Learning .
- William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020. A formal hierarchy of RNN architectures. In Annual Meeting of the Association for Computational Linguistics .
- [Marvin L. Minsky. 1967. Computation: Finite and Infinite Machines . Prentice-Hall.](https://archive.org/details/computationfinit0000mins)
- Omar Naim, Jerome Bolte, and Nicholas Asher. 2025. Analyzing limits for in-context learning. In What Can('t) Transformers Do? Workshop at the Conference on Neural Information Processing Systems .
- Jorge P´ erez, Pablo Barcel´ o, and Javier Marinkovic. 2021. Attention is Turing-complete. Journal of Machine Learning Research , 22.
- Amir Pnueli. 1977. The temporal logic of programs. In Symposium on Foundations of Computer Science .
- Marco S¨ alzer, Eric Alsmann, and Martin Lange. 2025. Transformer encoder satisfiability: Complexity and impact on formal reasoning. In International Conference on Learning Representations .
- Marco S¨ alzer, Chris K¨ ocher, Alexander Kozachinskiy, Georg Zetzsche, and Anthony Widjaja Lin. 2026. The counting power of transformers. In International Conference on Learning Representations .
- Franc ¸ois Schwarzentruber. 2019. The complexity of tiling problems. arXiv preprint arXiv:1907.00102.
- Hava T. Siegelmann and Eduardo D. Sontag. 1995. On the computational power of neural nets. Journal of Computer and System Sciences , 50(1).
- [Michael Sipser. 1997. Introduction to the Theory of Computation . PWS Publishing.](https://archive.org/details/introductiontoth00sips)
- A. Prasad Sistla and Edmund M. Clarke. 1985. The complexity of propositional linear temporal logics. Journal of the ACM , 32(3).
- Larry Joseph Stockmeyer. 1974. The Complexity of Decision Problems in Automata Theory and Logic . Ph.D. thesis, Massachusetts Institute of Technology.
- [Howard Straubing. 1994. Finite Automata, Formal Logic, and Circuit Complexity . Birkh¨ auser.](https://doi.org/10.1007/978-1-4612-0289-9)
- Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. 2024. What formal languages can transformers express? A survey. Transactions of the Association for Computational Linguistics , 12.
- Anej Svete and Ryan Cotterell. 2023. Recurrent neural language models as probabilistic finite-state automata. In Conference on Empirical Methods in Natural Language Processing .
- Peter van Emde Boas. 1997. The convenience of tilings. In Complexity, Logic, and Recursion Theory . CRC Press.
- Moshe Y. Vardi and Pierre Wolper. 1994. Reasoning about infinite computations. Information and Computation , 115(1).

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems .
- Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On the practical computational power of finite precision RNNs for language recognition. In Annual Meeting of the Association for Computational Linguistics .
- [Gail Weiss, Yoav Goldberg, and Eran Yahav. 2024. Extracting automata from recurrent neural networks using queries and counterexamples (extended version). Machine Learning , 113(5).](https://doi.org/10.1007/s10994-022-06163-2)
- Andy Yang, Pascal Bergstr¨ aßer, Georg Zetzsche, David Chiang, and Anthony Widjaja Lin. 2026. Length generalization bounds for transformers. Preprint , arXiv:2603.02238. (Accepted at ICML'26).
- Andy Yang, David Chiang, and Dana Angluin. 2024. Masked hard-attention transformers recognize exactly the star-free languages. In Advances in Neural Information Processing Systems .
- Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Joshua M. Susskind, Samy Bengio, and Preetum Nakkiran. 2024. What algorithms can transformers learn? A study in length generalization. In International Conference on Learning Representations .

## A PROOFS

A.1 PROOF OF THEOREM 4

Theorem 4. The non-emptiness problem for UHATs and B-RASP programs is EXPSPACE -complete.

Proof. The theorem asserts EXPSPACE -completeness of the non-emptiness problem for two formalisms simultaneously: B-RASP programs and UHATs. We prove the two directions of completeness separately.

Hardness (lower bound). We establish EXPSPACE -hardness first for B-RASP programs, then transfer it to UHATs.

- B-RASP . For B-RASP programs, Prop. 7 states that the 2 N -tiling problem is EXPSPACE -complete, so it is EXPSPACE -hard. By Lem. 8, any instance of the 2 N -tiling problem can be transformed, in time polynomial in N , into a B-RASP program whose recognized language is non-empty if and only if the original tiling instance has a solution. Composing these two results yields a polynomial-time reduction from an EXPSPACE -hard problem to B-RASP nonemptiness. Thus, B-RASP non-emptiness is itself EXPSPACE -hard.
- UHAT . For UHATs, we further compose the previous reduction with the polynomial-time, language-preserving translation from B-RASP to UHATs of Lem. 9. The composed reduction is again polynomial-time, so the EXPSPACE -hardness of 2 N -tiling transfers to UHAT nonemptiness; this is the content of Prop. 10.

Membership (upper bound). We show that the non-emptiness problem for both formalisms lies in EXPSPACE by translating to LTL and invoking the Sistla-Clarke decision procedure.

For B-RASP programs, the construction of Yang et al. (2024) converts any B-RASP program P into an equivalent LTL formula φ P in time exponential in | P | ; consequently | φ P | is itself at most exponential in | P | .

For UHATs, the analogous translation is supplied by Prop. 13, which turns any UHAT T into an equivalent LTL formula φ T in exponential time (and hence of at most exponential size). The proof of Prop. 13 relies crucially on Prop. 12: the latter guarantees that all rational values arising in the computation of T admit representations of polynomially many bits, which is what makes the set of layer-wise representations enumerable in exponential time.

In both cases we have, in exponential time, reduced the original non-emptiness problem to LTL satisfiability for a formula of size at most exponential in the size of the original UHAT or B-RASP program. Sistla and Clarke (1985) prove that LTL satisfiability is decidable in space polynomial in the formula size; applied to φ P or φ T , this uses space polynomial in an exponential quantity, that is, space exponential in | P | or |T | . The overall decision procedure therefore runs in EXPSPACE .

Combining the two halves yields EXPSPACE -completeness of non-emptiness for both B-RASP programs and UHATs, as claimed. ■

## A.2 PROOF OF LEMMA 8

Lemma 8. Given a 2 N -tiling problem instance, one can construct in time polynomial in N a BRASP program, whose language is non-empty iff the 2 N -tiling problem instance has a solution.

̸

Proof. Fix a 2 N -tiling problem instance I = ( N,T, tfi n ) as in Prob. 6. We construct a B-RASP program P I of size polynomial in N and | T | such that L ( P I ) = ∅ if and only if I admits a solution τ : [2 N ] × [ M ] → T in the sense of Prob. 6.

Encoding. We encode a candidate τ as a word over the alphabet Σ := T ∪ { 0 , 1 , # } . Define the per-cell encoding enc : [2 N ] × [ M ] → Σ + by

$$\text {enc} ( i , j ) \coloneqq \text {bin} _ { N } ( i - 1 ) \, \tau ( i , j ) \, \# ,$$

where bin N ( i -1) ∈ { 0 , 1 } N is the N -bit binary representation of i -1 , most significant bit first. The encoding of the full configuration scans columns within each row in order:

$$\ e n c ( \tau ) \coloneqq \text {enc} ( 1 , 1 ) \cdots \text {enc} ( 2 ^ { N } , 1 ) \text { enc} ( 1 , 2 ) \cdots \text {enc} ( 2 ^ { N } , M ) .$$

Let L I := { enc ( τ ) | τ is a solution of I} . It suffices to construct a B-RASP program that recognizes L I .

Plan. Throughout, let a = a 1 ... a L ∈ Σ + denote the input, with L := | a | . We build P I as a conjunction of five gadgets , each evaluated at the last position L :

- Gadget A-shape : a ∈ ( { 0 , 1 } N T #) ∗ .
- Gadget B-column counter : consecutive N -bit blocks count 0 , 1 , ..., 2 N -1 , 0 , ... .
- Gadget C-final tile : a ends with 1 N tfi n # (Item 1).
- Gadget D-boundary conditions : Item 2 and Item 3.
- Gadget E-adjacency : Item 4 and Item 5.

The output predicate is Y ( i ) := A ( i ) ∧ B ( i ) ∧ C ( i ) ∧ D ( i ) ∧ E ( i ) , and P I accepts a iff Y ( L ) = 1 . We define each gadget below and verify its soundness at L .

Gadget A: well-formed shape. To check whether a ∈ ( { 0 , 1 } N T #) ∗ , we construct the following B-RASP predicates:

$$A _ { T } ( i ) \coloneqq \blacktriangleright _ { j } \left [ j < i , 1 \right ] \, \bigvee _ { t \in T } Q _ { t } ( j ) \colon 0$$

$$A _ { b i t , 1 } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , 1 \right ] Q _ { 0 } ( j ) \vee Q _ { 1 } ( j ) \colon 0$$

$$A _ { b i t , k } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , 1 \right ] A _ { b i t , k = 1 } ( j ) \colon 0 \quad \text {for} \, k = 2 , \dots , N$$

A

#

,

1

(

i

) :=

▶

j

[

j &lt; i,

1]

Q

#

(

j

) : 1

(12d)

$$A _ { \# , k } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , 1 \right ] A _ { \# , k - 1 } ( j ) \colon 1 \quad \text {for } k = 2 , \dots , N + 1$$

$$A _ { \text {enc} } ( i ) \coloneqq ( Q _ { \# } ( i ) \rightarrow A _ { T } ( i ) ) \wedge \left ( \left ( \bigvee _ { t \in T } Q _ { t } ( i ) \right ) \rightarrow \left ( \bigwedge _ { k = 1 } ^ { N } A _ { b i t , k } ( i ) \right ) \wedge A _ { \# , N + 1 } ( i ) \right ) \ \ ( 1 2 f )$$

We aggregate A enc across all positions:

$$A ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , \neg A _ { \text {enc} } ( j ) \right ] 0 \colon A _ { \text {enc} } ( i ) .$$

Then A ( L ) = 1 iff A enc ( i ) = 1 at every position, which holds iff a matches ( { 0 , 1 } N T #) ∗ up to the requirement that the last symbol is # , which is enforced by Gadget C below.

Gadget B: column counter. We check that the N -bit binary blocks separated by # encode the integers 0 , 1 , ..., 2 N -1 , 0 , ... in order, generalizing the increment predicate of Ex. 2 from 4 bits to N bits.

$$B _ { 1 } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { 0 } ( j ) \lor Q _ { 1 } ( j ) \right ] Q _ { 1 } ( j ) \colon 0$$

$$B _ { k } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { 0 } ( j ) \vee Q _ { 1 } ( j ) \right ] B _ { k - 1 } ( j ) \colon 0 \quad \text {for } k = 2 , \dots , N$$

$$B _ { + 1 } ( i ) \coloneqq \blacktriangleright _ { j } \left [ j < i , Q _ { \# } ( j ) \right ] \, \bigvee _ { k = 1 } ^ { N } \left ( \bigwedge _ { r = 1 } ^ { k - 1 } \neg B _ { r } ( i ) \wedge B _ { r } ( j ) \right )$$

$$^ { - } )$$

$$\wedge B _ { k } ( i ) \wedge \neg B _ { k } ( j ) \wedge \bigwedge _ { r = k + 1 } ^ { N } B _ { r } ( i ) \leftrightarrow B _ { r } ( j ) \right ) \colon 0$$

$$B _ { 1 \rightarrow 0 } ( i ) \coloneqq \blacktriangleright _ { j } \left [ j < i , Q _ { \# } ( j ) \right ] \bigcap _ { k = 1 } ^ { N } \neg B _ { k } ( i ) \wedge B _ { k } ( j ) \colon \bigtriangleup _ { k } ^ { N } \neg B _ { k } ( i ) \\$$

$$B ( i ) \coloneqq \mathbf \triangleright _ { j } | j < i , Q _ { \# } ( j ) \wedge \to B _ { 1 \to 0 } ( j ) \wedge \to B _ { + 1 } ( j ) ] \ 0 \colon B _ { 1 \to 0 } ( i ) \wedge B _ { + 1 } ( i ) \quad$$

Then B ( L ) = 1 iff the binary blocks count up correctly.

Gadget C: final tile. For each tile t ∈ T , we record whether the most recent prior tile equals t via the predicate T t at every position, and use this to check that a ends with 1 N tfi n # (Item 1):

$$T _ { t } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , \bigvee _ { t ^ { \prime } \in T } Q _ { t ^ { \prime } } ( j ) \right ] Q _ { t } ( j ) \colon 0 \quad \text {for all } t \in T$$

$$C ( i ) \coloneqq Q _ { \# } ( i ) \wedge T _ { t _ { f _ { n } } } ( i ) \wedge \bigwedge _ { k = 1 } ^ { N } B _ { k } ( i )$$

Then C ( L ) = 1 iff a ends with 1 N tfi n # .

Gadget D: boundary conditions. We enforce Item 2 and Item 3 of Prob. 6: the bottom row uses tiles with down = 0 , the top row uses tiles with up = 0 , and the leftmost (resp. rightmost) column uses tiles with left = 0 (resp. right = 0 ).

$$D _ { \perp } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { \# } ( j ) \wedge \bigwedge _ { k = 1 } ^ { N } B _ { k } ( i ) \leftrightarrow B _ { k } ( j ) \right ] 1 \colon \bigvee _ { \substack { t \in T , \\ \text {down} ( t ) = 0 } }$$

̸

$$k = & 1 \quad \text {down} ( t ) = 0 \\ & \quad \text {down} ( t ) = 0 \\ D _ { \top } ( i ) \coloneqq & \cdot \cdot \cdot \left [ j < i , Q _ { \# } ( j ) \wedge \left ( \bigvee _ { t \in T , } ( j ) \vee \left ( \bigwedge _ { k = 1 } ^ { N } \neg B _ { k } ( j ) \right ) \right ) \right ] \\ & \quad \ \left ( \sup _ { \substack { t \in T , \\ \quad \intertext { D _ { \top } ( i ) \coloneqq \cdot \cdot \cdot } } } ^ { N } \left ( j \leq i , Q _ { \# } ( j ) \wedge \left ( \bigvee _ { t \in T , } ( j ) \vee \left ( \bigvee _ { \substack { t \in T , \\ \quad \intertext { \Delta } } } ^ { N } \neg B _ { k } ( j ) \right ) \right ) \right ] \\ & \quad \ \left ( \sup _ { \substack { t \in T , \\ \quad \intertext { \Delta } } } ^ { N } \left ( j \neq i \right ) \right ) \\ & \quad \ \left ( \bigvee _ { \substack { t \in T , \\ \quad \intertext { \Delta } } } ^ { N } ( j ) \bigwedge \left ( \bigvee _ { \substack { t \in T , \\ \quad \intertext { T } } } ( i ) \right ) \colon 0 \\ & \quad \ \left ( \sup _ { t \in T , } ( i ) = 0 \\ & \quad \ \left ( j ( t ) = 0 \right )$$

$$D _ { + } ( i ) \coloneqq \left ( \bigwedge _ { k = 1 } ^ { N } \neg B _ { k } ( i ) \right ) \to \bigvee _ { \substack { t \in T , \\ \text {left} ( t ) = 0 } } ( 1 )$$

$$D _ { + } ( i ) & \coloneqq \left ( \bigwedge _ { k = 1 } ^ { N } B _ { k } ( i ) \right ) \to \bigvee _ { t \in T , } T _ { t } ( i ) \\ \\ & \quad \ \ ( 1 6 d )$$

$$D ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { \# } ( j ) \wedge \neg ( D _ { \perp } ( j ) \wedge D _ { \neg } ( j ) \wedge D _ { \neg } ( j ) ) \right ] 0 \colon D _ { \perp } ( i ) \wedge D _ { \top } ( i ) \wedge D _ { \neg } ( i ) \wedge D _ { \neg } ( i ) \\$$

Then D ( L ) = 1 iff Item 2 and Item 3 hold.

Gadget E: adjacency. We enforce Item 4 and Item 5: horizontally adjacent tiles agree on ( right , left ) , and vertically adjacent tiles (same column, consecutive rows) agree on ( down , up ) as follows:

$$E _ { \leftarrow } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { \# } ( j ) \right ] \left ( \bigvee _ { k = 1 } ^ { N } B _ { k } ( i ) \right ) \rightarrow \bigvee _ { \substack { t , t ^ { \prime } \in T , \\ \left | t \right | = \text {right} ( t ^ { \prime } ) } } \left ( T _ { t } ( i ) \wedge T _ { t ^ { \prime } } ( j ) \right ) \colon 1 \\$$

$$E _ { \downarrow } ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , Q _ { \# } ( j ) \wedge \bigwedge _ { k = 1 } ^ { N } B _ { k } ( i ) \leftrightarrow B _ { k } ( j ) \right ] \bigvee T _ { t } ( i ) \wedge T _ { t ^ { \prime } } ( j ) \colon 1 \\ \underset { \text {down} ( t ) = \text {up} ( t ^ { \prime } ) } { \dots } \left ( 1 \right ) \\$$

$$E ( i ) \coloneqq \blacktriangledown _ { j } \left [ j < i , \, Q _ { \# } ( j ) \wedge \neg ( E _ { \downarrow } ( j ) \wedge E _ { \leftarrow } ( j ) ) \right ] 0 \colon E _ { \downarrow } ( i ) \wedge E _ { \leftarrow } ( i )$$

Then E ( L ) = 1 iff Item 4 and Item 5 hold.

Wrap-up. Define the output predicate

$$Y ( i ) \coloneqq A ( i ) \wedge B ( i ) \wedge C ( i ) \wedge D ( i ) \wedge E ( i ) ,$$

̸

and let P I accept iff Y ( L ) = 1 . By the soundness of each gadget, L ( P I ) = L I , so L ( P I ) = ∅ iff I admits a solution. Each gadget uses O ( N ) predicates of size polynomial in N and | T | , so | P I | is polynomial in N and | T | , hence in the size of I . ■

A.3 PROOF OF LEMMA 9

Lemma 9. Given a B-RASP program P 1 , ..., P Π where every attention operation is of the form

$$P _ { t + 1 } ( i ) \coloneqq \diamondsuit \left [ M ( i , j ) , S ( j ) \wedge \bigwedge _ { k \in K } P _ { k } ( i ) \leftrightarrow P _ { k } ( j ) \right ] V ( i , j ) \colon D ( i ) ,$$

where | Σ | ≤ t &lt; Π , ◀▶ ∈ { ◀ , ▶ } , S( j ) is a Boolean combinations of P 1 ( j ) , ..., P t ( j ) , and K ⊆ { 1 , ..., t } , one can construct in polynomial time a UHAT that recognizes the same language.

Proof. Let P = ( P 1 , ..., P Π ) be a B-RASP program over the alphabet Σ = { a 1 , ..., a | Σ | } , where P t is the initial vector Q a t for all 1 ≤ t ≤ | Σ | . We construct a UHAT over Σ that recognizes the same language as P . We use a one-hot symbol embedding emb : Σ →{ 0 , 1 } | Σ | , i.e., emb (a t ) := e t for all 1 ≤ t ≤ | Σ | , where e t denotes the t th unit vector. Then P t ( i ) coincides with the t th component of the i th input vector of the UHAT after the symbol embedding is applied. The UHAT will preserve these components in each layer and will gradually add new components to store the value of P t ( i ) for all | Σ | &lt; t ≤ Π . So assume we already defined the layers of the UHAT that compute the vector ( P 1 ( i ) , ..., P t ( i )) at position i for | Σ | ≤ t &lt; Π . We now define additional layers whose output will be ( P 1 ( i ) , ..., P t +1 ( i )) .

We first consider the case where P t +1 is a position-wise operation, i.e., P t +1 ( i ) is defined by a Boolean combination of P 1 ( i ) , ..., P t ( i ) . We define UHAT layers to compute the result of that Boolean combination bottom-up. Assume we already defined layers that output ( R 1 ( i ) , ..., R s ( i )) , where s ≥ t and R 1 ( i ) , ..., R s ( i ) contain P 1 ( i ) , ..., P t ( i ) and the results of previously computed subformulas. To compute the result of ¬ R k ( i ) for some k ∈ [ s ] , we add an attention layer that just forwards 1 -R k ( i ) at position i in an additional component while leaving the first s components unchanged. To compute R k ( i ) ∧ R ℓ ( i ) for some k, ℓ ∈ [ s ] , we first use an attention layer to forward R k ( i ) + R ℓ ( i ) -1 in an additional component followed by a ReLU layer that forwards the result of max { 0 , R k ( i ) + R ℓ ( i ) -1 } in this additional component, again leaving the first s components unchanged. We do not have to deal with R k ( i ) ∨ R ℓ ( i ) , since it can be rewritten as ¬ ( ¬ R k ( i ) ∧ ¬ ( R ℓ ( i )) . After computing the results of all subformulas, we add an additional attention layer to only forward ( P 1 ( i ) , ..., P t +1 ( i )) , i.e., removing the intermediate results. Observe that a Boolean combination has only linearly many subformulas.

Let us now consider the case where P t +1 is an attention operation of the form

$$P _ { t + 1 } ( i ) \coloneqq \diamondsuit \left [ M ( i , j ) , S ( j ) \wedge \bigwedge _ { k \in K } P _ { k } ( i ) \leftrightarrow P _ { k } ( j ) \right ] V ( i , j ) \colon D ( i )$$

as in the statement of Lem. 9. Throughout the construction below, K , M , ◀▶ , S , V , and D are the parameters of this input attention operation, as bound in the lemma statement; in particular, K ⊆ { 1 , ..., t } is the index set of predicates whose values at i and j are compared for equality in the operation's score predicate. We first use additional layers as in the case of position-wise operations to compute the result of ¬ S( i ) at every position i in an additional component to output ( P 1 ( i ) , ..., P t ( i ) , 1 -S( i )) . Next we use an attention layer to add the result of 1 -P k ( i ) for all k ∈ K at every position i in additional components. We then add an attention layer that uses mask predicate M , tie-breaking according to ◀▶ , and attention score

$$\left ( \sum _ { k \in K } \left ( P _ { k } ( i ) P _ { k } ( j ) + ( 1 - P _ { k } ( i ) ) ( 1 - P _ { k } ( j ) ) \right ) \right ) - \left ( 1 - S ( j ) \right )$$

which is equal to |{ k ∈ K | P k ( i ) = P k ( j ) }| -(1 -S( j )) since

$$P _ { k } ( i ) P _ { k } ( j ) + ( 1 - P _ { k } ( i ) ) ( 1 - P _ { k } ( j ) ) = \begin{cases} 1 , & \text {if } P _ { k } ( i ) = P _ { k } ( j ) \\ 0 , & \text {otherwise} . \end{cases}$$

Thus, the score is maximized (equal to | K | ) if P k ( i ) = P k ( j ) for all k ∈ K and S( j ) = 1 . For every position i let o ( i ) be the position of the vector that maximizes the attention score with respect to i .

The attention layer forwards the vector ( P 1 ( i ) , ..., P t ( i ) , P 1 ( o ( i )) , ..., P t ( o ( i )) , S( o ( i ))) at position i . We now compute the result of

$$R ( i ) \coloneqq S ( o ( i ) ) \wedge \bigwedge _ { k \in K } P _ { k } ( i ) \leftrightarrow P _ { k } ( o ( i ) )$$

at every position i as in the case of position-wise operations and forward the vector ( P 1 ( i ) , ..., P t ( i ) , P 1 ( o ( i )) , ..., P t ( o ( i )) , R ( i )) . Finally, we compute

$$\left ( R ( i ) \wedge V ( i , o ( i ) ) \right ) \vee \left ( \neg R ( i ) \wedge D ( i ) \right )$$

by again using additional layers as in the case of position-wise operations, whose result is exactly P t +1 ( i ) . We then forward ( P 1 ( i ) , ..., P t +1 ( i )) .

It remains to describe when the UHAT accepts. If P t is the output vector of P , then we stop the construction of the UHAT after the layers to compute P t are constructed. The acceptance vector of the UHAT is then defined as e t , i.e., the t th unit vector. This means that the UHAT accepts if and only if ⟨ e t , ( P 1 ( N ) , ..., P t ( N )) ⟩ &gt; 0 , which holds if and only if P t ( N ) = 1 , where N is the length of the input.

We observe that the resulting UHAT only has polynomially many layers since the result of each operation P t can be computed using an additional number of layers that is linear in the description size of P t . ■

## A.4 PROOF OF PROPOSITION 12

Proposition 12. For every UHAT T , the precision required to evaluate T on any input is polynomial in |T | , i.e., every rational value arising in the computation of T can be represented with at most poly( |T | ) bits.

Proof. Part 1: bounding the denominator. Let K be the number of rationals in the description of T , i.e., the entries of the embedding emb (a) for a ∈ Σ , the entries of every affine transformation A ℓ , B ℓ , C ℓ at each layer ℓ , and the entries of the acceptance vector t . Each of these rationals is part of T 's binary encoding, so K ≤ |T | and the bit-length b of any individual rational satisfies b ≤ |T | as well. Let D be the least common multiple (LCM) of the denominators of those K rationals. The LCM of K integers each of bit-length at most b has bit-length at most K · b (an upper bound is the product of the integers), so D has bit-length at most |T | 2 . By construction, every embedding entry and every coefficient of an affine transformation can be written with denominator dividing D .

We now show by induction on the layer number ℓ that there is a denominator d ℓ , of polynomial bitlength, common to every value at layer ℓ . The base case ℓ = 0 is the embedding: d 0 def = D works. For the inductive step, suppose every layerℓ value has denominator dividing d ℓ . An attention layer takes an affine combination of layerℓ values, weighted by coefficients drawn from the description of T . Each weight has denominator dividing D , and each input has denominator dividing d ℓ . The product of two fractions with denominators dividing D and d ℓ has denominator dividing D · d ℓ ; a sum of such products and the bias term (denominator dividing D , which divides D · d ℓ ) shares the same denominator. So we may take d ℓ +1 def = D · d ℓ . A ReLU layer applies max { 0 , ·} pointwise, which leaves denominators unchanged, so d ℓ +1 def = d ℓ suffices for that case. Iterating over at most |T | many layers, the final denominator is at most D |T | +1 , with bit-length ( |T | +1)log D , polynomial in |T | .

Part 2: bounding the numerator. At an attention layer of width R , each output numerator is an affine combination of 2 R layerℓ numerators with integer weights and a bias each of bit-length at most log D (the entries of C scaled by D ). The magnitude is therefore at most 2 R · 2 log D times the largest layerℓ numerator's magnitude-an additive per-layer bit-length increase of log R + log D + O (1) ; ReLU leaves bit-lengths unchanged. Since layer0 numerators have bit-length at most |T | +log D , iterating across the at-most|T | many layers gives a final bit-length of O ( |T | · (log R +log D )) , polynomial in |T | .

Note that the additive (rather than multiplicative) per-layer growth above follows because the forwarded vector at an attention layer is C ( v n , a n ) , an affine combination of two layerℓ vectors:

the position's own input v n and the attention-selected a n = v τ ( B n ) , which is just a verbatim copy of whichever existing layerℓ vector τ picks. The score function S( v n , · ) -including the dot product-does enter the picture and a single score value would have bit-length quadratic in its inputs. But the score is used only to rank positions; the argmax reduces it to a position index, and that index alone is used to fetch a n . The score's value never becomes a coordinate of any forwarded vector, so its quadratic bit-length never enters the next layer's input.

Combining the two parts: every value produced by T on any input has the form p/q with p and q each of bit-length polynomial in |T | . ■

## A.5 PROOF OF PROPOSITION 13

Proposition 13. Given a UHAT T recognizing a language L ⊆ Σ + , one can construct in exponential time an LTL formula φ that recognizes L .

Proof. Setup. Let T be a UHAT recognizing a language L ⊆ Σ + and let F be the set of binary representations of rational numbers that may occur during the computation of T , as in Prop. 12. For the ℓ th layer of T and every v ∈ F S , where S is the output dimension of layer ℓ , we construct an LTL formula φ ℓ v such that, on input a = a 1 ... a N ∈ Σ + , the ℓ th layer outputs v at position n ∈ [ N ] if and only if a , n | = φ ℓ v . We define φ ℓ v inductively on ℓ .

Base case ( ℓ = 0 ). Let emb : Σ → Q D be the symbol embedding of T , and for all v ∈ F D let

̸

$$\varphi _ { v } ^ { 0 } \colon = \begin{cases} \bigvee _ { a \in \bar { \ } e m b ^ { - 1 } ( v ) } Q _ { a } , & \text {if} \ e m b ^ { - 1 } ( v ) \neq \emptyset \\ \bot , & \text {otherwise} . \end{cases}$$

We now define the formula for layer ℓ +1 , splitting on the type of layer.

Inductive step-ReLU layer. If layer ℓ +1 is a ReLU layer of width R applying ReLU to the k th coordinate, we set

$$\varphi _ { v } ^ { \ell + 1 } \coloneqq \bigvee _ { \max \{ 0 , u \} = v _ { k } } \varphi _ { ( v _ { 1 \cdot k - 1 } , u , v _ { k + 1 \colon R } ) } ^ { \ell } \\$$

for all v ∈ F R .

Inductive step-attention with strict masking. If layer ℓ + 1 is an attention layer with strict future masking and rightmost tie-breaking defined by an affine transformation C : Q 2 R → Q S and a score function S: Q 2 R → Q R , we let

$$\varphi _ { v } ^ { \ell + 1 } \coloneqq \bigvee \varphi _ { u } ^ { \ell } \wedge \left ( \left ( \bigvee _ { b \in F ^ { R } } \omega _ { b } \right ) \mathbf S \left ( \varphi _ { a } ^ { \ell } \wedge \mathbf P \bigvee _ { b \in F ^ { R } } \right ) \right ) \\ \stackrel { u , a \in F ^ { R } , } { C } ( u , a ) = & v \quad S ( u , b ) < S ( u , a ) \quad S ( u , b ) > S ( u , a )$$

for all v ∈ F S . To account for the special case where the set of unmasked positions is empty, we take the disjunction of the previous formula with

$$( \neg P \top ) \wedge \bigvee _ { \substack { u \in F ^ { R } , \\ C ( u , 0 ) = v } } ^ { \ell }$$

We omit this special case in what follows. If the layer uses leftmost tie-breaking instead, we adapt the formula as follows:

$$\varphi _ { v } ^ { \ell + 1 } \colon = \bigvee _ { \substack { u , a \in F ^ { R } , \\ C ( u , a ) = v } } \varphi _ { u } ^ { \ell } \wedge \left ( \mathbf P \left ( \varphi _ { a } ^ { \ell } \wedge \mathbf P \right ) \bigvee _ { b } \varphi _ { b } ^ { \ell } \right ) \wedge \left ( - \mathbf P \bigvee _ { b } \varphi _ { b } ^ { \ell } \right ) \\$$

The case of strict past masking is similar, with U in place of S and F in place of P .

Inductive step-attention without masking. If the layer uses no masking and rightmost tiebreaking, we distinguish three cases according to where the attention vector lies relative to the current position: at the current position, strictly to the left, or strictly to the right. When the attention vector is at the current position, we define φ ℓ +1 v , at as

$$\varphi _ { v , a i } ^ { \ell + 1 } \colon = \bigvee \varphi _ { u } ^ { \ell } \wedge \left ( \neg P \bigvee \varphi _ { b } ^ { \ell } \right ) \wedge \left ( \neg F \bigvee \varphi _ { b } ^ { \ell } \right ) . \\ \underset { C ( u , u ) = v } { u \in F ^ { R } } \underset { S ( u , b ) > S ( u , u ) } { b \in F ^ { R } } \underset { S ( u , b ) \geq S ( u , u ) } { b \in F ^ { R } } , \\$$

When the attention vector is strictly to the left of the current position, we define φ ℓ +1 v , L as

$$\varphi _ { v , L } ^ { \ell + 1 } \colon = \bigvee _ { \substack { u , a \in F ^ { R } , \\ C ( u , a ) = v , \\ S ( u , a ) > S ( u , u ) } } \left \langle \varphi _ { u } ^ { \ell } \wedge \left ( - \mathbf F \bigvee _ { b \in F ^ { R } } \ell \right ) \wedge \left ( \left ( \bigvee _ { b \in F ^ { R } } \varphi _ { b } ^ { \ell } \right ) \mathbf S \left ( \varphi _ { a } ^ { \ell } \wedge \mathbf P \right ) \bigvee _ { b \in F ^ { R } } \varphi _ { b } ^ { \ell } \right ) \right ) . \quad ( 3 0 ) \\$$

Similarly, when the attention vector is strictly to the right, we define φ ℓ +1 v , R as

$$\varphi _ { v , R } ^ { \ell + 1 } \colon = \bigvee _ { \substack { u , a \in F ^ { R } , \\ C ( u , a ) = v , \\ S ( u , a ) \geq S ( u , u ) } } \left / \varphi _ { u } ^ { \ell } \wedge \left ( - \mathbf P \bigvee _ { b \in F ^ { R } } \ell \right ) \wedge \left ( \mathbf F ( \varphi _ { a } ^ { \ell } \wedge \mathbf F \bigvee _ { b \in F ^ { R } } \ell ) \right ) \wedge \left ( - \mathbf F \bigvee _ { b \in F ^ { R } } \ell \right ) . \, ( 3 1 ) \\ \quad S ( u , a ) = v , \quad S ( u , b ) > S ( u , a ) \quad S ( u , b ) \geq S ( u , a ) \quad S ( u , b ) > S ( u , a )$$

In the case of no masking and rightmost tie-breaking, we set

$$\varphi _ { v } ^ { \ell + 1 } \colon = \varphi _ { v , \mathbf a t } ^ { \ell + 1 } \vee \varphi _ { v , L } ^ { \ell + 1 } \vee \varphi _ { v , R } ^ { \ell + 1 } .$$

The case of no masking and leftmost tie-breaking is analogous.

Acceptance formula. If T has m layers whose last layer outputs vectors of dimension S , and t ∈ Q S is its acceptance vector, we define

$$\varphi \coloneqq \bigvee _ { \substack { v \in F ^ { S } \\ ( t , v ) > 0 } } ^ { m } \varphi _ { v } ^ { m } .$$

Then a , N | = φ if and only if a ∈ L .

Complexity. It remains to argue that φ can be computed in exponential time. By Prop. 12, | F | is exponential in |T | and every representation in F has polynomial bit-length; moreover, F can be computed in exponential time. At every layer ℓ + 1 of width R , the formula φ ℓ +1 v depends on | F | O ( R ) formulas from layer ℓ , and can be computed in time polynomial in | F | R · |T | , since we only have to evaluate affine transformations on vectors from F R , each of polynomial bit-length. At the last layer m , the formulas φ m v depend on | F | O ( R ′ m ) formulas from layer 0 , where R ′ is the maximum layer width. Hence φ m v , and therefore φ , has size exponential in |T | and is computable in exponential time. ■

## A.6 PROOF OF PROPOSITION 16

Proposition 16. UHATs have polynomially bounded expansion over LTL. In particular, given an LTL formula φ , one can construct in polynomial time a UHAT T such that L ( T ) = L ( φ ) .

Proof sketch. We construct T by induction on the subformula structure of φ , maintaining the following invariant: after processing a subformula ψ , the UHAT built so far has, at every position n of any input a , a designated output coordinate carrying the truth value of a , n | = ψ . Once the invariant is established for ψ = φ , the acceptance vector reads off that coordinate at the last position.

Atomic formulas. For φ = Q a , the token embedding already provides the indicator 1 [a n = a] at each position, which is precisely the required truth value.

Boolean combinations. The truth value of ¬ ψ , ψ 1 ∧ ψ 2 , or ψ 1 ∨ ψ 2 at position n depends only on the children's truth values at the same position n , so a single affine-and-ReLU layer computes the required coordinate point-wise from coordinates produced by the inductive hypothesis.

Since ( φ = φ 1 S φ 2 ). By inductive hypothesis we have coordinates for φ 1 and φ 2 at every position; an affine-and-ReLU layer computes ¬ φ 1 ∨ φ 2 at every position. We then use an attention layer with strict future masking and rightmost tie-breaking to get for every position i the maximal position j &lt; i where ¬ φ 1 ∨ φ 2 holds and output at position i the truth value of φ 2 from position j . For positions ℓ and subformulas ψ , we write ψ ( ℓ ) ∈ { 0 , 1 } for the indicator of a , ℓ | = ψ . By the maximality of j , every k with j &lt; k &lt; i satisfies φ 1 ( k ) ∧ ¬ φ 2 ( k ) (otherwise k would be a more recent witness); in particular φ 1 ( k ) holds for all k ∈ ( j, i ) . Thus if φ 2 ( j ) = 1 , then j witnesses φ 1 S φ 2 at i under the semantics of § 2, and if φ 2 ( j ) = 0 then ¬ φ 1 ( j ) = 1 blocks any earlier j ′ ≤ j from witnessing S at i . When no such j exists, the attention layer's default value outputs 0 , which is again consistent with the semantics.

The case where φ = φ 1 U φ 2 is similar using strict past masking and leftmost tie-breaking. ■

## A.7 PROOF OF THEOREM 19

Theorem 19. Deciding the equivalence between two UHATs is EXPSPACE -complete.

Proof. To prove the lower bound, we reduce from the non-emptiness problem for UHATs, which by Thm. 4 is EXPSPACE -complete. To this end, let T be a given UHAT and fix a UHAT T 0 that recognizes the empty language. Then we have that T and T 0 are equivalent if and only if T recognizes the empty language.

To prove the upper bound, let the UHATs T 1 and T 2 be given. We apply Prop. 13 to turn T 1 and T 2 in exponential time into LTL formulas φ 1 and φ 2 , respectively. Now, T 1 and T 2 are equivalent if and only if φ 1 and φ 2 are equivalent. The latter can be decided in polynomial space (Sistla and Clarke, 1985), which results in an exponential-space algorithm in total. ■