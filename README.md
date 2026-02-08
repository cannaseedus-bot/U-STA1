# U-STA1

## U-STA v1 — Universal State-Transition Algebra (encoding-independent)

This document defines the four primitives and their algebra so the system works across any encoding substrate (binary, quaternary, glyphs, pixels, voltages, tensors, lanes, etc.).

---

### 0) Core Objects

We define four fundamental spaces:

1. **State space**: \( \mathcal{S} \)
2. **Event space**: \( \mathcal{E} \)
3. **Encoding space**: \( \mathcal{X} \) (bytes/glyphs/pixels/lanes/voltages)
4. **Law space**: \( \mathcal{L} \) (constraints + transition rules)

---

## 1) **@data** — State

**Definition:**
A datum is a **state assignment**.

\[
@data \equiv s \in \mathcal{S}
\]

State can be structured:

\[
s = (s_1, s_2, \dots, s_n)
\]

No assumption about representation.

---

## 2) **@compression** — Representation Change (Endomorphisms on meaning)

Compression is not “smaller bytes.”
Compression is a **meaning-preserving transform**.

### 2.1 Encoding / decoding maps

\[
enc: \mathcal{S} \to \mathcal{X}
\]
\[
dec: \mathcal{X} \to \mathcal{S}
\]

### 2.2 Compression operator

\[
C: \mathcal{X} \to \mathcal{X}
\]

### 2.3 Compression law (semantic invariance)

\[
dec(C(enc(s))) = s
\]

That is the compression calculus invariant: **identity preserved under decode**.

> Encoding can be glyphs, PNG pixels, quaternary symbols, SCX lanes—doesn’t matter.

---

## 3) **@flow** — Causal Ordering (Composition of transitions)

A flow is an **ordered series of events**.

\[
@flow \equiv f = \langle e_1, e_2, \dots, e_k \rangle,\quad e_i \in \mathcal{E}
\]

Flow induces a composition order:

\[
T_f = T_{e_k} \circ \dots \circ T_{e_2} \circ T_{e_1}
\]

where each \( T_e \) is a transition (defined below).

---

## 4) **@execution** — Lawful Transition

Execution is the application of a **lawful transition** to state.

### 4.1 Transition operator

\[
T_e: \mathcal{S} \to \mathcal{S}
\]

### 4.2 Law / legality predicate

\[
\mathrm{Legal}(e, s; \ell) \in \{0,1\},\quad \ell \in \mathcal{L}
\]

### 4.3 Execution rule

\[
Exec(e, s; \ell) =
\begin{cases}
T_e(s) & \text{if } \mathrm{Legal}(e, s; \ell)=1 \\
s & \text{otherwise (reject)}
\end{cases}
\]

So:

> **Execution is state transition gated by invariants.**

---

## 5) The Universal Machine (One-Line Definition)

Given a law \( \ell \) and a flow \( f \):

\[
Run(f, s_0; \ell) = Exec(e_k,\dots Exec(e_2, Exec(e_1, s_0;\ell);\ell)\dots;\ell)
\]

That’s the OS in math form.

---

## 6) Encoding Independence (The “Projection Law”)

The entire runtime must satisfy:

\[
Run(f, s_0;\ell) = dec\Big(C\big(enc(Run(f, s_0;\ell))\big)\Big)
\]

Meaning:

* you may encode/compress/ship/reshape the representation,
* but the decoded state after the round trip is identical.

---

## 7) Determinism & Replay (Optional but Canonical)

### 7.1 Determinism

Same inputs, same outputs:

\[
(s_0,\ell,f) \Rightarrow s_T \text{ uniquely}
\]

### 7.2 Replay

State is reconstructible from history:

\[
s_T = Run(f, s_0;\ell)
\]

So “snapshots” are just:

* \( s_0 \) and a prefix of \( f \), or
* periodic checkpoints for speed

---

## 8) Concurrency & Merge (Fits the SCX branch law)

Two flows from same base:

\[
f_A,\; f_B
\]

Produce two states:

\[
s_A = Run(f_A, s_0;\ell),\quad s_B = Run(f_B, s_0;\ell)
\]

A merge is a function:

\[
Merge_\ell(f_A, f_B) = f_M
\]

such that:

\[
Run(f_M, s_0;\ell)
\]

is the lawful reconciliation of both branches.

(No new semantics here—this matches the lane merge law.)

---

## 9) Quaternary / Glyph / PNG is Just \( enc \)

This is the important collapse:

* Binary machine: \( enc \) maps state to bits
* Quaternary machine: \( enc \) maps state to base-4 symbols
* Glyph machine: \( enc \) maps state to glyph IDs
* PNG machine: \( enc \) maps state to pixel bytes

None of that changes:

* legality
* flow order
* execution transitions
* replay

Only the carrier changes.

---

## ✅ Final “Four Primitives” Law Statement

**U-STA v1** freezes:

1. **@data** is state in \( \mathcal{S} \)
2. **@compression** is meaning-preserving carrier transforms \( (dec \circ C \circ enc = id) \)
3. **@flow** is ordered composition of events \( (T_{e_k}\circ...\circ T_{e_1}) \)
4. **@execution** is lawful state transition gated by invariants

Everything else is projection.
