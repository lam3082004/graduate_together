# Problem Specification
## UAV-Assisted Anti-Jamming D2D Communication Network with Ambient Backscatter and Multi-Agent Deep Reinforcement Learning

---

## 1. Problem Overview

This research addresses the challenge of maintaining reliable wireless communication between ground-level user devices (Device-to-Device, D2D) in the presence of malicious RF jamming attacks. Rather than treating jamming signals purely as threats to avoid, the system adopts a counter-intuitive strategy: **exploiting high-power jamming signals as ambient RF energy sources** to power Ambient Backscatter Communications (AmBC). UAVs are deployed as mobile relay nodes that dynamically reposition in 3D space to assist ground users, while a multi-agent deep reinforcement learning (MADRL) framework coordinates all decision-making agents in real time.

The core research question is:

> *How can multiple UAV relay agents and ground user agents be jointly trained — using deep reinforcement learning — to simultaneously maximize D2D throughput, satisfy QoS constraints, and minimize energy consumption, under continuous jamming attacks and partial channel observability?*

---

## 2. System Model

### 2.1 Network Entities

The network operates within a bounded 2D geographical area and consists of the following entities:

| Entity | Symbol | Role |
|---|---|---|
| Source User (ground) | $S_i$, $i \in \mathcal{N} = \{1,\ldots,N\}$ | Energy-constrained IoT/mobile transmitter |
| Destination User (ground) | $D_i$ | Intended receiver for $S_i$ |
| Fixed Relay Base Station | RBS | Static ground-level relay infrastructure |
| UAV Relay | $U_k$, $k \in \mathcal{K} = \{1,\ldots,K\}$ | Mobile aerial relay; repositions each time slot |
| Malicious Jammer | $J$ | Ground-level device; continuously broadcasts interference at power $P_J$ |

The system has $N$ SU–DU pairs operating simultaneously under a single persistent jammer.

### 2.2 Signal and Backscatter Model (Ambient Backscatter)

Rather than generating their own carrier signals — which would require dedicated power — source nodes $S_i$ operate in **Ambient Backscatter** mode. The jammer's high-power transmission inadvertently serves as the RF excitation source. $S_i$ modulates its data onto the reflected jamming signal by dynamically varying its reflection coefficient $\alpha_i \in [0, 1]$.

The received signal at destination $D_i$ (Direct D2D mode) is:

$$y_{D_i} = \sqrt{P_J}\, h_{J,S_i} h_{S_i,D_i} \alpha_i x_{S_i} + n_{D_i}$$

where $h_{X,Y}$ is the complex channel coefficient between nodes $X$ and $Y$, $x_{S_i}$ is the transmitted symbol, and $n_{D_i} \sim \mathcal{CN}(0, N_0)$ is additive white Gaussian noise.

**Three Backscatter Paradigms (for background context):**

- **Monostatic:** The reader both emits the carrier and receives the backscattered signal (e.g., classic RFID). High hardware cost; not used here.
- **Bistatic:** A dedicated carrier emitter, a passive tag, and a separate reader (e.g., phone acting as reader while tag harvests energy). More flexible but requires coordination.
- **Ambient (adopted in this work):** No dedicated carrier emitter. The tag (here, $S_i$) harvests and reflects existing ambient RF signals — in this case the jammer's transmission — to communicate. Zero additional power expenditure on the transmitter side.

### 2.3 Transmission Modes

Each source node $S_i$ selects a transmission mode $m_i$ at every time slot:

**Mode 0 — Direct D2D**

```
S_i  ──[AmB reflection of jamming signal]──▶  D_i
```

$S_i$ backscatters the jamming signal directly to $D_i$. Effective when the two devices are geographically close. The SINR is:

$$\gamma_i^{\text{D2D}} = \frac{G P_J g_{J,S_i} g_{S_i,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}$$

where $G$ is the composite backscatter gain and $g_{X,Y} = |h_{X,Y}|^2$.

**Mode 1 — Fixed RBS Relay**

```
S_i  ──[AmB]──▶  RBS  ──[forward]──▶  D_i
```

$S_i$ backscatters to the fixed Relay Base Station, which harvests energy from the jamming signal to decode and forward the message to $D_i$. The end-to-end SINR is bottlenecked by the weaker of the two hops:

$$\gamma_i^{\text{RBS}} = \min\!\left(\frac{G P_J g_{J,S_i} g_{S_i,R} \alpha_i^2}{P_J g_{J,R} + N_0},\;\frac{G P_J g_{J,R} g_{R,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}\right)$$

**Mode 2 — UAV Relay** *(primary extension)*

```
S_i  ──[AmB]──▶  U_k  ──[forward]──▶  D_i
```

$S_i$ backscatters to UAV $U_k$, which uses its onboard power to decode and forward the message to $D_i$. The UAV can reposition to improve both hops simultaneously. Because the UAV-to-ground links are aerial, they follow a probabilistic Line-of-Sight (LoS) model:

$$P_{\text{LoS}}(\theta) = \frac{1}{1 + a\exp\!\bigl(-b(\theta - a)\bigr)}, \quad \theta = \frac{180}{\pi}\arctan\!\left(\frac{H_k}{d_{U_k, X}}\right)$$

The average channel gain between UAV $U_k$ and ground node $X$ is:

$$\bar{g}_{U_k,X} = P_{\text{LoS}} \cdot L_{\text{LoS}}^{-1} + (1 - P_{\text{LoS}}) \cdot L_{\text{NLoS}}^{-1}$$

The SINR for Mode 2:

$$\gamma_i^{\text{UAV}} = \min\!\left(\frac{G P_J g_{J,S_i} \bar{g}_{S_i,U_k} \alpha_i^2}{P_J \bar{g}_{J,U_k} + N_0},\;\frac{G P_J \bar{g}_{U_k,D_i} \alpha_i^2}{P_J g_{J,D_i} + N_0}\right)$$

**Why is Mode 2 necessary?**

| Scenario | Direct D2D | Fixed RBS | UAV Relay |
|---|---|---|---|
| $S_i$ and $D_i$ are close | ✅ Works well | Overhead | Overhead |
| $S_i$ and $D_i$ are far apart | ❌ High path loss | ✅ if RBS is central | ✅ UAV moves closer |
| RBS is far from both nodes | — | ❌ Both hops weak | ✅ UAV repositions |
| Jammer blocks line-of-sight | ❌ | Partial | ✅ UAV flies around obstruction |

---

## 3. Problem Formulation

### 3.1 Decision Variables

Each time slot $t$, the following decisions are made jointly:

- **For each $S_i$:** reflection coefficient $\alpha_i(t) \in [0,1]$ and transmission mode $m_i(t) \in \{0, 1, 2\}$.
- **For each UAV $U_k$:** 3D displacement $\Delta\mathbf{p}_{U_k}(t) = (\Delta x_k, \Delta y_k, \Delta z_k)$, subject to physical constraints.

### 3.2 Reward Function

The per-agent reward at time step $t$ is designed to balance throughput, reliability, and energy efficiency:

$$r_i(t) = \underbrace{w_1 \log_2(1 + \gamma_i(t))}_{\text{throughput}} + \underbrace{w_2 \tanh\!\left(\frac{\gamma_i(t)}{\Gamma_{\text{th}}} - 1\right)}_{\text{soft QoS constraint}} - \underbrace{w_3\, \alpha_i(t)^2}_{\text{energy penalty}}$$

- The **throughput** term rewards higher spectral efficiency.
- The **soft QoS** term (using $\tanh$) provides a smooth gradient signal: positive when $\gamma_i \geq \Gamma_{\text{th}}$, negative otherwise. This avoids sparse reward issues.
- The **energy penalty** discourages unnecessarily high reflection coefficients.

A UAV positioning reward is added to the global critic:

$$r_{U_k}^{\text{pos}}(t) = \sum_{i \in \mathcal{C}_k} \Delta\gamma_i(t) - w_4 \|\Delta\mathbf{p}_{U_k}(t)\|$$

where $\mathcal{C}_k$ is the set of SU nodes served by UAV $k$ and $\Delta\gamma_i(t)$ is the SINR improvement from the UAV's repositioning.

### 3.3 Optimization Objective

Maximize total accumulated network reward over $T$ steps across all $N$ user pairs:

$$\max_{\{\alpha_i,\, m_i,\, \Delta\mathbf{p}_{U_k}\}} \;\mathbb{E}\!\left[\sum_{t=1}^{T} \sum_{i=1}^{N} r_i(t)\right]$$

### 3.4 Constraints

| Constraint | Description |
|---|---|
| $\alpha_i \in [0,1]$ | Valid reflection coefficient range |
| $m_i \in \{0, 1, 2\}$ | Discrete mode selection |
| $\mathbf{p}_{U_k}(t+1) = \mathbf{p}_{U_k}(t) + \Delta\mathbf{p}_{U_k}(t)$ | UAV position update |
| $H_{\min} \leq H_{U_k}(t) \leq H_{\max}$ | Altitude bounds |
| $\|\Delta\mathbf{p}_{U_k}(t)\| \leq v_{\max} \cdot \delta_t$ | Maximum flight speed |
| $\sum_t E_{U_k}^{\text{move}}(t) \leq E_{\max}$ | UAV total energy budget |

---

## 4. POMDP Formulation

The problem is cast as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**, since no agent has access to the full global state. There are two classes of agents:

- **SU agents** $\{S_1, \ldots, S_N\}$: control $(\alpha_i, m_i)$
- **UAV agents** $\{U_1, \ldots, U_K\}$: control $(\Delta x_k, \Delta y_k, \Delta z_k)$

### 4.1 Observation Spaces

**SU agent $i$** observes only local information:

$$o_i(t) = \bigl\{\gamma_i^{(t-1)},\; E_i^{(t-1)},\; m_i^{(t-1)},\; \hat{P}_J^{(t-1)}\bigr\}$$

(previous SINR, energy cost, transmission mode, estimated jamming power)

**UAV agent $k$** observes:

$$o_{U_k}(t) = \bigl\{\mathbf{p}_{U_k}^{(t)},\; \{\bar{g}_{S_i,U_k}\}_i,\; \{\bar{g}_{U_k,D_i}\}_i,\; \hat{P}_J^{(t-1)},\; E_{U_k}^{\text{remain}}(t)\bigr\}$$

(current position, channel gains to/from served nodes, estimated jamming power, remaining energy)

### 4.2 Action Spaces

| Agent | Action | Type |
|---|---|---|
| SU $i$ | $a_i = (\alpha_i,\; m_i)$ | Hybrid: continuous $\alpha_i$ + discrete $m_i$ |
| UAV $k$ | $a_{U_k} = (\Delta x_k, \Delta y_k, \Delta z_k)$ | Continuous |

---

## 5. Proposed Solution: IA-MADDPG Extended Framework

Standard MADDPG suffers from slow convergence and "cold-start" instability in high-dimensional continuous action spaces. The proposed framework — an extension of **Imitation-Augmented MADDPG (IA-MADDPG)** — addresses this with two key additions:

### 5.1 Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                  CENTRALIZED CRITIC                       │
│  Inputs : global obs (o_1,...,o_N, o_U1,...,o_UK)        │
│           all actions (a_1,...,a_N, a_U1,...,a_UK)       │
│  Output : global Q-value                                  │
└────────────────────┬─────────────────────────────────────┘
                     │ gradient (training only)
          ┌──────────┴──────────┐
          ▼                     ▼
  ┌───────────────┐    ┌────────────────────┐
  │ SU Actor μ_i  │    │ UAV Actor μ_{Uk}   │
  │ in:  o_i      │    │ in:  o_{Uk}        │
  │ out: α_i, m_i │    │ out: Δx, Δy, Δz    │
  └───────────────┘    └────────────────────┘
```

Execution is fully **decentralized**: each actor uses only its own local observation. Centralized critics are used only during training (CTDE paradigm — Centralized Training, Decentralized Execution).

### 5.2 Phase 1 — Expert Warm-Up

Before RL training begins, the replay buffer is pre-populated with high-quality transitions generated by analytical expert policies:

- **SU expert $\mu^E_{\text{SU}}$:** Greedily selects mode and $\alpha_i$ to maximize instantaneous SINR given full channel knowledge.
- **UAV expert $\mu^E_{\text{UAV}}$:** Moves each UAV toward the position that maximizes the sum SINR of its served SU nodes (geometric heuristic).

This eliminates random early exploration and prevents convergence to poor local equilibria.

### 5.3 Phase 2 — Behavior Cloning Regularization

During training, each actor's loss combines the standard policy gradient with a Behavior Cloning (BC) term:

$$\mathcal{L}(\theta^{\mu_i}) = \underbrace{-\mathbb{E}_{o,a \sim \mathcal{D}}\bigl[Q_i(o, \mathbf{a})\bigr]}_{\mathcal{L}_{\text{MADDPG}}} + \lambda_{\text{IL}} \underbrace{\mathbb{E}_{o \sim \mathcal{D}}\bigl[\|\mu_i(o_i) - \mu^E(o_i)\|^2\bigr]}_{\mathcal{L}_{\text{BC}}}$$

The annealing coefficient $\lambda_{\text{IL}}$ decays over episodes, allowing agents to initially imitate the expert and gradually discover strategies beyond what the expert can achieve.

### 5.4 Supporting Techniques

- **Prioritized Experience Replay (PER):** Samples transitions with high TD-error more frequently, accelerating learning from informative experiences.
- **Target Policy Smoothing (TD3-style):** Adds clipped Gaussian noise to target actions during critic updates, preventing Q-value overfitting to sharp peaks.
- **Curriculum Learning (optional):** Dynamically adjusts the SINR threshold $\Gamma_{\text{th}}$ during training, starting from an easy regime and gradually increasing difficulty.

---

## 6. Baseline Methods for Comparison

| Method | Description |
|---|---|
| Direct Transmission (DT) | Fixed D2D mode only, $\alpha = 0.2$, no relay |
| Greedy Strategy | Selects mode (D2D or RBS) to maximize instantaneous SINR; no learning |
| Frequency Hopping (FH) | Stochastic channel avoidance; does not exploit jamming energy |
| Standard MADDPG | No imitation learning; UAV included but cold-start from random policy |
| IA-MADDPG (RBS only) | Original paper's method; no UAV relay |
| **IA-MADDPG + UAV** *(proposed)* | Full framework with mobile UAV relay and imitation learning |

---

## 7. Simulation Setup

| Parameter | Value |
|---|---|
| Simulation area | $200 \times 200$ m² |
| Number of SU–DU pairs ($N$) | 5 |
| Number of UAVs ($K$) | 2 |
| Jammer power ($P_J$) | 1 W |
| Noise power ($N_0$) | $10^{-4}$ W |
| Backscatter gain ($G$) | $10^4$ |
| Ground path-loss exponent ($\eta$) | 2.7 |
| UAV altitude range | 20–100 m |
| UAV max speed ($v_{\max}$) | 10 m/s |
| SINR threshold ($\Gamma_{\text{th}}$) | 5 dB |
| Training episodes | 600 |
| Steps per episode | 200 |
| Batch size | 256 |
| SU Actor network | [256, 256, 128] |
| UAV Actor network | [256, 256, 128] |
| Critic network | [512, 256, 128, 64] |
| IL annealing decay ($\lambda_{\text{decay}}$) | 0.995 |
| Weights $(w_1, w_2, w_3, w_4)$ | To be tuned |
| Implementation language | Python |

---

## 8. Performance Metrics

| Metric | Symbol | Description |
|---|---|---|
| Average network reward | $\bar{r}$ | Mean reward per step across all SU agents |
| Transmission Success Rate | TSR | Fraction of time steps where $\gamma_i \geq \Gamma_{\text{th}}$ for all $i$ |
| Average throughput | $\bar{R}$ | Mean spectral efficiency in bits/s/Hz |
| Mode selection distribution | — | Proportion of D2D / RBS relay / UAV relay choices |
| UAV trajectory | — | Visualized 3D flight path over one episode |
| Convergence speed | — | First episode where proposed method surpasses best baseline |
| Energy efficiency | — | Bits per Joule of total circuit + UAV movement energy |

---

## 9. Research Work Packages

**Work Package 1 — System Modeling and Problem Formulation**

- Survey jamming and anti-jamming techniques in wireless networks.
- Design the D2D + AmBC system model with fixed RBS and mobile UAV relays under jamming.
- Analyze channel characteristics (ground Rayleigh + aerial probabilistic LoS) under jamming conditions.
- Formulate the joint optimization problem as a Dec-POMDP.

**Work Package 2 — Algorithm Development and Simulation**

- Derive expert policies for SU agents (greedy SINR) and UAV agents (position heuristic).
- Implement the full IA-MADDPG framework with PER and behavior cloning.
- Build the Python simulation environment modeling all channel effects, jamming, and UAV kinematics.
- Evaluate the proposed method against all baselines across the defined performance metrics.
- Conduct ablation studies (no imitation, no UAV, no PER) to isolate each contribution.

---

## 10. Key Novelty Over Prior Work

| Dimension | Existing work | This work |
|---|---|---|
| Relay type | Fixed RBS only [IA-MADDPG paper] | Fixed RBS **+** mobile UAV relay |
| Number of agents | $N$ SU agents | $N$ SU agents + $K$ UAV agents |
| Transmission modes | 2 (D2D, RBS) | 3 (D2D, RBS, UAV) |
| UAV role | Single UAV, RL power control only [8][10] | Multiple UAVs with joint trajectory + mode decisions |
| Channel model | Ground path-loss | Ground path-loss **+** probabilistic LoS A2G |
| Anti-jamming strategy | Avoidance (FH) or static adaptation | **Exploit** jamming as RF energy source (AmBC) |
| Learning | Single-agent DRL [8][10] | Multi-agent CTDE with imitation learning |

---

## 11. Future Extensions

- **Imperfect CSI:** Evaluate robustness when channel state information is noisy or delayed.
- **Mobile jammer:** Extend the model to a jammer with unknown trajectory.
- **Multiple jammers:** Study cooperative or competitive multi-jammer scenarios.
- **Heterogeneous energy sources:** Combine jamming-harvested energy with solar or battery power for UAVs.
- **Large-scale deployment:** Scale to $N > 20$ SU–DU pairs and $K > 5$ UAVs using parameter sharing.
