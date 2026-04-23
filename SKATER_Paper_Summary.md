# SKATER: Synthesized Kinematics for Advanced Traversing Efficiency on a Humanoid Robot via Roller Skate Swizzles

**Paper:** arXiv:2601.04948 (January 8, 2026)  
**Authors:** Junchi Gu, Feiyang Yuan, Weize Shi, Tianchen Huang, Haopeng Zhang, Xiaohu Zhang, Yu Wang, Wei Gao, Shiwu Zhang  
**Affiliation:** Institute of Humanoid Robots, Department of Precision Machinery and Precision Instrumentation, University of Science and Technology of China (USTC), Hefei, Anhui 230026, China  
**Contact:** weigao@ustc.edu.cn; swzhang@ustc.edu.cn

---

## 1. Core Motivation

Traditional bipedal walking/running generates high instantaneous impact forces at each foot strike, leading to accelerated joint wear and poor energy utilization. Roller skating leverages body inertia for continuous sliding with minimal kinetic energy loss (~50% of joint impact forces compared to running). SKATER proposes roller skating as a superior locomotion mode for humanoid robots, specifically using the **swizzle gait** (forward propulsion through rhythmic opening and closing of both feet while maintaining continuous ground contact).

---

## 2. Hardware: SKATER Robot

- **Total DoF:** 33 (structural), **25 actuated DoF**
  - **6 DoF per leg** (confirmed from paper)
  - Remaining actuated DoF distributed across torso, arms, head
- **Foot Design:** Each foot equipped with a row of **4 passive inline wheels** — a specifically optimized roller skating mechanism (unlike traditional humanoid robots that employ either rigid flat feet or actuated wheels)
- **Custom Platform:** Purpose-built humanoid robot (not based on a commercial platform like Unitree G1)
- **Onboard Compute:** Intel NUC for state feedback
- **Remote Control:** Open-source ELRS (ExpressLRS) remote controller for sending operation commands
- **Actuation:** Joint-level PD control
  - Joint torque at time step $t$: $\tau_t = K_p (a_t - q_t) + K_d (\dot{q}_t)$
  - $K_p$, $K_d$: proportional and derivative gains
- **Action Limits:** Parameter $\beta$ bounds the action limits, thereby implicitly regulating the motion velocity

---

## 3. RL Framework Overview

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Architecture:** Actor-Critic MLP
- **Simulator:** Isaac Gym (GPU-accelerated parallel simulation) — paper cites the Humanoid-Gym framework [5] as a pivotal technical route for large-scale parallel simulation
- **Policy Input:** Proprioceptive + exteroceptive sensor data
- **Policy Output:** Joint-level position commands (fed to PD controller)
- **Control Paradigm:** Velocity-conditioned joystick-style control
  - Command input: $(v^{cmd}_{xy}, \omega^{cmd}_z)$ — target linear velocity and yaw rate
  - The trained policy tracks these commands in real time
  - At deployment, commands are sent via the ELRS remote controller

---

## 4. Reward Function Design (Table I)

**Total: 22 reward terms**, designed as a multi-objective reward function that balances task completion and motion constraints.

**Key Design Philosophy — "Implicit Gait Guidance":**
The reward function does **NOT** explicitly introduce kinematic trajectories or gait timing constraints for swizzle. Instead, only necessary physical constraints and motion preferences are provided through boundary constraints on inter-foot distance, motion symmetry, and physical consistency constraints. Swizzle gaits **naturally emerge** during the training process.

More precisely, the paper describes this as: *"an implicit gait guidance strategy based on inter-leg geometric relationships, motion symmetry and physical consistency constraints"* — which demonstrates *"superior modeling flexibility and generalization"* compared to traditional phase-based or foot-trajectory-tracking methods, especially given the nonholonomic constraints between passive wheels and the ground.

This "implicit" approach contrasts with "explicit gait planning" methods (e.g., Itabashi/Hashimoto) that manually specify foot trajectories and gait phase timing. The reward itself is hand-engineered (not learned from data as in IRL/AMP).

**Theoretical Basis:** The paper cites Peng et al. [15] as precedent — they successfully generated natural walking gaits by introducing bilateral symmetry rewards without explicitly specifying gait timing. SKATER extends this principle to the more challenging roller skating domain with nonholonomic constraints.

### 4.1 Task Rewards (positive — drive locomotion)

| Reward Term | Formula | Weight |
|---|---|---|
| Linear Velocity Track | $\exp\left(-\frac{\|v^{cmd}_{xy} - v^{base}_{xy}\|^2}{\sigma^2}\right)$ | 3.2 |
| Angular Velocity Track | $\exp\left(-\frac{(\omega^{cmd}_z - \omega^{base}_z)^2}{\sigma^2}\right)$ | 1.2 |
| Alive | Constant | 0.15 |

- Linear Velocity Track (weight **3.2**) is the dominant reward, using Gaussian kernel form
- Angular Velocity Track (weight **1.2**) enables steering/turning
- Alive bonus (weight **0.15**) prevents the agent from learning to fall immediately to avoid all penalties

### 4.2 Base Penalties (regularization — prevent erratic behavior)

| Reward Term | Formula | Weight |
|---|---|---|
| Angular Velocity XY | $-\|\omega^{base}_{xy}\|^2$ | -0.05 |
| Joint Velocity | $-\|\dot{q}\|^2$ | -0.001 |
| Joint Acceleration | $-\|\ddot{q}\|^2$ | $-2.5 \times 10^{-7}$ |
| Action Rate | $-\|a_t - a_{t-1}\|^2$ | -0.05 |
| Joint Position Limits | $-\sum_i \max(0, |q_i - q^0_i| - \ell_i)^2$ | -5.0 |
| Energy | $-\sum_i |\tau_i \dot{q}_i|$ | $-2 \times 10^{-5}$ |

- Joint Acceleration has an extremely small weight ($-2.5 \times 10^{-7}$), acting as a light regularizer
- Joint Position Limits has a high penalty (-5.0), hard boundary on joint ranges
- Energy penalizes mechanical power ($\tau \cdot \dot{q}$), encouraging energy efficiency

### 4.3 Joint Penalties (upper body stability)

| Reward Term | Formula | Weight |
|---|---|---|
| Arms Deviation | $-\sum_{arms} |q_i - q^0_i|$ | -0.4 |
| Waist Deviation | $-\sum_{waist} |q_i - q^0_i|$ | -2.0 |
| Head Deviation | $-|q_{head} - q^0_{head}|$ | -1.0 |
| Ankle Roll Deviation | $-\sum_{ankle} |q_i - q^0_i|$ | -0.2 |

- Waist deviation has the highest penalty (-2.0) in this group — waist stability is critical for balance during roller skating
- All terms penalize deviation from a default/nominal joint configuration $q^0_i$

### 4.4 Posture Penalties (maintain upright stance)

| Reward Term | Formula | Weight |
|---|---|---|
| Flat Orientation | $-\|g_{proj} - [0, 0, -1]^T\|^2$ | **-7.0** |
| Base Height | $-(h_{base} - h_{target})^2$ | -2.0 |

- **Flat Orientation (weight -7.0) is the single largest penalty in the entire reward function**, enforcing that the projected gravity vector aligns with $[0, 0, -1]^T$ (i.e., the torso must stay upright)
- Base Height penalizes deviation from target standing height

### 4.5 Foot Constraints (core mechanism for swizzle emergence)

| Reward Term | Formula | Weight |
|---|---|---|
| Feet Too Near | $-\mathbb{I}(d_{feet} < 0.2m)$ | -1.0 |
| Feet Too Far | $-\mathbb{I}(d_{feet} > 0.5m)$ | -5.0 |

- These two indicator-function penalties define the **geometric boundary** of the swizzle motion
- $d_{feet}$: distance between the two feet
- Minimum distance: **0.2m** (feet can't be too close — no propulsion)
- Maximum distance: **0.5m** (feet can't be too far — loss of stability)
- **Feet Too Far has 5× the penalty of Feet Too Near**, reflecting that over-extension is more dangerous
- The agent is NOT told when to open or close its legs — only the spatial boundaries are constrained
- This is the key "implicit" mechanism: swizzle rhythm emerges as the optimal solution within these bounds

### 4.6 Wheel Penalties (roller skating-specific constraints)

| Reward Term | Formula | Weight |
|---|---|---|
| Wheel Axial Slip | $-\sum_{wheels} |v_{axial}|$ | -0.1 |
| Wheel Air Time | $-\mathbb{I}(n_{contact} < n_{min})$ | -1.0 |

- **Wheel Axial Slip**: penalizes lateral (axial) sliding of wheels — wheels should only roll along their rolling direction, enforcing the nonholonomic constraint
- **Wheel Air Time**: penalizes wheels losing ground contact — this is the **opposite** of traditional walking rewards that encourage swing-phase foot clearance; in roller skating, both feet must maintain continuous ground contact

### 4.7 Symmetry Rewards (encourage bilateral coordination)

| Reward Term | Formula | Weight |
|---|---|---|
| Leg Symmetry | $-(w_q\|q_L - q_R\|^2 + w_v\|\dot{q}_L - \dot{q}_R\|^2)$ | 0.5 |
| Arm Symmetry | $-(w_q\|q_L - q_R\|^2 + w_v\|\dot{q}_L - \dot{q}_R\|^2)$ | 0.5 |

- The formula is zero when perfectly symmetric and negative when asymmetric; multiplied by positive weight 0.5, the net contribution penalizes asymmetry (0 at best, negative otherwise)
- Penalize asymmetry in both joint positions ($q$) and joint velocities ($\dot{q}$) between left and right limbs
- Uses weighting factors $w_q$ and $w_v$ to balance position vs. velocity symmetry
- Guides the agent toward symmetric alternating swizzle rather than biased one-sided motion
- **Notable:** The positive weight (0.5) distinguishes these from the hard penalties (negative weights) — this is a softer encouragement of symmetry rather than a hard constraint

### 4.8 Contact Penalties

| Reward Term | Formula | Weight |
|---|---|---|
| Undesired Contacts | $-\sum_{body} \mathbb{I}(F_{contact} > 1.0N)$ | -1.0 |

- Penalizes any body part (other than wheels) contacting the ground with force exceeding 1.0N
- Prevents knees, torso, or other non-wheel surfaces from hitting the ground

---

## 5. Multi-Stage Curriculum Learning

The training employs a multi-stage curriculum to progressively increase task complexity. The **reward function (all 22 terms) remains the same** across all stages — what changes are the **task difficulty parameters**.

**Confirmed from paper:**
- Multi-stage curriculum learning strategy is employed to progressively increase task complexity during training
- The curriculum, combined with the implicit reward design, enables natural emergence of swizzle gaits
- $\beta$ bounds action limits, thereby implicitly regulating motion velocity (likely scheduled across stages)

**Inferred from standard practices in similar work (not explicitly stated in accessible text):**
- Progressive velocity command range expansion: early stages likely use zero or low target velocities (focusing on balance/standing), later stages increase the command velocity range
- Domain randomization intensity may increase progressively across training stages

> **Note:** The exact number of curriculum stages, their specific parameters, and the scheduling of $\beta$ are detailed in the paper's methodology section. The PDF contains this information but was not fully extractable during this review. Consult the original paper (Section III) for precise curriculum specifications.

---

## 6. Domain Randomization (Table II)

All parameters are sampled from uniform distributions $\mathcal{U}(a, b)$ at each episode reset.

### 6.1 Physics Material — Wheels

| Parameter | Value |
|---|---|
| Static Friction | $\mathcal{U}(0.1, 0.8)$ |
| Dynamic Friction | $\mathcal{U}(0.1, 0.4)$ |
| Restitution | 0.0 |

- Very wide friction range — accounts for diverse real-world ground surfaces (tile, concrete, smooth floors, etc.)
- Restitution fixed at 0.0 (no bouncing)

### 6.2 Link Mass

| Parameter | Value |
|---|---|
| Link Mass | $\mathcal{U}(0.9, 1.1) \times$ default kg |

- ±10% mass randomization per link
- Accounts for manufacturing tolerances and potential payload variations

### 6.3 Center of Mass Offset

| Parameter | Value |
|---|---|
| x-axis | $\mathcal{U}(-0.01, 0.01)$ m |
| y-axis | $\mathcal{U}(-0.01, 0.01)$ m |
| z-axis | $\mathcal{U}(-0.01, 0.01)$ m |

- ±1cm CoM offset in all three axes
- Accounts for uneven mass distribution in the physical robot

### 6.4 Actuator Gains

| Parameter | Value |
|---|---|
| Stiffness ($K_p$) | $\mathcal{U}(0.9, 1.1) \times$ default |
| Damping ($K_d$) | $\mathcal{U}(0.9, 1.1) \times$ default |

- ±10% randomization on PD controller gains
- Models motor characteristic variations between joints and units

### 6.5 Joint Damping — Wheels

| Parameter | Value |
|---|---|
| Wheel Joint Damping | $\mathcal{U}(0.002, 0.005)$ N·m·s/rad |

- Models real-world bearing friction variability in passive wheels
- Ensures the policy is robust to wheels with different levels of "smoothness"

---

## 7. Sim-to-Real Transfer

Two-pronged approach:

1. **System Identification**: Physical robot parameters (damping coefficients, moments of inertia) are carefully calibrated using system identification methods
2. **Domain Randomization**: Uncalibrated parameter discrepancies are addressed through the randomization scheme described in Table II during training

---

## 8. Observation and Action Space

- **Observations (Policy Input):**
  - Proprioceptive: joint positions, joint velocities, base orientation, base angular velocity
  - Exteroceptive: additional sensor data (details in paper)
  - Velocity commands: $(v^{cmd}_{xy}, \omega^{cmd}_z)$

- **Actions (Policy Output):**
  - Target joint positions for PD controller
  - Bounded by parameter $\beta$

---

## 9. Key Control Challenges Addressed

1. **Violation of static contact assumption**: Roller skating involves continuous foot sliding during stance phase, unlike traditional bipedal locomotion that assumes static foot-ground contact
2. **Nonholonomic constraints**: Wheels can only roll along specific directions without lateral sliding, adding complexity to motion planning and control
3. **Centroidal dynamics coordination**: Requires precise coordination of center-of-mass dynamics, leg swing, and sliding motions simultaneously

---

## 10. Experimental Results

### 10.1 Quantitative Improvements (Swizzle vs. Walking on SKATER)

| Metric | Reduction |
|---|---|
| Impact Intensity | **75.86%** |
| Cost of Transport (CoT) | **63.34%** |

- Peak torques in hip and ankle joints significantly reduced (exact percentage in paper)
- **100% success rate** across ground surfaces with various friction coefficients

### 10.2 Capabilities

- Forward locomotion at varying speeds via velocity command tracking
- Turning via angular velocity command tracking
- Smooth, continuous swizzle gait with both feet maintaining ground contact

### 10.3 Limitations (stated by authors)

- **Lateral drift** during straight-line skating
- **Velocity tracking response lag**
- Only **swizzle gaits** realized (continuous ground contact of both feet) — no single-leg gliding, crossover steps, or other advanced skating maneuvers

---

## 11. Related Work Context (cited by SKATER)

Key prior work cited in the paper that may be relevant for roller skating RL projects:

- **ANYmal quadruped roller skating (ETH)** [2/11]: Replaced traditional friction cone model with a **friction triangle model** to handle nonholonomic constraints; employed hierarchical force control for whole-body motion. Achieved CoT reduction of over 80% compared to walking. This is the most mature prior work on learning-based roller skating locomotion.
- **Hashimoto et al.** [6]: Swizzle skating control for bipedal robot with passive wheels, analyzed anisotropic friction characteristics between rolling and lateral directions. Early explicit gait planning approach.
- **Itabashi et al.** [8]: Bipedal robot with variable-curvature roller skating mechanisms, used fifth-order Bézier curves for trajectory planning. Limited to position control only, resulting in low robustness.
- **Chen et al.** [12]: Quadrupedal robot with 4-DoF legs equipped with passive wheels; proposed geometrically characterized passive wheel model for improved contact point accuracy.
- **Peng et al.** [15]: Generated natural walking gaits via bilateral symmetry rewards without explicit gait timing — the theoretical basis for SKATER's implicit gait guidance approach.
- **Humanoid-Gym** [5]: Isaac Gym-based RL framework for humanoid locomotion with zero-shot sim-to-real transfer — the technical infrastructure SKATER builds upon.

---

## 12. Key Takeaways for Roller Skating RL Projects

### What makes SKATER's reward design unique:

1. **No motion reference data needed**: Unlike AMP-based approaches, SKATER does not use any expert demonstration or motion capture data. Swizzle emerges purely from reward engineering.

2. **Boundary constraints instead of trajectory tracking**: The foot distance constraints ($0.2m < d_{feet} < 0.5m$) elegantly encode the geometric essence of swizzle without specifying temporal dynamics.

3. **Wheel-specific penalties are essential**: Wheel Axial Slip and Wheel Air Time are roller-skating-specific terms absent from standard walking reward functions. They enforce the nonholonomic constraint and continuous contact requirement.

4. **Flat Orientation dominance**: The largest single penalty (weight -7.0) is on body orientation — maintaining an upright torso is the #1 priority, even above velocity tracking.

5. **Symmetry as an explicit reward term**: Rather than just relying on the environment to produce symmetric motion, symmetry rewards with separate $w_q$ (position) and $w_v$ (velocity) weighting actively guide the agent toward bilateral coordination — inspired by Peng et al.'s work on generating natural gaits via symmetry rewards.

### Comparison with AMP+PPO approach:

| Aspect | SKATER (Pure PPO) | AMP + PPO |
|---|---|---|
| Motion Reference | None | Expert demonstration required |
| Reward Source | 22 hand-engineered terms | Discriminator (learned) + task reward (hand-written) |
| Gait Specification | Implicit via boundary constraints | Implicit via style matching |
| Generalization | Flexible (no reference data dependency) | Tied to quality of reference data |
| Engineering Effort | Heavy reward tuning (22 terms + weights) | Reference data collection + discriminator training |
| Training Stability | Standard PPO | Adversarial training dynamics |

---

*This summary is based on arXiv:2601.04948v1 (submitted January 8, 2026). Refer to the original paper for complete mathematical formulations, training hyperparameters, and detailed experimental analysis. Specific values for Table I and Table II are transcribed directly from the paper.*
