#import "@preview/touying:0.7.4": *
#import themes.metropolis: *

#set text(font: (
  (name: "libertinus serif", covers: "latin-in-cjk"),
  "Noto Serif CJK SC"
))
#set par(justify: true)

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Trace Is the Interface],
    subtitle: [A Unified Paradigm in Industrialized Post-Training System],
    author: [
      #box[葛煦旸]
    ],
    date: [#box[08-31]],
  ),
  config-colors(
    primary: rgb("#1d6fa5"),
    primary-light: rgb("#c9dceb"),
    secondary: rgb("#0f4c5c"),
    neutral-lightest: rgb("#ffffff"),
    neutral-dark: rgb("#0f4c5c"),
    neutral-darkest: rgb("#1f2a2c"),
  ),
)

#let cover-slide = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-common(freeze-slide-counter: true),
    config-page(fill: self.colors.secondary, margin: (x: 3em, top: 2.5em, bottom: 2em)),
  )
  let info = self.info
  let accent = self.colors.primary.lighten(45%)
  let body = {
    set text(fill: self.colors.neutral-lightest)
    place(top + right, dx: 1.5em, dy: -1em, circle(radius: 6em, fill: white.transparentize(94%)))
    place(top + right, dx: -7em, dy: 6em, circle(radius: 3.2em, fill: accent.transparentize(80%)))
    align(left + horizon)[
      #block(spacing: 0pt, line(length: 3em, stroke: 4pt + accent))
      #v(1em)
      #text(size: 2.6em, weight: "bold", info.title)
      #v(0.1em)
      #text(size: 1.15em, fill: self.colors.neutral-lightest.darken(15%), info.subtitle)
    ]
    place(bottom + left, dy: 0.5em)[
      #set text(size: 0.9em)
      #let dim = self.colors.neutral-lightest.darken(20%)
      #info.author
      #if info.date != none [ #h(0.8em) #text(fill: dim)[·] #h(0.8em) #text(fill: dim, utils.display-info-date(self)) ]
      #if info.institution != none [ \ #text(size: 0.85em, fill: dim, info.institution) ]
    ]
  }
  touying-slide(self: self, body)
})

#cover-slide

== Common Techniques in Post-Training

#let ro(body) = text(fill: rgb("#1e6b86"), weight: "bold", body)          // produced by rollout
#let rq(body) = text(fill: rgb("#b4471b"), body)          // new requirement vs. the previous loss


- *SFT / Distillation*

  $
    cal(L)_"SFT" (theta) = - EE_(x tilde.op cal(D)) space ro(EE_(y tilde.op rq(pi_"teacher") (dot | x))) [ sum_(t in cal(G)(y)) log pi_theta (y_t | x, y_(<t)) ]
  $

- *RFT (rejection sampling)*

  $
    cal(L)_"RFT" (theta) = - EE_(x tilde.op cal(D)) space ro(EE_(y tilde.op rq(pi_(theta_"old")) (dot | x))) [ rq(bb(1)[r(x, y) >= tau]) sum_(t in cal(G)(y)) log pi_theta (y_t | x, y_(<t)) ]
  $

Often, SFT also filters on rewards in the data pipeline, resulting in mixed effect with rejection sampling.

== Common Techniques in Post-Training

- *RL*

  $
    cal(L)_"RL" (theta) = - EE_(x tilde.op cal(D)) space ro(EE_({y^i}_(i=1)^G tilde.op rq(pi_(theta_"old")) (dot | x))) [
      1/G sum_(i=1)^G sum_(t in cal(G)(y^i)) rq(m^i_t hat(A)^i_t) log pi_theta (y^i_t | x, y^i_(<t))
    ]
  $

  - Different algorithm leads to different masking $m^i_t$ and different estimation of $A^i_t$.

- *OPD*

  $
    hat(A)_t = op("sg")[log pi_"teacher" (y_t | x, y_(<t)) - log pi_(theta_"old") (y_t | x, y_(<t))]
  $
  $
    cal(L)_"OPD" (theta) = - EE_(x tilde.op cal(D)) space ro(EE_(ro(y) ro(tilde.op) rq(pi_(theta_"old")) (dot | x))) [ sum_(t in cal(G)(y)) rq(hat(A)_t) log pi_theta (y_t | x, y_(<t)) ]
  $

  - MOPD uses multiple models for $pi_"teacher"$.

// #text(size: 0.8em, fill: gray)[Agarwal et al., GKD, 2023; Thinking Machines, _On-Policy Distillation_, 2025: 9–30× cheaper than SFT distillation to the same reasoning score]

== What Each Technique Needs from Rollout

#speaker-note[
  所有算法都需要 Generation

  SFT 的采样模型不同

  多数时候我们会希望有 reward，即使算法不需要

  RL 需要 TITO。在文本模态下保证 token 一致是很困难的事情。BPE。

  RL 的采样模型和训练模型需要尽可能接近。这需要 rollout 具有足够高的效率。
]

#text(size: 16pt)[
  #set par(justify: false)
  #table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1.4fr),
    align: (left, center, center, center, center, center),
    stroke: (x: none, y: 0.4pt + luma(160)),
    inset: 6pt,
    table.header([], [*SFT*], [*RFT*], [*RL*], [*OPD*], [*MOPD*]),
    [*Trace* at scale, long-tailed episodes], [✓], [✓], [✓], [✓], [✓],
    [Sampling policy], [teacher], [student], [student], [student], [student],
    [Reward from a verifier], [?], [✓], [✓], [---], [---],
    [Exact tokens and log-probs], [---], [---], [✓], [✓], [✓],
    [Policy freshness: $theta_"old" approx theta$], [---], [---], [✓], [✓], [✓],
  )
]

== Agentic RL

#speaker-note[
  在 Agentic RL 的时代，Rollout 任务非常繁重。
  
  沙盒、Harness 生命周期的管理也为系统增加了挑战。
]

#text(size: 20pt)[
  - Long Running
    - Typically 10min \~ 2h for SWE/Terminal tasks, based on task difficulty.
    - For long-horizon tasks, this can reach even longer.
  
    #align(center)[#image("long-horizon.png", height: 40%)]
  
  - Environment: Sandbox, Harness and Tools
    - Simple ReAct agents
    - Harnessed: Multi-agent, subagents and compact
]

== RL System

#grid(
  columns: (1fr, 1fr),
  column-gutter: 1.5em,
  row-gutter: 0.8em,
  align: (bottom, bottom, top, top),
  image("verl.png", height: 60%),
  image("slime.png", width: 100%),
  ..([*verl*],
     [*slime*],
  ).map(c => align(center, text(size: 15pt, c))),
)

Representative of modern RL frameworks. Both have a controller over:

- A *Training Engine* backed by Megatron-Core;
- An *Inference Engine* backed by vLLM or SGLang.

== RL System

Rollout becomes far more complex and important in the era of Agentic RL.


== Token-in, Token-out

== RL


// == Accuracy/Performance v.s. Sparsity

// #align(center)[#image("imgs/accuracy-performance-vs-sparsity.png", height: 50%)]

// #columns(2, gutter: 8pt)[
//   #text(16pt)[
//     - Accuracy
//       - Increases due to reduction of noise
//       - Then remains stable
//       - Eventually degrades
//   ]


//   #colbreak()

//   #text(16pt)[
//     - Computational Performance
//       - Initially grows slowly due to overheads in storing sparse structures and controlling sparse computations
//       - Then sustained growth
//   ]
// ]

// == Sparse Storage Format

// #align(center)[#image("imgs/sparse-storage.png")]

// #text(15pt)[
//   For $m lt.eq n$ elements in a space of $n$ elements:
//   - *Bitmap (BM)*: Stores a map with $n$ bits, each bit indicating whether an element is present. Requires $o=n$ additional bits.
//   - *Runlength encoding*: Stores difference of neighboring element indices. Requires $o=m ceil.l log_2 hat(d) ceil.r$, where $hat(d)$ is maximum difference of neighboring element indices.
//   - *Compressed Sparse Row (CSR)*: Represents indices in $n_c times n_r$ matrix using column and row index arrays. Requires $o=m ceil.l log_2 n_c ceil.r + n_r ceil.l log_2 m ceil.r$.
//   - *Coordinate Offset (COO)*: Stores each non-zero element together with its absolute offset. Requires $o=m ceil.l log_2 n ceil.r$.
// ]
