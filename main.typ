#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#set text(font: (
  (name: "libertinus serif", covers: "latin-in-cjk"),
  "Noto Serif CJK SC"
))
#set par(justify: true)

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  align: top,
  config-info(
    title: [Trace Is the Interface],
    subtitle: [A Unified Paradigm in Industrialized Post-Training System],
    author: [
      #box[葛煦旸]
    ],
    date: datetime.today(),
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

== Background: Post-Training Is Rollout-Bound

- Every technique in use consumes *samples from a model*: SFT on synthesized or distilled data, rejection fine-tuning, RL, on-policy distillation.
- Rollout dominates the step: 50–80% of RL step time (DORA, 2026), over 80% on agentic workloads (Heddle, 2026).
- A sample is no longer one response: multi-turn, tool calls, sandboxes, third-party harnesses (Claude Code, Codex, and others); one episode runs seconds to an hour.
- Consumers disagree on what a sample *is*: text for SFT, token ids and masks for training, engine log-probs for RL, teacher log-probs for OPD, a reward for RFT and RL.

== Motivation: One Rollout, Many Stacks

#grid(columns: (1fr, 1fr), gutter: 1.5em)[
  *State of practice*
  - Each technique ships its own stack: eval harness, data-synthesis pipeline, RL trainer with a built-in rollout worker, distillation script.
  - Each re-implements generation, chat-template rendering, the tool loop and storage — each with its own answer to _what a sample is_.
][
  *What it costs*
  - Re-tokenised text silently turns on-policy training off-policy (Yao et al., 2025; Miles, 2026).
  - The same benchmark, evaluated three ways, gives three numbers.
  - A harness written for evaluation cannot be trained on.
  - At around 10k in-flight requests the Python driver, not the inference server, is the ceiling.
]

== Thesis: The Trace Is the Interface

- Record once, at the LLM endpoint: the exact token ids the engine consumed and produced, the generated-position mask, log-probs, and the messages they render to.
- Every consumer reads the same artifact — evaluation scores it, SFT trains on it, RFT filters it, RL reuses its log-probs, OPD has a teacher score it.

#v(1em)
#let bx(body) = box(inset: (x: 0.8em, y: 0.5em), radius: 4pt, stroke: 1pt + rgb("#1d6fa5"), body)
#align(center)[
  #bx[Harness] #h(0.6em) #sym.arrow.r #h(0.6em) #bx[Endpoint proxy] #h(0.6em) #sym.arrow.r #h(0.6em) #bx[*Trace*] #h(0.6em) #sym.arrow.r #h(0.6em) #bx[Eval · SFT · RFT · RL · OPD]
]
#v(0.8em)
- The stacks collapse into one rollout layer; the differences between techniques become differences in what they *read* from the trace.

== Common Techniques in Post-Training

#let ro(body) = text(fill: rgb("#1e6b86"), body)          // produced by rollout
#let rq(body) = text(fill: rgb("#b4471b"), body)          // new requirement vs. the previous loss


- *SFT / Distillation*

$
  cal(L)_"SFT" (theta) = - EE_(x tilde.op cal(D)) space EE_(ro(y) ro(tilde.op) rq(pi_"teacher") (dot | x)) [ sum_(t in cal(G)(y)) log pi_theta (y_t | x, y_(<t)) ]
$

- *RFT (rejection sampling)*

$
  cal(L)_"RFT" (theta) = - EE_(x tilde.op cal(D)) space EE_(ro(y) ro(tilde.op) rq(pi_(theta_"old")) (dot | x)) [ rq(bb(1)[r(x, y) >= tau]) sum_(t in cal(G)(y)) log pi_theta (y_t | x, y_(<t)) ]
$

- *RL*

$
  cal(L)_"RL" (theta) = - EE_(x tilde.op cal(D)) space EE_({ro(y^i)}_(i=1)^G ro(tilde.op) rq(pi_(theta_"old")) (dot | x)) [
    1/G sum_(i=1)^G sum_(t in cal(G)(y^i)) rq(m^i_t A^i_t) log pi_theta (y^i_t | x, y^i_(<t))
  ]
$


== On-Policy Distillation

- *OPD* — the student samples, the teacher scores every token it sampled

$
  hat(A)_t = op("sg")[log pi_"teacher" (y_t | x, y_(<t)) - log pi_(theta_"old") (y_t | x, y_(<t))]
$
$
  cal(L)_"OPD" (theta) = - EE_(x tilde.op cal(D)) space EE_(ro(y) ro(tilde.op) rq(pi_(theta_"old")) (dot | x)) [ sum_(t in cal(G)(y)) rq(hat(A)_t) log pi_theta (y_t | x, y_(<t)) ]
$

#[
#set text(size: 0.8em)
#set par(justify: false)
#grid(columns: (1fr, 1fr), gutter: 1.5em)[
  *SFT* — off-policy
  - $y tilde.op pi_"teacher"$: learns on states the teacher visits; errors compound once the student leaves them
  - target = the teacher token; $approx nabla op("KL")(pi_"teacher" || pi_theta)$, mass-covering
  - rollout: teacher generation, text suffices
][
  *OPD* — on-policy
  - $y tilde.op pi_theta$: learns on its own states
  - signal = teacher log-prob of the student token, dense; $= nabla op("KL")(pi_theta || pi_"teacher")$, mode-seeking — RL whose reward needs no verifier
  - rollout: student generation *and* a teacher scoring pass on the exact token ids
]
]
#text(size: 0.8em, fill: gray)[Agarwal et al., GKD, 2023; Thinking Machines, _On-Policy Distillation_, 2025: 9–30× cheaper than SFT distillation to the same reasoning score]

== Multi-Teacher On-Policy Distillation

- *MOPD* — one student, a frozen expert teacher $phi_d$ per domain, each trained by RL on its own task; the domain $d(x)$ of the prompt picks the teacher

$
  hat(A)_t = op("sg")[log pi_(phi_(d(x))) (y_t | x, y_(<t)) - log pi_(theta_"old") (y_t | x, y_(<t))] + alpha hat(A)_"ORM"
$
$
  cal(L)_"MOPD" (theta) = - EE_(x tilde.op cal(D)) space EE_(ro(y) ro(tilde.op) rq(pi_(theta_"old")) (dot | x)) [ sum_(t in cal(G)(y)) rq(hat(A)_t) log pi_theta (y_t | x, y_(<t)) ]
$

- Teachers are developed in parallel and never merged; the student integrates them without the see-saw of sequential multi-domain RL
- Rollout adds to OPD: one scoring endpoint per teacher, a router from prompt to teacher, and with $alpha > 0$ the verifier as well; the trace must record which teacher scored it

#text(size: 0.7em, fill: gray)[MiMo-V2-Flash (Xiaomi, 2026), with the $alpha hat(A)_"ORM"$ term; Ma et al., _MOPD_, 2026: pure routing, $alpha = 0$]

== What Each Technique Needs from Rollout

#set text(size: 15pt)
#set par(justify: false)
#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr, 1.4fr),
  align: (left, center, center, center, center, center),
  stroke: (x: none, y: 0.4pt + luma(160)),
  inset: 6pt,
  table.header([], [*SFT*], [*RFT*], [*RL*], [*OPD*], [*MOPD*]),
  [Generation at scale, long-tailed episodes], [✓], [✓], [✓], [✓], [✓],
  [Token ids in the student tokenizer, generated-position mask], [✓], [✓], [✓], [✓], [✓],
  [Samples from the policy under training], [teacher], [✓], [✓], [✓], [✓],
  [Reward from a verifier or environment], [—], [✓], [✓], [—], [$alpha > 0$],
  [Engine log-probs on the exact sampled tokens], [—], [—], [✓], [✓], [✓],
  [Log-probs of a second model on those tokens (scoring pass)], [—], [—], [—], [1 teacher], [$N$ teachers + router],
  [Policy freshness: $theta_"old" approx theta$], [—], [—], [✓], [✓], [✓],
)


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
