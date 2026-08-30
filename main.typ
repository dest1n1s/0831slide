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

Rollout claims a far more complex and important position in the era of Agentic RL.

#align(center)[#image("AvaCore System.png", height: 85%)]

== RL System

#align(center)[#image("AvaCore.png")]

== Tokens-in, Tokens-out


There\'re basically 3 forms of trace representing the same data:

- *Chat messages:* `[{"role": "user", "content": "hi"}]`
  - Public API/Harness interacts in this format.
- *Templated string:* `"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant"`
- *Tokens:* `[151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198]`
  - Training Engine need this.

Inference Engine like SGLang support both messages and tokens form.

== Tokens-in, Tokens-out

#text(size: 16pt)[
Going through text breaks the token stream the policy actually sampled — silently: the text reads the same, the log-probs belong to a different sequence, and on-policy training has become off-policy.

- *Detokenize–retokenize drift:* BPE admits many encodings of one string and the sample need not be the canonical one, so `encode(decode(y)) != y`.
- *Template pruning:* reasoning templates erase earlier thinking spans before the last user turn — training sees a history the model never conditioned on.
- *Lossy re-rendering:* tool-call arguments round-trip through JSON; whitespace and key order drift, and so do the ids.

*Invariant:* prompt + response of turn $n - 1$ is a bit-exact prefix of the prompt of turn $n$. AvaCore keeps the trace as tokens:

- `TokenTrace` is a sequence of `ChatSegment`s — `tokens`, `is_generated`, `log_probs`, and the `message_attrs` the template drops (tool-call ids, names, reasoning signature); `tokens`, `loss_mask`, `log_probs` are flattened views, never re-encoded.
- Extending renders only the appended messages behind a guard prefix and splices the delta onto the stored ids; a truncated turn gets its missing terminator repaired on the next extend.
- Messages are *derived*: decode $arrow.r$ `Parser.parse` $arrow.r$ `restore(message_attrs)`, asserted equal to what was stored; `trace.tokens == encode(trace.template)` holds after a multi-turn tool rollout.
]

== Trace Resolution in Black-box Harness

Finding what *trace* a new *tool-call observation* or *assistant response* _continues_ is trivial in self-owned tool-call loop.

But what if the agent loop is owned by in-sandbox black-boxed harness?

== Trace Resolution in Black-box Harness

- The harness runs *inside the sandbox* and speaks a stateless OpenAI-compatible API: every request carries its whole message history, and nothing identifies the conversation it belongs to.
- AvaCore fronts the inference engine with one `OpenAIProxy` and one `TraceStore` per rollout. Per request:

  #align(center)[resolve $arrow.r$ extend the matched trace $arrow.r$ `client.step` (SGLang, tokens) $arrow.r$ commit]

- So the proxy has to decide, from the messages alone, *which recorded trace this request continues*.
- The answer decides whether TITO survives the turn: a resolved continuation splices onto the stored token prefix and hits the KV cache; a miss re-encodes the whole history from text.

== Trace Resolution in Black-box Harness

#align(center)[#image("trace-resolution-1.png", width: 90%)]

#align(center)[#image("trace-resolution-2.png", width: 90%)]

== Trace Resolution in Black-box Harness


Typical harnesses will do at least all of the following actions:

- *Normal:* Send *T2* after *[S1, U1, A1, T1, A2]*, waiting for *A3*;
- *Subagent:* Launch a subagent, which is a brand-new conversation *[S1', U1', A1']*
  - Based on settings, the harness could send a *T2* representing the launch of the subagent immediately, or wait till the subagent finishes.
- *Compact:* Compact all previous history, and send a summary.
- *Retry:* When facing network issues, it may also send *T1* again.

Some harnesses (like Claude Code) may also 

- Drop previous tool responses;
- *Send different random string before system prompt.* (cch=xxxxx)


== Trace Resolution in Black-box Harness

#text(size: 15pt)[
- Two messages *match* when role and `tool_call_id` agree and either their tool-call ids are equal or their text is — exactly, or within a `SequenceMatcher` ratio of 0.95.
- A request *continues* a trace when every stored position matches, in order; there is no realignment.
- `resolve` tries every stored trace *and its one-step rollback* `query()` — the snapshot before the last generation — and takes the longest match; none means a new trace.
- Each request commits twice, after its own messages are appended and after the reply, under one lock per proxy.

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: (x: none, y: 0.4pt + luma(160)),
  inset: 5pt,
  [*Normal*], [the full trace matches $arrow.r$ extend; new tokens splice onto the stored prefix],
  [*Retry*], [matches the committed request, or its rollback if a reply was already produced $arrow.r$ regenerate once; the trace is replaced in place, never duplicated],
  [*Subagent*], [no stored prefix $arrow.r$ a new trace; the main trace is the first one],
  [*Compact*], [the summary matches nothing $arrow.r$ a new trace],
  [*Dropped tool response*], [positions shift $arrow.r$ a new trace, re-encoded from text: TITO is lost on that branch],
  [*Random `cch=` prefix*], [the system message stays within tolerance $arrow.r$ normal continuation],
)
]

== Stability

#text(size: 16pt)[
- *Transport retry* — `HTTPRetry`: timeouts, connection errors, 408 / 409 / 425 / 429 / 5xx; exponential backoff 0.5 s $arrow.r$ 30 s; applied once at client construction, so the hot path has no policy branches.
- *Failure carries its evidence* — exhausted retries raise `RetryExceeded`, an `ExceptionGroup` of every attempt: the sequence says what the server was doing when no single error would.
- *Error budget* — `tolerating(stream, tolerance)`: a ratio per error type; crossing it raises `ErrorBudgetExceeded` and stops the run instead of quietly finishing with a half-empty dataset.
- *Rollout retry policy* — `retry = "errors"` or `{kind = "unrewarded", min_score = 1.0}`; `recoverable` lists the exceptions that mean retry, everything else is a failed rollout.
- *Lifecycle* — a sandbox `Runtime` is a context manager: `start`, graceful `stop`, unconditional `cleanup`; services (harness, tunnel) stack in an `AsyncExitStack`; a proxy failure is an event every awaiter of that rollout sees.

```
RetryExceeded: 4 attempts
  +-- 1: ConnectError        connection refused
  +-- 2: TimeoutException    read timeout after 30 s
  +-- 3: HttpStatusError     503 Service Unavailable
  +-- 4: HttpStatusError     503 Service Unavailable
```
]

== Observability

#text(size: 16pt)[
- *Every rollout has a status* — `running / completed / verifying / rollout_error / verification_error`; the run goes `pending → running → recovering → finished | stopped | failed`.
- *Partial updates* — a progress tick writes `status`, `score`, `samples` only; `size`, `trials`, `sampling_params`, `config` are set once by `run()` and never clobbered.
- *Resume* — `generate.resuming(traces)` short-circuits every instance that already has a trace; `--resume` keeps the scored rollouts and retries the failed ones.
- *Progress you can act on* — a periodic tick logs finished / expected, failures, score and the running-age p10 / p50 / p90 / p99: the long tail is visible while it happens.
- *Harness logs* — every tool call and its output are recorded per rollout.
- *SQL over traces* — `trace` and `reward` are Postgres composite types, not JSON:

```sql
SELECT (reward).score, ((trace).messages)[3].reasoning_content
FROM samples
WHERE run_id = 42 AND (reward).score < 0.5;
```
]

== Asynchronous RL

#text(size: 14pt)[
#set par(spacing: 0.7em)
#set block(spacing: 0.7em)
- *On-policy, synchronous* — generate a batch, train, sync weights. Every GPU waits for the slowest episode; at 10 min – 2 h per episode most of the step is idle.
- *One-step off-policy* — batch $k + 1$ is generated by $pi_k$ while step $k$ trains; staleness is exactly one step and the PPO ratio $pi_theta \/ pi_"old"$ absorbs it.
- *Fully asynchronous (AReaL)* — workers never stop; weights reload mid-generation, so one trajectory spans several policy versions. Staleness is bounded ($<= eta$) and the objective is *decoupled*: a proximal policy anchors the clipping, the behaviour policy is only an importance weight.
  $
    cal(L) = - EE_(y tilde.op pi_"behav") [ sum_t (pi_"prox" (y_t | dot)) / (pi_"behav" (y_t | dot)) dot min(rho_t hat(A)_t, op("clip")(rho_t, 1 - epsilon, 1 + epsilon) hat(A)_t) ], quad rho_t = (pi_theta (y_t | dot)) / (pi_"prox" (y_t | dot))
  $
- *Even on-policy is off-policy* — inference and training engines compute $pi_"old"$ differently (kernels, precision, batching); TIS reweights each token by their truncated ratio:
  $
    w_t = min((pi_"train" (y_t | dot)) / (pi_"infer" (y_t | dot)), C), quad C approx 2
  $
- *What rollout must supply* — engine log-probs on the exact sampled tokens (TIS, every regime); the weight version behind every generated segment (async); a stream, not a batch. AvaCore records both per step and yields rollouts as they finish.

#text(size: 0.75em, fill: gray)[Fu et al., _AReaL_, 2025; Yao et al., _Your Efficient RL Framework Secretly Brings You Off-Policy RL Training_, 2025]
]

== More Challenges

#text(size: 16pt)[
*Resolution edge cases*
- A dropped tool response or a compaction breaks the positional prefix: the branch restarts from text and loses its token prefix — recoverable only by matching at the token level instead of the message level.
- _The first trace is the main one_ is a heuristic: OpenCode emits a title-generation conversation before the real one.
- Harness nondeterminism (random `cch=` prefixes) is absorbed by tolerance today; every such tolerance trades resolution precision against KV-cache reuse.

*Scheduling: every resource busy*
- Three pools on different clocks: training GPUs (steps), inference GPUs (tokens), sandboxes (wall-clock, CPU, network). An episode holds its sandbox for the whole 10 min – 2 h, including the time it waits on the model.
- Verification competes with rollout for the same sandboxes: prewarming, reuse and a verification queue decide whether inference or the sandbox pool is the bottleneck.
- The long tail blocks the batch on its slowest episode; partial rollouts, interruptible generation and staleness-aware admission turn that idle time into throughput.
]

== Convenient Usage

#grid(columns: (1.1fr, 1fr), gutter: 1.5em)[
  #show raw.where(block: true): set text(size: 9.5pt)
  #set par(spacing: 0.5em)
```toml
[run]
kind = "distillation"
collection = "super_gpqa"
resume = true
[rollout]
concurrency = 1200
retry = "unrewarded"
[data]
kind = "jsonl"
path = "/mnt/data/super_gpqa.jsonl"
[generate]
kind = "single_turn"
instance.messages = [
  { role = "system", content = "... \\boxed{X} ..." },
  { role = "user", content = "{{ question }}\n{% for c in choices %}..." },
]
[generate.model]
endpoint = "http://localhost:30001"
name = "Qwen/Qwen3.5-35B-A3B"
concurrency = 1000
[reward]
kind = "exact_match"
format = '(?<=\\boxed\{)[A-Z]'
reference = "{{ answer }}"
```
][
  #set text(size: 13.5pt)
  - `kind` names a registered class; the rest of the table is the fields of that class — there is no separate config schema.
  - Strings are Jinja templates over the row; `{field = "answer"}` passes a value untouched; `{call = "file.py:fn"}` runs code. Logic never lives in TOML.
  - `retry = "unrewarded"` re-rolls instances that produced no score; `resume = true` reopens the run.
  - `[generate.model]` is one flat table: endpoint, retries, sampling params and the TokioPost `concurrency`.
  - `${ENV}` anywhere in a string; the recorded config never contains a secret.

  #v(0.6em)
  #show raw.where(block: true): set text(size: 11pt)
  ```sh
  avacore rollout run gpqa.toml \
    --postgres $POSTGRES --set run.trials=4
  ```
]

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
