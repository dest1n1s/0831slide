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
    subtitle: [A Unified Paradigm for Industrialized Post-Training Systems],
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

#speaker-note[
  大家好，我是24级的硕转博葛煦旸。我先简单介绍一下我自己吧。我在之前的两年中一直在做可解释性，围绕着 mechinterp 和 SAE 做了一系列回路分析和其他的工作。然后今年五月份，我从 ICLR 回来之后，正式转来做后训练。在这几个月中我也做过很多不同的事情，一开始是做评测，后来又去做 RL 和 MOPD，最后我们现在还是准备先上量做好蒸馏，所以我转来负责 Infra，负责整体系统设计上的一些问题。

  其实我在做可解释性的时候就做过很多 Infra 的工作。我主要开发了一个 Llamascopium 框架，是一个 SAE 的训练框架，现在还挂在 OpenMOSS 的 GitHub 主页上，大家如果之后有机会训 SAE 或者其他变种的话有机会也可以试用一下。

  我今天要讲的主题是 Trace Is the Interface: A Unified Paradigm for Industrialized Post-Training Systems。主要是讲讲我对于后训练系统设计上的一些认知，它需要围绕着 Trace 来做，作为一种接口串联起所有组件。关于这个如何理解我们后面会详细展开。
]

== Common Techniques in Post-Training

#speaker-note[。
  虽然今天要讲的主题是后训练系统，但是系统是为了算法而设计的，它是我们设计系统的背景和 motivation。所以我们现在来看后训练的算法是什么。

  后训练与预训练和 mid training 都不太一样。它没有一种固定的算法，而更像一种算法的集合，通过这几种方法来达到让模型在广泛的下游任务上有更好的性能的目的。可能是只用 SFT，或者只用 RL，也可能是这几种方法一起用。

  简单讲讲这几种算法是什么，怎么从数学上理解。
]


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

In practice, SFT data pipelines often filter on rewards as well, blurring the line between SFT and rejection sampling.

== Common Techniques in Post-Training

#speaker-note[
  RL 其实也区别不大。

  GRPO 组内的差异来估计 advantage; PPO: Critic

  多数算法都会有 clip。PPO 会比较当前策略和 pi old 之间的差异，如果太高就会 clip 掉。

  OPD 本质上是一种 RL。

  所有算法都有这样一个采样的过程。这种采样我们在后训练的语境下一般叫做 Rollout，因为它其实不完全是我们从模型的分布上采出来某个 token，它还会包含一系列跟工具和环境的交互。这个 Rollout 的结果我们一般叫 Trajectory 或者 Trace。
]

- *RL*

  $
    cal(L)_"RL" (theta) = - EE_(x tilde.op cal(D)) space ro(EE_({y^i}_(i=1)^G tilde.op rq(pi_(theta_"old")) (dot | x))) [
      1/G sum_(i=1)^G sum_(t in cal(G)(y^i)) rq(m^i_t hat(A)^i_t) log pi_theta (y^i_t | x, y^i_(<t))
    ]
  $

  - Different algorithms lead to different masking $m^i_t$ and different estimates of $A^i_t$.

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
    [*Traces* at scale, long-tailed episodes], [✓], [✓], [✓], [✓], [✓],
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

  简单的场景我们可以用 Simple ReAct agent，自己带一些工具，去 parse 模型的输出然后执行。

  更多的，或者更真实的，我们的模型会跑在一个 Harness 里面
]

#text(size: 20pt)[
  - Long-running
    - Typically 10min \~ 2h for SWE/Terminal tasks, depending on task difficulty.
    - For long-horizon tasks, episodes can run even longer.
  
    #align(center)[#image("long-horizon.png", height: 40%)]
  
  - Environment: Sandbox, Harness and Tools
    - Simple ReAct agents
    - Harnessed: multi-agent, subagents and compaction
]

== RL System

#speaker-note[
  我们再反过来看我们现在的流行的系统设计。

  verl, slime

  rollout 也有被这两个系统考虑到，AgentLoop， custom rollout generation

  但是你看这边它标了 User responsibility，这边是 custom，说明 rollout 没有被系统性的考虑，实际使用时也更多是用户自己传入他们写的某个任务上的 rollout。
  
]

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

Representative modern RL frameworks. Both run a controller over:

- A *Training Engine* backed by Megatron-Core;
- An *Inference Engine* backed by vLLM or SGLang.

== RL System

#speaker-note[
  我们现在做的事情就是系统地管理整个 rollout 的流程。我们设计的这套系统是 AvaCore。

  
]

Rollout takes on a far more complex and important role in the era of Agentic RL.

#align(center)[#image("AvaCore System.png", height: 85%)]

== RL System

#speaker-note[
  Rollout Engine 这套抽象中，主要就是把 rollout 分成了 Generate 和 Reward 两部分
]

#align(center)[#image("AvaCore.png")]

== Trace

#speaker-note[
  大家肯定有调过 OpenAI 的 API，我们调 chat completions 接口的时候，我们需要给他发送我们的对话历史，与我们能够支持的 tool schema 列表。

  那么 Trace 就是这样的 messages 和 tools
]

- *Trace* = messages + tools
- ```python Client.step(trace: Trace) -> Trace```
- ```python GenerateFunction.__call__(instance: I) -> Trace```
- ```python RewardFunction.__call__(trace: Trace, reference: R) -> Reward```
- Finally, record *Trace* in database and send *Trace* for training.

== Agentic Rollout

#[
#show raw.where(block: true): set text(size: 11.5pt)
```python
class Agentic(GenerateFunction[I]):
    """Run a harness inside a sandbox against a recording proxy of ``model``."""

    model: Client
    harness: Harness = Codex()
    services: Sequence[Service] = (WSTunnel(), Mihomo())

    async def __call__(self, instance: I, *, sampling_params=None) -> Trace:
        async with (
            serving(instance.runtime_spec.runtime(), self.services) as runtime,
            OpenAIProxy(self.model, sampling_params) as proxy,  # own trace store
            runtime.reach(proxy.endpoint) as endpoint,
        ):
            await self.harness.ensure(runtime)
            await proxy.guarding(self.harness.run(
                runtime=runtime, endpoint=endpoint,
                cwd=instance.cwd, prompt=instance.prompt,
            ))
            return proxy.main_trace
```
]

== Tokens-in, Tokens-out


There are basically three forms of trace representing the same data:

- *Chat messages:* `[{"role": "user", "content": "hi"}]`
  - Public APIs and harnesses interact in this format.
- *Templated string:* `"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant"`
- *Tokens:* `[151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198]`
  - The Training Engine needs this.

Inference Engines like SGLang support both the messages and the tokens form.

== Tokens-in, Tokens-out

#text(size: 20pt)[
Transferring *chat messages* breaks token identity: 

- *Detokenize–retokenize drift:* Under BPE, it can happen that `encode(decode(y)) != y`.
- *Template pruning:* Reasoning templates erase earlier thinking spans before the last user turn.
- *Lossy re-rendering:* The tool-call arguments <-> JSON round-trip drifts whitespace.

*Solution:* Keep *tokens* as the source of truth.

- No mature solution exists for the *templated string -> chat messages* conversion.
]

== Tokens-in, Tokens-out

*Solution:* Keep *tokens* as the source of truth.

- Trace = messages + tools = ChatTrace | TokenTrace
  - TokenTrace = list of Segments, Segment = list of Tokens

== Trace Resolution in Black-box Harness

Finding which *trace* a new *tool-call observation* or *assistant response* _continues_ is trivial in a self-owned tool-call loop.

But what if the agent loop is owned by an in-sandbox, black-box harness?

== Trace Resolution in Black-box Harness

- The harness runs *inside the sandbox* and speaks a stateless OpenAI-compatible API: every request carries its whole message history, and nothing identifies the conversation it belongs to.

#align(center)[#image("proxy.png", width: 70%)]

- So the proxy has to decide, from the messages alone, *which recorded trace this request continues*.
- A resolved continuation splices onto the stored token prefix and hits the KV cache; a miss re-encodes the whole history and creates a spurious second trace.

== Trace Resolution in Black-box Harness

#align(center)[#image("trace-resolution-1.png", width: 90%)]

#align(center)[#image("trace-resolution-2.png", width: 90%)]

== Trace Resolution in Black-box Harness


A typical harness performs at least the following actions:

- *Normal:* Send *T2* after *[S1, U1, A1, T1, A2]*, waiting for *A3*;
- *Subagent:* Launch a subagent, which is a brand-new conversation *[S1', U1', A1']*
  - Depending on settings, the harness may send a *T2* representing the subagent launch immediately, or wait until the subagent finishes.
- *Compact:* Compact all previous history and send a summary.
- *Retry:* On network issues, it may also send *T1* again.

Some harnesses (like Claude Code) may also 

- Drop previous tool responses;
- *Send a different random string before the system prompt.* (cch=xxxxx)


== Trace Resolution in Black-box Harness

*Solution:*

- *Whitebox the harness:*
  - For open-source harnesses (like Codex), we can modify the source code to make it *carry session ids*.
  - Drop the in-sandbox harness and use a self-controlled loop.

- *Heuristic prefix matching*

== Stability and Observability

#text(size: 20pt)[
- A tremendous number of concurrent HTTP requests.
  - Do not use `httpx.AsyncClient`!
- *Retry* over HTTP requests, sandbox actions, and low-reward results.
- *Concurrency control* over rollout tasks and resource acquisition (like sandbox creation).
- *Error transparency:* Faithfully propagate and record all errors, and panic on unexpected ones. Traces are saved eagerly to help find the cause of errors.
- *Resource lifecycle:* Release all resources promptly. Allow cancellation of ongoing tasks.
- *Audit system:* LLM-based audit to analyze the failure mode of each trace (infra issue, model issue, reward hacking, etc.).
]

== Asynchronous RL

#text(size: 20pt)[
#set par(spacing: 0.7em)
#set block(spacing: 0.7em)
- *On-policy.* Rollout -> Train -> Rollout -> Train -> ...
- *One-step off-policy*: Batch $k + 1$ is generated by $pi_k$ while step $k$ trains; staleness is exactly one step.
- *Fully asynchronous (AReaL):* Rollout runs continuously, and training draws *traces* from it. Staleness is capped at $k$ (typically 8).
  $
    cal(L) = - EE_(y tilde.op pi_"behav") [ sum_t (pi_"prox" (y_t | dot)) / (pi_"behav" (y_t | dot)) dot min(rho_t hat(A)_t, op("clip")(rho_t, 1 - epsilon, 1 + epsilon) hat(A)_t) ], quad rho_t = (pi_theta (y_t | dot)) / (pi_"prox" (y_t | dot))
  $ #text(size: 0.75em, fill: gray)[Fu et al., _AReaL_, 2025;]
  - Pipeline RL: In-flight weight updates.
]

== More Challenges

#text(size: 20pt)[
  
*Scheduling: keep every resource busy*
- Full utilization of training GPUs, inference GPUs, and sandboxes.
- KV-cache hit rate.

*Rollback*
- Snapshot of (sandbox) environment.
- Rollback to any previous point and continue rollout there.
]



== Evaluation

No difference from normal rollout. Reward is the score.

== Ease of Use

#grid(columns: (1.1fr, 1fr), gutter: 1.5em)[
  #show raw.where(block: true): set text(size: 9.5pt)
  #set par(spacing: 0.5em)
```toml
[rollout]
concurrency = 768
retry = "unrewarded"
[data]
kind = "jsonl"
path = "super_gpqa.jsonl"
[generate]
kind = "single_turn"
instance.messages = [
  { role = "system", content = "... \\boxed{X} ..." },
  { role = "user", content = "{{ question }}\n{% for c in choices %}..." },
]
[generate.model]
endpoint = "http://localhost:30000"
name = "Qwen3.5-35B-A3B"
[reward]
kind = "exact_match"
format = '(?<=\\boxed\{)[A-Z]'
reference = "{{ answer }}"
```
][
  Start rollout in one command:
  #show raw.where(block: true): set text(size: 13pt)
  ```sh
  avacore rollout run gpqa.toml
  ```

  Evaluation is just as simple:
  ```sh
  avacore eval run http://localhost:30000 \
    --benchmarks aime25,gpqa_diamond --resume
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
