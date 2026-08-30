#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#set text(font: (
  (name: "libertinus serif", covers: "latin-in-cjk"),
  "Noto Serif CJK SC"
))
#set par(justify: true)

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.title,
  config-info(
    title: [Sparsity in Deep Learning],
    subtitle: [Pruning and growth for efficient inference and training in neural networks],
    author: [
      #box[Torsten Hoefler] #h(1em) #box[Dan Alistarh] #h(1em) #box[Tal Ben-Nun]
      #h(1em) #box[Nikoli Dryden] #h(1em) #box[Alexandra Peste]
    ],
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

#[
  #set text(size: 22pt)
  #title-slide(title: text(size: 1.5em)[Sparsity in Deep Learning])
]

== Model Compression Techniques

#align(center)[#image("imgs/model-compression.png", height: 84%)]

== Accuracy/Performance v.s. Sparsity

#align(center)[#image("imgs/accuracy-performance-vs-sparsity.png", height: 50%)]

#columns(2, gutter: 8pt)[
  #text(16pt)[
    - Accuracy
      - Increases due to reduction of noise
      - Then remains stable
      - Eventually degrades
  ]


  #colbreak()

  #text(16pt)[
    - Computational Performance
      - Initially grows slowly due to overheads in storing sparse structures and controlling sparse computations
      - Then sustained growth
  ]
]

== Sparse Storage Format

#align(center)[#image("imgs/sparse-storage.png")]

#text(15pt)[
  For $m lt.eq n$ elements in a space of $n$ elements:
  - *Bitmap (BM)*: Stores a map with $n$ bits, each bit indicating whether an element is present. Requires $o=n$ additional bits.
  - *Runlength encoding*: Stores difference of neighboring element indices. Requires $o=m ceil.l log_2 hat(d) ceil.r$, where $hat(d)$ is maximum difference of neighboring element indices.
  - *Compressed Sparse Row (CSR)*: Represents indices in $n_c times n_r$ matrix using column and row index arrays. Requires $o=m ceil.l log_2 n_c ceil.r + n_r ceil.l log_2 m ceil.r$.
  - *Coordinate Offset (COO)*: Stores each non-zero element together with its absolute offset. Requires $o=m ceil.l log_2 n ceil.r$.
]
