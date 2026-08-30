#import "@preview/touying:0.6.1": *
#import themes.simple: *

#set text(font: (
  (name: "libertinus serif", covers: "latin-in-cjk"),
  "Noto Serif CJK SC"
))

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [],
)

#title-slide[
  = Sparsity in Deep Learning
  == Pruning and growth for efficient inference and training in neural networks
  #v(2em)

  Torsten Hoefler #h(1em) Dan Alistarh #h(1em) Tal Ben-Nun 
  #h(1em) Nikoli Dryden #h(1em) Alexandra Peste
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
