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
    title: [Trace Is the Interface],
    subtitle: [A Unified System in Post-Training],
    author: [
      #box[葛煦旸]
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
      #info.author
      #if info.institution != none [ \ #text(size: 0.85em, fill: self.colors.neutral-lightest.darken(20%), info.institution) ]
      #if info.date != none [ \ #text(size: 0.85em, fill: self.colors.neutral-lightest.darken(20%), utils.display-info-date(self)) ]
    ]
  }
  touying-slide(self: self, body)
})

#cover-slide

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
