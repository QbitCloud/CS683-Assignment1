// ---------------------------------------------------------------------------
// Styling and helpers for the CS683 PA-1 report.
// Content lives in report.typ; this file only defines look-and-feel.
// ---------------------------------------------------------------------------

#let accent = rgb("#1f4e79")
#let muted = rgb("#5b6470")
#let rule = rgb("#c9ced6")
#let shade = rgb("#eef1f5")

// Main document wrapper: title block + global styles.
#let report(
  title: none,
  subtitle: none,
  course: none,
  authors: (),
  date: none,
  body,
) = {
  set document(title: title, author: authors.map(a => a.name))

  set page(
    paper: "a4",
    margin: (x: 2cm, top: 2.3cm, bottom: 2cm),
    numbering: "1",
    number-align: center,
  )

  set text(font: ("New Computer Modern", "Libertinus Serif"), size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em, spacing: 1.05em)

  // Headings
  set heading(numbering: "1.1")
  show heading: set block(above: 1.4em, below: 0.75em)
  show heading.where(level: 1): set text(size: 13.5pt, weight: 700, fill: accent)
  show heading.where(level: 2): set text(size: 11.5pt, weight: 700)
  show heading.where(level: 3): set text(size: 10.5pt, weight: 700, style: "italic")

  // Code
  show raw: set text(font: ("JetBrains Mono", "DejaVu Sans Mono"), size: 8.8pt)
  show raw.where(block: false): box.with(
    fill: rgb("#f0f2f5"),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 3pt),
    radius: 2pt,
  )
  show raw.where(block: true): it => block(
    width: 100%,
    fill: rgb("#f7f8fa"),
    stroke: (left: 2pt + accent),
    inset: (x: 10pt, y: 8pt),
    radius: 2pt,
    it,
  )

  // Figures and tables
  set figure(gap: 0.9em)
  show figure.caption: set text(size: 9pt, fill: muted)
  show figure: set block(above: 1.4em, below: 1.4em)

  // Links
  show link: set text(fill: accent)

  // ---- Title block ----
  block(width: 100%)[
    #set align(center)
    #if course != none {
      text(size: 10pt, fill: muted, tracking: 0.08em, upper(course))
      v(0.4em, weak: true)
    }
    #text(size: 19pt, weight: 700, title)
    #if subtitle != none {
      v(0.35em, weak: true)
      text(size: 12pt, fill: muted, subtitle)
    }
    #v(0.9em, weak: true)
    #line(length: 100%, stroke: 0.8pt + rule)
    #v(0.6em, weak: true)
    #grid(
      columns: (1fr,) * calc.max(authors.len(), 1),
      column-gutter: 1.5em,
      ..authors.map(a => [
        #text(weight: 600, a.name) \
        #text(size: 9pt, fill: muted, a.roll)
      ])
    )
    #if date != none {
      v(0.5em, weak: true)
      text(size: 9pt, fill: muted, date)
    }
  ]
  v(1.2em)

  body
}

// Booktabs-style results table.
// cols: track sizes, header: array of header cells, body cells as varargs.
#let restable(cols, header, ..body, cellalign: none) = {
  let cells = body.pos()
  let ncol = header.len()
  let nrow = calc.ceil(cells.len() / ncol)
  let al = if cellalign == none {
    (x, y) => if x == 0 { left + horizon } else { right + horizon }
  } else { cellalign }
  align(center, table(
    columns: cols,
    align: al,
    inset: (x: 8pt, y: 5.5pt),
    fill: (_, y) => if y == 0 { shade },
    stroke: (x, y) => (
      top: if y == 0 { 0.9pt + black } else if y == 1 { 0.5pt + rule } else { 0pt },
      bottom: if y == nrow { 0.9pt + black } else { 0pt },
    ),
    ..header.map(h => strong(h)),
    ..cells,
  ))
}

// A real plot, once the PNG/SVG exists in figures/.
#let plot(path, caption, width: 85%) = figure(
  image(path, width: width),
  caption: caption,
)

// Stand-in for a plot that has not been generated yet.
#let plotsoon(caption, height: 6cm) = figure(
  rect(
    width: 100%,
    height: height,
    fill: rgb("#fafbfc"),
    stroke: (paint: rule, thickness: 0.8pt, dash: "dashed"),
    radius: 3pt,
  )[
    #set align(center + horizon)
    #text(size: 9pt, fill: muted)[plot pending]
  ],
  caption: caption,
)

// Inline marker for anything still missing.
#let todo(body) = text(fill: rgb("#b3261e"), weight: 600)[#box[[TODO: #body]]]
