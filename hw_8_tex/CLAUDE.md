# CLAUDE.md for HW 8 LaTeX Project

This file provides guidance to Claude Code when working with the homework 8 LaTeX project.

## Project Structure

```
hw_8_tex/
├── hw_8.tex              # Main document - includes preamble and all section files
├── preamble.tex          # LaTeX packages, custom commands, and styling (copied from HW 7)
├── prob1.tex             # Final project proposal: team, title, description
├── prompts.tex           # AI interaction documentation
├── images/               # Directory for any figures
└── CLAUDE.md             # This file
```

## File Roles

- **hw_8.tex**: Main document. Uses `\input{}` to include preamble and content files.
- **preamble.tex**: Shared LaTeX preamble (identical to HW 7).
- **prob1.tex**: Project proposal — team name, member names, project title, and one-page description.
- **prompts.tex**: Documents all AI-assisted interactions (required by ME 523 course policy).

## Compilation

```bash
cd hw_8_tex
pdflatex hw_8.tex
# Run twice for proper references:
pdflatex hw_8.tex && pdflatex hw_8.tex
```

## What to Fill In

1. **prob1.tex** — replace the TODO comments with:
   - Team name
   - Project title
   - Project description (objective, modeling assumptions, flow configuration — domain size, grid, ICs/BCs, and the +X extension)

2. **prompts.tex** — append a new `\subsection*{Interaction N}` for each subsequent AI-assisted interaction.

## AI Use Documentation (prompts.tex)

Follow the same format as HW 7:

```latex
\subsection*{Interaction N}

\textbf{Prompt:}
\begin{quote}
[Your prompt text here]
\end{quote}

\textbf{Response summary:}
[Brief description of what Claude generated or modified]

\textbf{Date and Model:} [e.g., Claude Sonnet 4.6, 2026-04-14]
```

## Key Requirements for ME 523

- Team name and all member names listed
- Project title
- One-page project description (objective, modeling assumptions, flow configuration)
- AI documentation in prompts.tex (if AI was used)
- Submit as a single PDF to Canvas by Wednesday, April 15 at 11:59 PM
