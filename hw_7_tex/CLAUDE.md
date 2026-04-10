# CLAUDE.md for HW 7 LaTeX Project

This file provides guidance to Claude Code when working with the homework 7 LaTeX project.

## Project Structure

```
hw_7_tex/
├── hw_7.tex              # Main document - includes preamble and all problem/section files
├── preamble.tex          # LaTeX packages, custom commands, and styling
├── prob1.tex             # Problem 1: 2D Navier-Stokes Solver for the Taylor-Green Vortex
├── prompts.tex           # AI interaction documentation (template provided)
├── code.tex              # Source code listings (template provided)
├── images/               # Directory for plot images
└── CLAUDE.md             # This file
```

## File Roles

- **hw_7.tex**: Main document. Uses `\input{}` to include preamble and all other content files.
- **preamble.tex**: Shared LaTeX preamble with packages and custom commands. Copied from HW 6.
- **prob1.tex**: Problem statement and blank solution sections for parts (a)-(f).
- **prompts.tex**: Documents all AI-assisted interactions (when/how AI was used, prompts given, summaries of responses).
- **code.tex**: Lists the source code files using `\lstinputlisting{}`.

## Compilation

Compile the main document:

```bash
cd hw_7_tex
pdflatex hw_7.tex
# Run twice for proper references:
pdflatex hw_7.tex && pdflatex hw_7.tex
```

The compiled PDF (hw_7.pdf) will be created in the hw_7_tex/ directory.

## AI Use Documentation (prompts.tex)

The `prompts.tex` file documents your interactions with AI tools (e.g., Claude). This is **required** for ME 523.

### How to Fill Out prompts.tex

When you use Claude or other AI tools during this homework, Claude will help you automatically append interactions to `prompts.tex`. Here's how:

1. **First time using AI**: Ask Claude to update `prompts.tex` with your first interaction.
   - Example: "Document this interaction in prompts.tex"
   - Claude will add an Interaction 1 section with your prompt and response summary.

2. **Subsequent interactions**: Ask Claude to append to `prompts.tex`.
   - Example: "Add this interaction to prompts.tex as Interaction 2"
   - Claude will create a new subsection with the next interaction number.

3. **Content to include in each interaction**:
   - **Prompt**: The exact text you gave to Claude (or a typeset summary)
   - **Response summary**: A brief description of what Claude generated or modified
   - **Date**: Timestamp if relevant
   - **Model**: Which Claude model was used (e.g., Claude Opus 4.6)

### Format Template

Each interaction should follow this format:

```latex
\subsection*{Interaction N}

\textbf{Prompt:}
\begin{quote}
[Your prompt text here]
\end{quote}

\textbf{Response summary:}
[Brief description of what Claude generated or modified]

\textbf{Date and Model:}
[e.g., Claude Opus 4.6, 2026-04-01]
```

## Source Code (code.tex)

The `code.tex` file uses `\lstinputlisting{}` to include your Python/Matlab source code files.

### How to Use code.tex

1. **Add your source files**: Place your code files (e.g., `HW7.py`) in the `hw_7_tex/` directory or a sibling `code/` directory.

2. **Uncomment and modify**: In `code.tex`, uncomment the `\lstinputlisting{}` lines and specify your filenames:
   ```latex
   \lstinputlisting{HW7.py}
   ```

3. **Multiple files**: Use multiple `\lstinputlisting{}` commands, separated by `\pagebreak` if desired.

## Workflow Summary

1. **Implement the solver** in Python:
   - Staggered grid setup on $[0, 2\pi]^2$ with $64 \times 64$ cells
   - Fractional step method (Kim & Moin, 1985)
   - Adams-Bashforth for convective terms, Crank-Nicolson for viscous terms
   - ADI splitting with Thomas algorithm + Sherman-Morrison for periodic BCs
   - Multigrid for the pressure Poisson equation
   - Taylor-Green vortex initial condition
   - CFL-based time stepping, simulate to $t = 10$

2. **Generate results** for parts (a)-(e):
   - (a) Description of numerical method
   - (b) Divergence norm vs time plot
   - (c) Kinetic energy vs time, compared to analytical decay
   - (d) Contour plots of velocity magnitude and pressure
   - (e) Report: time step, tolerances, iterations, cost, comparison to analytical solution

3. **Save your code** as `.py` files.

4. **Use Claude** whenever you need help.

5. **Document AI interactions** by asking Claude to update `prompts.tex`.

6. **Include code** by uncommenting lines in `code.tex`.

7. **Compile** the full document with `pdflatex hw_7.tex`.

8. **Submit** the resulting PDF.

## Key Requirements for ME 523

- ✅ **Solution sections** in prob1.tex (answers to all questions a-f)
- ✅ **Source code** included via code.tex
- ✅ **AI documentation** in prompts.tex (if you used AI; required by course policy)
- ✅ **Proper attribution** of team members and AI interactions

## Problem Overview

**Problem 1: 2D Navier-Stokes Solver for the Taylor-Green Vortex**

Extend the pressure projection solver from HW6 to a full incompressible Navier-Stokes solver. Key components:

- **Governing equations**: 2D incompressible Navier-Stokes (continuity + momentum)
- **Grid**: Staggered (MAC) arrangement on $[0, 2\pi]^2$, $64 \times 64$, periodic BCs
- **Initial condition**: Taylor-Green vortex ($u = \sin x \cos y$, $v = -\cos x \sin y$)
- **Time integration**: Fractional step with AB2 (convective) + CN (viscous)
- **Viscous solve**: ADI with Thomas algorithm + Sherman-Morrison for periodicity
- **Pressure solve**: Multigrid for the Poisson equation
- **CFL**: $u_{\max}\Delta t / \Delta x \lesssim 0.5$
- **Final time**: $t = 10$

## Tips

- Run `pdflatex hw_7.tex` twice to ensure proper cross-references and page numbers.
- The staggered grid convention: $u(i,j) = u_{i-1/2,j}$, $v(i,j) = v_{i,j-1/2}$, $p(i,j) = p_{i,j}$.
- The Sherman-Morrison formula handles the periodic wrap-around in the Thomas algorithm tridiagonal solve.
- The analytical solution for the Taylor-Green vortex decays as $e^{-2t/\mathrm{Re}}$.
- If LaTeX compilation fails, check that all referenced files exist and are spelled correctly.
