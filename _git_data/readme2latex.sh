#!/bin/bash

python -m readme2tex --rerender --valign --output ./readme-latex-output.md --readme ./readme-latex.md --svgdir ./svgs

# python -m readme2tex --nocdn --rerender --valign --output ../readme-latex-output.md --readme ./readme-latex.md --svgdir ./svgs

# python -m readme2tex --nocdn --rerender --valign --pngtrick --output ./readme-latex-output.md --readme ./readme-latex.md --svgdir ./svgs