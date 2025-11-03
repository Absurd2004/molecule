#!/usr/bin/env python
# coding=utf-8
"""Check whether a SMILES string is valid and save its 2D depiction."""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore
from PIL import Image
import io

import utils.chem as uc  # pylint: disable=import-error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a SMILES string and store a 2D depiction image."
    )
    parser.add_argument(
        "--smiles",
        "-s",
        default="[*:0]C1CC(O)(C(=O)CO)Cc2c(O)c3c(c(O)c21)C(=O)c1c(OC)cccc1C3=O",
        help="SMILES string to validate.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/sample/smiles.svg",
        help="Path to the output image file (PNG or SVG).",
    )
    parser.add_argument(
        "--image-size",
        "-i",
        type=int,
        default=800,
        help="Side length (pixels) for the generated image (after downsampling).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI metadata to embed in the output image (PNG only).",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=3.0,
        help=(
            "Factor to oversample the drawing before resizing back to image-size. "
            "Values >1 yield crisper PNGs."
        ),
    )
    parser.add_argument(
        "--bg-color",
        default="255,255,255",
        help="Background color as 'R,G,B' in 0-255 range (default: 255,255,255).",
    )
    return parser.parse_args()


def _parse_rgb(color: str) -> Tuple[Tuple[float, float, float, float], Tuple[int, int, int, int]]:
    """Parse `R,G,B` or `R,G,B,A` where R/G/B are 0-255 ints and A is either 0-255 int or 0-1 float.

    Returns ((r_f, g_f, b_f, a_f), (r_i, g_i, b_i, a_i)) where floats are 0-1 and ints are 0-255.
    """
    parts = [part.strip() for part in color.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError("Expected three or four comma-separated values for RGB[A]")

    try:
        integers = [int(parts[i]) for i in range(min(3, len(parts)))]
    except ValueError as exc:
        raise ValueError("RGB values must be integers for R,G,B") from exc

    if any(component < 0 or component > 255 for component in integers):
        raise ValueError("RGB components must be between 0 and 255")

    # alpha handling
    if len(parts) == 4:
        # allow alpha as float 0-1 or int 0-255
        a_part = parts[3]
        try:
            if "." in a_part:
                a_f = float(a_part)
                if not (0.0 <= a_f <= 1.0):
                    raise ValueError("Alpha float must be between 0.0 and 1.0")
                a_i = int(round(a_f * 255))
            else:
                a_i = int(a_part)
                if not (0 <= a_i <= 255):
                    raise ValueError("Alpha int must be between 0 and 255")
                a_f = a_i / 255.0
        except ValueError as exc:
            raise ValueError("Alpha must be int 0-255 or float 0.0-1.0") from exc
    else:
        a_i = 255
        a_f = 1.0

    floats = tuple(component / 255.0 for component in integers) + (a_f,)
    ints = tuple(integers) + (a_i,)
    return floats, ints


def main() -> int:
    args = parse_args()

    smiles_input = (args.smiles or "").strip()
    if not smiles_input:
        print("No SMILES string provided.", file=sys.stderr)
        return 1

    try:
        bg_float, bg_int = _parse_rgb(args.bg_color)
    except ValueError as exc:
        print(f"Invalid background color: {exc}", file=sys.stderr)
        return 1

    # create output path early so we can switch suffix for grid
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # allow multiple smiles separated by '|'
    parts: List[str] = [p.strip() for p in smiles_input.split("|") if p.strip()]

    def _render_single_png(mol_obj: object, render_size: tuple, target_size: tuple) -> Image.Image:
        drawer = rdMolDraw2D.MolDraw2DCairo(render_size[0], render_size[1])
        opts = drawer.drawOptions()
        opts.bgColor = tuple(bg_float)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_obj)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        if render_size != target_size:
            img = img.resize(target_size, resample=Image.LANCZOS)
        if bg_int[3] < 255:
            back = Image.new("RGBA", target_size, tuple(bg_int))
            back.paste(img, (0, 0), img)
            return back.convert("RGB")
        return img.convert("RGB")

    # if user provided exactly 4 SMILES separated by '|', render a 2x2 grid
    if len(parts) == 4:
        mols = []
        for smi in parts:
            mol_obj = uc.to_mol(smi)
            if mol_obj is None:
                print(f"Invalid SMILES in list: {smi}", file=sys.stderr)
                return 1
            mols.append(mol_obj)

        target_size = (args.image_size, args.image_size)
        suffix = output_path.suffix.lower()

        if suffix == ".svg":
            drawer = rdMolDraw2D.MolDraw2DSVG(target_size[0], target_size[1], target_size[0] // 2, target_size[1] // 2)
            opts = drawer.drawOptions()
            opts.setBackgroundColour(bg_float) 
            drawer.DrawMolecules(mols)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            output_path.write_text(svg, encoding="utf-8")
        else:
            scale = max(args.oversample, 1.0)
            render_size = (int(target_size[0] * scale), int(target_size[1] * scale))
            panel_size = (render_size[0] // 2, render_size[1] // 2)
            drawer = rdMolDraw2D.MolDraw2DCairo(render_size[0], render_size[1], panel_size[0], panel_size[1])
            opts = drawer.drawOptions()
            opts.bgColor = tuple(bg_float)
            drawer.DrawMolecules(mols)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()
            image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            if scale > 1.0 and image.size != target_size:
                image = image.resize(target_size, resample=Image.LANCZOS)
            if bg_int[3] < 255:
                back = Image.new("RGBA", target_size, tuple(bg_int))
                back.paste(image, (0, 0), image)
                final_grid = back.convert("RGB")
            else:
                final_grid = image.convert("RGB")
            final_grid.save(str(output_path), dpi=(args.dpi, args.dpi))

        print(f"Saved 2x2 grid to {output_path}")
        return 0

    # single SMILES path follows
    mol = uc.to_mol(smiles_input)
    if mol is None:
        print(f"Invalid SMILES string: {smiles_input}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(args.image_size, args.image_size)
        opts = drawer.drawOptions()
        opts.setBackgroundColour(bg_float) 
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        output_path.write_text(svg, encoding="utf-8")
    else:
        target_size = (args.image_size, args.image_size)
        render_size = tuple(
            max(1, int(dimension * max(args.oversample, 1.0))) for dimension in target_size
        )

        # Use Cairo drawer so we can control RGBA background directly and obtain PNG bytes
        drawer = rdMolDraw2D.MolDraw2DCairo(render_size[0], render_size[1])
        opts = drawer.drawOptions()
        # rdMolDraw2D expects RGB floats and alpha float
        opts.bgColor = tuple(bg_float)  # (r, g, b, a)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()

        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(png_bytes))
        # Ensure RGBA for compositing
        image = image.convert("RGBA")

        # Resize down to target if oversampled
        if args.oversample > 1.0 and image.size != target_size:
            image = image.resize(target_size, resample=Image.LANCZOS)

        # If background is not fully opaque, composite over requested solid background
        if bg_int[3] < 255:
            background = Image.new("RGBA", target_size, bg_int)
            background.paste(image, (0, 0), image)
            final = background.convert("RGB")
        else:
            # fully opaque background already baked by drawer; convert to RGB
            final = image.convert("RGB")

        final.save(str(output_path), dpi=(args.dpi, args.dpi))

    print(f"Valid SMILES. Image saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
