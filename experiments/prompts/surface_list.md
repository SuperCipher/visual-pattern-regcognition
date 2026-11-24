You are an interior finishes specialist. Look at the interior photo and list every **finished
surface** that a wall-material supplier could supply.

Include: walls, ceilings, columns, kitchen islands and counter fronts (the vertical faces),
backsplashes, wall niches and arch surrounds.
Exclude: floors, furniture, countertops (horizontal tops), doors, windows, mirrors, lamps,
plants, artwork and anything outdoors.

For each surface return:
- `box`: [x1, y1, x2, y2], the tightest box around the visible surface. Coordinates are
  normalized to 0–1000, where (0,0) is the top-left corner and (1000,1000) the bottom-right.
- `surface_type`: one of wall, ceiling, column, island, counter_front, backsplash, niche, other
- `family`: one of plaster, paint, wood, stone, tile, mosaic, wallpaper, metal, glass, fabric,
  brick, concrete, other
- `pattern`: a short phrase for the finish, e.g. "fluted vertical slats", "veined marble slab",
  "chinoiserie mural", "small square glossy tiles", "knockdown texture"
- `colour`: the main colour as a plain name plus a hex estimate, corrected for warm or coloured
  lighting, e.g. "sage green #8A9A84"
- `material_group`: a short id such as "A", "B". Give surfaces the **same id when they are the
  same material** (same family, pattern and colour), even if they are far apart in the photo.
- `confidence`: 0.0–1.0

If one material covers several separate areas (for example both side walls), return one entry
per area with the same `material_group`.

Respond with JSON only, no prose, in exactly this shape:

{"surfaces": [{"box": [0, 0, 1000, 1000], "surface_type": "wall", "family": "wood",
  "pattern": "...", "colour": "...", "material_group": "A", "confidence": 0.9}]}
