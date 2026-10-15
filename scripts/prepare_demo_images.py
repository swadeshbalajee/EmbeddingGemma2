"""Tiny synthetic pictures for pipeline verification only, NOT caption-quality scoring."""
from pathlib import Path
from PIL import Image, ImageDraw
import json

out = Path("data/eval_images")
out.mkdir(parents=True, exist_ok=True)
styles = [("red",(220,45,45)), ("blue",(45,85,220)), ("green",(40,160,75)),
          ("yellow",(235,190,35))]
with (out/"references.jsonl").open("w") as f:
    for name, rgb in styles:
        for shape in ("circle","square","triangle"):
            im = Image.new("RGB",(320,240),(245,245,240))
            d = ImageDraw.Draw(im)
            if shape == "circle": d.ellipse((95,55,225,185), fill=rgb)
            elif shape == "square": d.rectangle((95,55,225,185), fill=rgb)
            else: d.polygon([(160,45),(80,190),(240,190)], fill=rgb)
            path = out/f"{name}_{shape}.png"
            im.save(path)
            f.write(json.dumps({"path":str(path),"reference":f"A {name} {shape} on a light background."})+"\n")
print("Wrote",12,"pipeline-check images to",out)
