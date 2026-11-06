"""Contact sheets of native images from the direct-T factorial."""
import argparse
import json
from pathlib import Path


def render(root, output):
    from PIL import Image, ImageDraw, ImageFont
    recipes = json.loads((root / 'source-recipes.json').read_text())
    arms = ('sham', 'replay-A.full', 'replay-B.full', 'A@A', 'A@B', 'B@A', 'B@B')
    font = ImageFont.load_default(size=17); output.mkdir(parents=True, exist_ok=True); paths = []
    for case, panel in recipes.items():
        if not all((root / (case + '.' + arm + '.png')).exists() for arm in arms):
            continue
        sheet = Image.new('RGB', (1536, 936), 'white'); draw = ImageDraw.Draw(sheet)
        for index, arm in enumerate(arms):
            x, y = 384 * (index % 4), 468 * (index // 4); recipe = panel[arm]
            with Image.open(root / (case + '.' + arm + '.png')) as image:
                sheet.paste(image.resize((384, 384)), (x, y + 84))
            draw.text((x + 5, y + 4), case + ' / ' + arm, font=font, fill='black')
            draw.text((x + 5, y + 25), f"alpha={recipe['alpha']:.4f}", font=font, fill='black')
            draw.text((x + 5, y + 46), f"actual text delta L2={recipe['landed_delta_l2']:.4f}", font=font, fill='black')
        draw.text((1157, 492), 'Payload @ receiving rows', font=font, fill='black')
        draw.text((1157, 522), 'A: changed attribute rows', font=font, fill='black')
        draw.text((1157, 552), 'B: later contextual rows', font=font, fill='black')
        draw.text((1157, 602), 'All seven cells: native VAE', font=font, fill='black')
        path = output / (case + '-route-payload.png'); sheet.save(path); paths.append(str(path))
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); print('\n'.join(render(a.root, a.output)))
