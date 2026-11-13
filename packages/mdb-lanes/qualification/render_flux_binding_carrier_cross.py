"""Native RGB 2x2 panels: image source by working-text source."""
import argparse
import json
from pathlib import Path


def render(root, output):
    from PIL import Image, ImageDraw, ImageFont
    recipes = json.loads((root / 'source-recipes.json').read_text())
    arms = ('I.P_T.P', 'I.P_T.Q', 'I.Q_T.P', 'I.Q_T.Q')
    output.mkdir(parents=True, exist_ok=True); paths = []; font = ImageFont.load_default(size=20)
    for case in recipes:
        if not all((root / (case + '.' + arm + '.png')).exists() for arm in arms): continue
        sheet = Image.new('RGB', (1024, 1200), 'white'); draw = ImageDraw.Draw(sheet)
        for index, arm in enumerate(arms):
            x, y = 512*(index % 2), 568*(index // 2)
            with Image.open(root / (case + '.' + arm + '.png')) as image:
                if image.size != (512, 512): raise ValueError('native image geometry differs')
                sheet.paste(image, (x, y+56))
            draw.text((x+6, y+3), case + ' / ' + arm, font=font, fill='black')
            label = 'known self replay' if arm in ('I.P_T.P', 'I.Q_T.Q') else 'new hybrid continuation'
            draw.text((x+6, y+29), label, font=font, fill='black')
        draw.text((6, 1142), 'P: prior A@A transfer-side state; Q: prior A@B base-side state', font=font, fill='black')
        draw.text((6, 1170), 'Rows vary image source; columns vary text source. All cells: native VAE.', font=font, fill='black')
        path = output / (case + '-carrier-cross.png'); sheet.save(path); paths.append(str(path))
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); print('\n'.join(render(a.root, a.output)))
