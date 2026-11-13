"""Contact sheets of unchanged native PNGs; labels only, no generated imagery."""
import argparse
from pathlib import Path


def render(root, output):
    from PIL import Image, ImageDraw, ImageFont
    names = ('natural.base', 'natural.source', 'base-replay', 'I', 'C', 'IC.early',
             'IC.late', 'IC.control-rows', 'IC.restoreI', 'IC.restoreT', 'IC.restoreIT')
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=19)
    for case in ('anchor', 'seed', 'wording', 'colors', 'objects', 'layout', 'material', 'texture'):
        if not all((root / (case + '.' + n + '.png')).exists() for n in names):
            continue
        sheet = Image.new('RGB', (4*384, 3*416), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(names):
            x, y = (i % 4)*384, (i // 4)*416
            with Image.open(root / (case + '.' + name + '.png')) as image:
                sheet.paste(image.resize((384,384)), (x,y+32))
            draw.text((x+6,y+6), case + ' / ' + name, font=font, fill='black')
        sheet.save(output / (case + '-contact-sheet.png'))
    return sorted(str(p) for p in output.glob('*-contact-sheet.png'))


def render_controls(root, controls, output):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=19)
    output.mkdir(parents=True, exist_ok=True)
    for case in ('anchor', 'seed', 'wording', 'colors', 'objects', 'layout', 'material', 'texture'):
        cells = [(root / (case + '.I.png'), 'Image prime only'),
                 (root / (case + '.IC.early.png'), 'Changed attribute rows'),
                 (controls / (case + '.context-controls.png'), 'Later unchanged words'),
                 (root / (case + '.natural.source.png'), 'Natural source reference')]
        sheet = Image.new('RGB', (1536, 416), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (path, title) in enumerate(cells):
            with Image.open(path) as image:
                sheet.paste(image.resize((384,384)), (384*i,32))
            draw.text((384*i+5,6), title, font=font, fill='black')
        sheet.save(output / (case + '-context-controls.png'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--controls',type=Path);a=p.parse_args()
    print('\n'.join(render(a.root,a.output)))
    if a.controls is not None:
        render_controls(a.root,a.controls,a.output)
