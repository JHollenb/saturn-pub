"""Labeled contact sheets of native dose futures; no synthetic image content."""
import argparse
import json
from pathlib import Path


def render(root, output):
    from PIL import Image, ImageDraw, ImageFont
    doses=json.loads((root/'source-doses.json').read_text())
    arms=('sham','A.half-context','A.context','A.mid','A.full','B.half','B.full','B.attribute-norm')
    font=ImageFont.load_default(size=17);output.mkdir(parents=True,exist_ok=True);results=[]
    for case, recipes in doses.items():
        if not all((root/(case+'.'+arm+'.png')).exists() for arm in arms):continue
        sheet=Image.new('RGB',(1536,896),'white');draw=ImageDraw.Draw(sheet)
        full=recipes['A.full']['landed_delta_l2']
        for index,arm in enumerate(arms):
            x,y=384*(index%4),448*(index//4);r=recipes[arm]
            with Image.open(root/(case+'.'+arm+'.png')) as image:sheet.paste(image.resize((384,384)),(x,y+64))
            draw.text((x+5,y+4),case+' / '+arm,font=font,fill='black')
            ratio=r['landed_delta_l2']/full if full else 0.
            draw.text((x+5,y+24),f"alpha={r['alpha']:.4f}",font=font,fill='black')
            draw.text((x+5,y+44),f"delivered L2/A.full={ratio:.4f}",font=font,fill='black')
        path=output/(case+'-dose-contact-sheet.png');sheet.save(path);results.append(str(path))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print('\n'.join(render(a.root,a.output)))
