"""Small typed handoff/clamp/rescue panel before step1/single6."""


def recipes():
    rows=[{'id':'reference.'+r,'role':r,'entry':7,'mode':'reference'} for r in ('base','shape')]
    rows += [{'id':f'handoff{k}.{mode}','role':'base','entry':k,'mode':mode} for k in (0,3,6) for mode in ('I','T')]
    rows += [{'id':f'handoff{k}.IT','role':'base','entry':k,'mode':'IT'} for k in (0,6)]
    rows += [{'id':name,'role':role,'entry':0,'mode':name} for name,role in (
        ('Tseed0.reset6','base'),('Tclamp0to6','shape'),('Tclamp0to5.rescue6','shape'),('Iclamp0to6','shape'))]
    return rows


def edits(recipe,index,source_cuts):
    """Declared pre-operator interventions only; index7 is observation-only."""
    mode=recipe['mode'];low,high=source_cuts['base'][index],source_cuts['shape'][index]
    if mode in ('I','T','IT') and index==recipe['entry']:
        return {slot:high.payload[slot] for code,slot in (('I','image'),('T','text')) if code in mode}
    if mode=='Tseed0.reset6':
        if index==0:return {'text':high.payload['text']}
        if index==6:return {'text':low.payload['text']}
    if mode=='Tclamp0to6' and index<=6:return {'text':low.payload['text']}
    if mode=='Tclamp0to5.rescue6' and index<=6:return {'text':(high if index==6 else low).payload['text']}
    if mode=='Iclamp0to6' and index<=6:return {'image':low.payload['image']}
    return {}


def census(*,fresh=False):
    r=recipes();native=14+sum(78-v['entry'] for v in r)
    acts=2+6*2+2*3+3+3*8
    if fresh:native+=40;acts+=2
    return dict(loads=1,encodes=0,noise_draws=int(fresh),native_transitions=native,acts=acts,receipts=native+acts,images=len(r))
