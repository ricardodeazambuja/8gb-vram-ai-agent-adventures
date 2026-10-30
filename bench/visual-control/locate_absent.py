"""Does the detect-all prompt answer <box>None</box> for things that are not there? Argument: capture folder."""
import base64
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0,'.')
import locate_probe as lp
import run_control as rc


def ask(png, prompt):
    body={"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png).decode()}},{"type":"text","text":prompt}]}],"max_tokens":96,"temperature":0}
    return json.load(urllib.request.urlopen(urllib.request.Request(rc.URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=120))["choices"][0]["message"]["content"]
T="Locate all the instances that matches the following description: {}."
def none(t): return 'None' in t or '<box>' not in t
absent=present=pres_ok=abs_ok=0
for i in range(30):
    img,_,_,_,tr=rc.scene_click(random.Random(f"click-{i}"),i)
    png=base64.b64decode(rc.png_b64(img)); t=ask(png,T.format(f"{tr['target']} button"))
    if tr['absent']: absent+=1; abs_ok+=none(t); print('absent',tr['target'],t[:80])
    else: present+=1; pres_ok+=not none(t)
print('scenes all-prompt: present answered',pres_ok,'/',present,'absent None',abs_ok,'/',absent)
import glob

a=b=c=d=0
for f in glob.glob(sys.argv[1]+'/*.json'):
    n=f[:-5]; png=Path(n+'.png').read_bytes(); e=json.loads(Path(f).read_text())['elements']
    for x in random.Random(1).sample(e,min(4,len(e))): c+=1; d+=not none(ask(png,T.format(x['text']+' link')))
    for w in lp.ABSENT[:3]: a+=1; r=ask(png,T.format(w+' link')); b+=none(r); print(n[-10:],w,r[:70])
print('web all-prompt: present answered',d,'/',c,'absent None',b,'/',a)
