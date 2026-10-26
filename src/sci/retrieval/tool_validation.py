from sci.paths import ROOT
import json
EXAMPLES={'graphsearch':{'queries':['...']},'graphexpand':{'source_id':'123','query':'...'},'read':{'path':'.cep/s/123.txt'},'bash':{'command':'rg ... .cep/s','timeout':20}}
def validate_arguments(name,args):
    specs={x['name']:x['parameters'] for x in json.loads((ROOT/'configs/tools.json').read_text())}
    if name not in specs:raise ValueError('Unknown tool: '+name)
    schema=specs[name];allowed=set(schema['properties']);example=json.dumps(EXAMPLES[name])
    if not isinstance(args,dict):raise ValueError(f'{name} arguments must be a JSON object. Example: {example}')
    extra=set(args)-allowed
    if extra:raise ValueError(f'{name} received unsupported argument(s): {", ".join(sorted(extra))}. Accepted fields: {", ".join(sorted(allowed))}. Remove the unsupported fields. Example: {example}')
    missing=set(schema.get('required',[]))-set(args)
    if missing:raise ValueError(f'{name} is missing required field(s): {", ".join(sorted(missing))}. Example: {example}')
    def check(value,spec,label):
        typ=spec.get('type')
        if typ=='string' and not isinstance(value,str):raise ValueError(label+' must be a string')
        if typ=='number' and (not isinstance(value,(int,float)) or isinstance(value,bool)):raise ValueError(label+' must be a number')
        if typ=='array':
            if not isinstance(value,list):raise ValueError(label+' must be an array')
            if len(value)<spec.get('minItems',0):raise ValueError(label+' must contain at least '+str(spec['minItems'])+' item(s)')
            for i,v in enumerate(value):check(v,spec.get('items',{}),f'{label}[{i}]')
    for key,value in args.items():check(value,schema['properties'][key],name+'.'+key)
