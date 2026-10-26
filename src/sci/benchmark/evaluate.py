"""Evaluate QA predictions with a judge model."""
import argparse
import json
import os
from pathlib import Path
import urllib.request

SYSTEM = ('You are grading a question-answer benchmark. '
          'Mark the prediction correct only if it identifies the same final answer as the gold answer. '
          'Ignore case, surrounding punctuation, whitespace, and extra explanation or supporting file paths. '
          'Do not give partial credit. Return exactly one compact JSON object.')


def load_rows(path,key):
    result={}
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);qid=str(row.get(key,row.get('query_id','')))
        if not qid or qid in result:raise ValueError('Missing/duplicate ID')
        result[qid]=row
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('questions');p.add_argument('predictions');p.add_argument('output',type=Path)
    p.add_argument('--model',required=True);p.add_argument('--max-output-tokens',type=int,default=512)
    args=p.parse_args()
    questions=load_rows(args.questions,'id');predictions=load_rows(args.predictions,'query_id')
    if predictions.keys()!=questions.keys():raise ValueError('Question/prediction ID sets differ; use the same evaluation subset')
    base=os.environ['SCI_JUDGE_API_BASE'].rstrip('/');key=os.environ['SCI_JUDGE_API_KEY']
    args.output.mkdir(parents=False,exist_ok=False)
    grades=[]
    for qid,row in questions.items():
        pred=predictions[qid]
        if pred['status']!='answered' or not pred.get('prediction'):
            grades.append({'query_id':qid,'is_correct':False,'reason':'No completed answer','judge_called':False});continue
        prompt=(f"Question:\n{row.get('question',row.get('query'))}\n\nGold answer:\n{row['answer']}\n\n"
                f"Predicted answer:\n{pred['prediction']}\n\n"
                'Return JSON with keys "is_correct" (boolean), "normalized_prediction" (string), and "reason" (string).')
        request={'model':args.model,'reasoning':{'effort':'low'},'text':{'verbosity':'low'},
                 'max_output_tokens':args.max_output_tokens,'input':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]}
        # Use sequential filenames because query IDs may contain path separators.
        index=len(grades)
        (args.output/f'{index}.request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
        req=urllib.request.Request(base+'/responses',data=json.dumps(request).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=300) as response:payload=json.load(response)
        (args.output/f'{index}.response.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        text=payload.get('output_text') or '\n'.join(b['text'] for item in payload.get('output',[]) for b in item.get('content',[]) if b.get('type')=='output_text')
        grade=json.loads(text)
        if type(grade.get('is_correct')) is not bool:raise ValueError('Invalid judge output; raw response retained')
        grades.append({**grade,'query_id':qid,'judge_called':True,'usage':payload.get('usage')})
        (args.output/'grades.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in grades))
    (args.output/'grades.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in grades))
    summary={'questions':len(questions),'correct':sum(x['is_correct'] for x in grades),'accuracy':sum(x['is_correct'] for x in grades)/len(questions) if questions else None,'judge_model':args.model,'protocol':'single-attempt, incomplete answers incorrect, no best-of retry'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
