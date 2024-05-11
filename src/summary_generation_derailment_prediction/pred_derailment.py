import json, os, glob, openai, time
from tqdm import tqdm
from openai.error import APIError, RateLimitError, Timeout
from sklearn.metrics import classification_report

openai.api_key = '' 

transcript_path = ''
pred_on_summ=True #set this to False if want to predict on transcripts 
summary_type='generated_summary'
trial_names =[
'summ_gpt_procedural_2024_04_24-23_24_29'
]

default_model = "gpt-3.5-turbo-0613"  

if pred_on_summ is False:
    summary_type='NA'
    trial_names =['transcript']
    default_model = "gpt-3.5-turbo-16k-0613"
   



############################################
#classifier prompt for zeroshot/procedural SCDs. The classification examplars are zeroshot SCDs. We use this prompt for both zeroshot SCDs and procedural SCDs, to control for the variance of exemplars. 

pred_prompt="""Based on its trajectory summary, predict if an online conversation will go awry (i.e. lead to personal attack or offensive language by some participants). Examples are provided. Output your answer as a single word, either “true” or “false”. 

Examples:
"trajectory summary": "Speaker1 presents a hypothesis about biased hiring practices, which Speaker2 disagrees with, leading to a discussion on the ability to draw conclusions from observations. Speaker3 raises the possibility of white men being the best candidates, but Speaker4 brings up counterarguments. Speaker5 argues for equal performance regardless of race, leading to misunderstandings with Speaker4. The conversation ends with Speaker4 emphasizing the need for diversity in workplaces. Tensions arise as participants question each other's viewpoints."
"will go awry": true

"trajectory summary": "A user expresses concern about increased expenses due to higher employee salaries, while another user questions why taxpayers should subsidize businesses that pay low wages. They debate the impact on taxes and argue about the effects of minimum wage on the market and unemployment. Another user criticizes the belief that minimum wage is the solution, leading to a disagreement around the intentions behind advocating for higher minimum wage. The conversation becomes more heated as they accuse each other of misrepresenting arguments and present a flawed analogy."
"will go awry": true

"trajectory summary": "Speakers start by discussing the government's role in compensation, with a conflict arising between Speaker1 and Speaker2. Speaker3 intervenes, emphasizing morality, and Speaker4 joins, debating compensation for those with children. Speaker5 introduces the future population argument. Speaker4 responds, disagreeing with Speaker5's point. Speaker6 clarifies Speaker5's intent, leading to further disagreement. The conversation ends with Speaker4 responding to an anonymous user, addressing the issue of immigration. Overall, the conversation transitions from government involvement and morality to the implications of compensating those with children and future population concerns."
"will go awry": false

"trajectory summary": "The conversation starts with Speaker1 accusing Speaker2 of sounding racist. They discuss the admissions process at MIT and debate the inclusion of race as a factor. Speaker2 brings up statistics on Asian acceptance rates, which Speaker1 argues can be attributed to wealth disparities. The conversation becomes increasingly focused on wealth and its impact on acceptance rates. As Speaker2 tries to clarify their point, Speaker1 continues to emphasize the role of wealth in the discussion. The conversation ends with both participants still somewhat unclear on each other's perspectives."
"will go awry": true

"trajectory summary": "Speaker1 and Speaker2 engage in a conversation about race and college admissions. They discuss the use of race as a factor, disparities in acceptance rates, and whether it can be considered racism. Speaker3 introduces the idea of diversity as a goal for schools. Speaker4 and Speaker5 question the concept of diversity, while Speaker3 defends it. Speaker6 disagrees with the notion that diversity improves education. Speaker7 provides links supporting the importance of diversity in universities."
"will go awry": false

Predict for the following conversation summary:\n
"""


############################################
#classifier prompt for traditional summaries. Traditional summaries follow a very different distribution from SCDs, so they require a separate set of exemplars for the few-shot classifier. The exemplars below are also generated with the traditional summary prompt.



# pred_prompt="""Based on its summary, predict if an online conversation will go awry (i.e. lead to personal attack or offensive language by some participants). Examples are provided. Output your answer as a single word, either “true” or “false”. 

# Examples:
# "summary": "In an online discussion, Speaker 1 compares coin flips to hiring practices, suggesting that a lack of diversity may indicate bias. Speaker 2 disagrees, stating you can't conclude bias from observation alone. Speaker 3 defends using statistics to infer bias. Speaker 4 suggests that hiring the best candidates, even if they're all white men, is a valid conclusion. Speaker 5 argues that other data contradicts this. Speaker 6 believes that cultural factors affect outcomes, while Speaker 5 questions the likelihood of an all-white workforce."
# "will go awry": true

# "summary": "In an online discussion, Speaker 1 expresses concerns about paying higher entry-level salaries, fearing business closure and job loss, especially for smaller local businesses. Speaker 2 argues that government support for employees should replace low wages. Speaker 3 questions how shutting down the business would reduce taxes and Speaker 4 cautions against labor price controls. Speaker 5 criticizes the belief that raising the minimum wage can solve unemployment and Speaker 3 points out a misrepresentation in their argument."
# "will go awry": true

# "summary": "In this conversation, Speaker1 questions the need for a higher minimum wage, arguing that it shifts the burden of paying for basic services from companies to the government, potentially raising taxes. Speaker2 counters, stating that an employee's value is determined by their skills, not their financial management. Speaker3 argues that people deserve a wage allowing them to live, suggesting that companies failing to provide this should be dissolved. Speaker4 raises the issue of minimum wage variation for those with children."
# "will go awry": false

# "summary": "In this online conversation, Speaker 1 criticizes an individual's comment, suggesting it sounded racist. They discuss college admissions and the importance of diversity. Speaker 2 presents statistics on racial disparities in university acceptance rates and argues it's racism. Speaker 1 counters by mentioning economic differences among racial groups. The conversation shifts to a debate about minimum wage, with Speaker 1 arguing for fair compensation for employees, while Speaker 3 brings up the responsibilities of business owners. Finally, they discuss the challenges faced by single parents."
# "will go awry": true

# "summary": "The conversation revolves around race in the college admissions process. Speaker 1 outlines a holistic evaluation method, mentioning that race is one of several factors considered. Speaker 2 expresses concern about racial disparities, citing data. Speaker 3 suggests diversity as a goal, and Speaker 4 questions this as potentially racist. Speaker 5 advocates for focusing on qualification and intellectualism over diversity. Speaker 6 disagrees, while Speaker 7 provides links supporting the importance of diversity in universities. The debate centers on whether diversity should be a priority in admissions decisions."
# "will go awry": false

# Predict for the following conversation summary:\n
# """



def gpt_query(query, output_max_tokens=3, model=default_model, times_retried=0):
    retry_after=10
    if times_retried>3:
      raise Exception('retry failed')

    messages=[
              {"role": "user", "content": f"{query}"}
          ]

    try:
        response = openai.ChatCompletion.create(
          model=model,
          messages=messages,
          max_tokens=output_max_tokens,
          temperature=0,
          request_timeout = 20
        )
         
    except APIError as e:
        retry_after=10
        print(f"API error. Waiting and retrying in {retry_after} seconds...")
        time.sleep(retry_after)
        times_retried+=1
        return gpt_query(query, output_max_tokens, times_retried=times_retried)

 
    except RateLimitError as e:
        print(e)
        print(f"Rate limit exceeded. Waiting and retrying in {retry_after} seconds...")
        time.sleep(retry_after)
        times_retried+=1
        return gpt_query(query, output_max_tokens, times_retried=times_retried)
    
    except Timeout as e:
        print(e)
        print(f"Timeout. Waiting and retrying in {retry_after} seconds...")
        time.sleep(retry_after)
        times_retried+=1
        return gpt_query(query, output_max_tokens, times_retried=times_retried)

       
    return response


for trial_name in trial_names:
    all_convos=[]
    if pred_on_summ:
      all_fps=glob.glob(f"{transcript_path}/pair_*/{trial_name}/summ.json")
    else:
      all_fps=glob.glob(f"{transcript_path}/pair_*/script.json")

    for fp in all_fps:
        with open(fp) as f:
            convo_pair=json.load(f)
            all_convos.extend(convo_pair)


    targets=[]
    preds=[]
    for convo in tqdm(all_convos):
        if pred_on_summ:
          query=pred_prompt+"\"summary\": "+"\""+convo[summary_type]+"\""+"\n\"will go awry\": "
        else:
          query=pred_prompt+"\n"+'\n\n'.join(convo['convo_content'])+"\n\n"+"[label]: "
            
            
        response=gpt_query(query)
        result=response.choices[0].message['content'].strip()
        if not pred_on_summ:
            result = result.replace('CIVIL', 'false')
            result = result.replace('DERAIL', 'true')
            
        convo['pred_awryness']=result.lower()
        assert convo['pred_awryness'] in ['true', 'false']
        targets.append(convo['has_removed_comment'])
        pred_label=(convo['pred_awryness']=='true')
        preds.append(pred_label)
        if pred_label==convo['has_removed_comment']:
            convo['pred_correct']=True
        else:
            convo['pred_correct']=False
    preds_save_fn=os.path.join(transcript_path, f'{trial_name}_gpt_classifier_pred.json')

    with open(preds_save_fn, 'w', encoding='utf-8') as f:
        result=json.dump(all_convos,f, indent=4)

    
    print(classification_report(targets,preds))


