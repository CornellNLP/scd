import json, os, glob, openai, time
from datetime import datetime
from tqdm import tqdm
from openai.error import APIError, RateLimitError, InvalidRequestError, Timeout

openai.api_key = ''
default_model, backup_model = "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"
output_max_tokens = 128

#####################################################
# procedural summary prompt
prompt_p1="""Write a short summary capturing the trajectory of an online conversation.  
Do not include specific topics, claims, or arguments from the conversation. The style you should avoid: 
Example Sentence 1: “Speaker1, who is Asian, defended Asians and pointed out that a study found that whites, Hispanics, and blacks were accepted into universities in that order, with Asians being accepted the least. Speaker2 acknowledged that Asians have high household income, but argued that this could be a plausible explanation for the study's findings. Speaker1 disagreed and stated that the study did not take wealth into consideration.” 
This style mentions specific claims and topics, which are not needed.

Instead, do include indicators of sentiments (e.g., sarcasm, passive-aggressive, polite, frustration, attack, blame), individual intentions (e.g., agreement, disagreement, persistent-agreement, persistent-disagreement, rebuttal, defense, concession, confusion, clarification, neutral, accusation) and conversational strategies (if any) such as 'rhetorical questions', 'straw man fallacy', 'identify fallacies',  and 'appealing to emotions.' 
The following sentences demonstrate the style you should follow:

Example Sentence 2: “Both speakers have differing opinions and appeared defensive. Speaker1 attacks Speaker2 by diminishing the importance of his argument and Speaker2 blames Speaker1 for using profane words. Both speakers accuse each other of being overly judgemental of their personal qualities rather than arguments.”

Example Sentence 3: “The two speakers refuted each other with back and forth accusations. Throughout the conversation, they kept harshly fault-finding with overly critical viewpoints, creating an intense and inefficient discussion.”

Example Sentence 4: “Speaker1 attacks Speaker2 by questioning the relevance of his premise and Speaker2 blames Speaker1 for using profane words. Both speakers accuse each other of being overly judgemental of their personal qualities rather than arguments.”

Overall, the trajectory summary should capture the key moments where the tension of the conversation notably changes. Here is an example of a complete trajectory summary. 

Trajectory Summary: 
Multiple users discuss minimum wage. Four speakers express their different points of view subsequently, building off of each other’s arguments. Speaker1 disagrees with a specific point from Speaker2’s argument, triggering Speaker2 to contradict Speaker1 in response. Then, Speaker3 jumps into the conversation to support Speaker1’s argument, which leads Speaker2 to adamantly defend their argument. Speaker2 then quotes a deleted comment, giving an extensive counterargument. The overall tone remains civil.

Now, provide the trajectory summary for the following conversation.
Conversation Transcript:
"""

prompt_p2="""\n\nNow, summarize this conversation. Remember, do not include specific topics, claims, or arguments from the conversation. Instead, try to capture the speakers' sentiments, intentions, and conversational/persuasive strategies. Limit the trajectory summary to 80 words. 

Trajectory Summary:
"""

#####################################################
# traditional summary prompt
# prompt_p1="""
# Summarize this transcript of online conversation in 80 words. 
# """

# prompt_p2="""
# Summary:
# """

#####################################################

#zeroshot summary prompt
# prompt_p1="""
# Give a trajectory summary of the transcript. Do not include specific topics, claims, or arguments from the conversation. Instead, try to capture how the speakers' sentiments, intentions, and conversational/persuasive strategies change or persist throughout the conversation. Limit the trajectory summary to 80 words. 

# prompt_p2="""\n\nNow, summarize this conversation. Remember, do not include specific topics, claims, or arguments from the conversation. Instead, try to capture the speakers' sentiments, intentions, and conversational/persuasive strategies. Limit the trajectory summary to 80 words. 

# Trajectory Summary:
# """






def gpt_query(query, output_max_tokens=512, model=default_model, times_retried=0):
    retry_after=30

    if times_retried>4:
      raise Exception('retry failed')

    messages=[
              {"role": "system", "content": "You are an assistant who writes concise summaries of online conversations."},
              {"role": "user", "content": f"{query}"},
            ]

    try:
      response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=output_max_tokens,
        request_timeout = 20, 
      )

    except APIError as e:
        print(e)
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
        print(f"Waiting and retrying in {retry_after} seconds...")
        time.sleep(retry_after)
        times_retried+=1
        return gpt_query(query, output_max_tokens, times_retried=times_retried)
    
    except InvalidRequestError as e:

        if "Please reduce the length of the messages" in e._message:
          if model != backup_model:
            print("Query Too Long, Use Backup Model Instead")
            return gpt_query(query, output_max_tokens, model=backup_model)

        raise Exception(f"Error: {e._message}")


    return response, model



def run_streamline(transcript_path, output_max_tokens=512, unfinished_trialname=None, trial_prefix=None):
    
    if unfinished_trialname:
       save_dirname=unfinished_trialname
    else:
        trial_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        save_dirname=f"{trial_prefix}_{trial_time}"
    
    convo_pairs_path = glob.glob(f"{transcript_path}/*/script.json")
    convo_pairs_path=[os.path.dirname(path) for path in convo_pairs_path]
    convo_pairs_path = sorted(convo_pairs_path)

    pbar = tqdm(total=len(convo_pairs_path))
    for path in convo_pairs_path: 
        if unfinished_trialname and os.path.isfile(os.path.join(path, unfinished_trialname,'summ.json')):
            pbar.write(f'skipped {path}')
            pbar.update(1)
            continue


           
        convo_pair = json.load(open(f"{path}/script.json"))
        summ_pair = []

        for convo in convo_pair:

            query = "{}{}{}".format(prompt_p1, '\n\n'.join(convo['convo_content']), prompt_p2)
            response, model_tag = gpt_query(query, output_max_tokens)
            txt = response.choices[0].message['content'].strip()

                
            summ_pair.append(
                {
                    "convo_id": convo["convo_id"],
                    "has_removed_comment": convo["has_removed_comment"],
                    "generated_summary": txt,
                    "response_model": model_tag,
                }
            )
        
        summ_path = f"{path}/{save_dirname}"
        if not os.path.exists(summ_path):
            os.mkdir(summ_path)
        with open(f"{summ_path}/summ.json", "w") as f:
            json.dump(summ_pair, f, indent=4)
            f.close()
        pbar.write(f'Saving {path}')
        pbar.update(1)
    print('all completed!')



run_streamline(transcript_path='', output_max_tokens=output_max_tokens, trial_prefix='summ_gpt_procedural')
