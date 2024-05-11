### Generating Summaries 

Before you start, organize the conversation scripts such that each pair of conversation has its own subfolder named with the convo_ids of the two conversations in the pair. Inside this subfolder, let `script.json` contain the transcripts of the pair of conversations.

In `summ_gen.py`, 

* change openai.api_key to your own api key.

* make sure `transcript_path`  points at the correct directory. 

* choose the correct sets of prompt in `summ_gen.py` for generating procedural/zeroshot/traditional summaries; comment out the other prompts

* Name your trial with the `trial_prefix` variable 

Run `summ_gen.py`, the generated summaries will be placed under the respective subfolders containing the transcripts under a new folder with the trial name. 



Note: 

* we used openai 0.27.8 library, newer versions may have a different syntax so we recommend using 0.27.8. 
* Conversation cy2y2sg and cxx4vnu from our dev set were used to develop our procedural prompt. Thus, our procedural prompt includes example summary segments partly inspired by the content of these conversations. The machine-generated summaries of these two conversations (based on the procedural prompt) did not seem stylistically different from other conversations’, so we still include these two conversations and their machine-generated summaries in the dev set. 



### Predicting on generated summaries/transcripts

In `pred_derailment.py` 

* change openai.api_key to your own api key.
* choose the correct `pred_prompt` with exemplar summaries following the same distribution as your test set; comment out other prompts. (for predicting on transcripts, the prompt is provided in `few-shot_transcript_classifier_prompt.txt`)
* set `pred_on_summ` depending on whether you are predicting on summaries or transcripts
* make sure `transcript_path`  points at the correct directory.  At this point, it should also contain the generated summaries
  * specify a list of trial_names in `trial_names`, for example `['procedural_2023_XXXX','procedural_2023_YYYY]`, to predict for summaries from different trials. 

Run `pred_derailment.py`, the generated predictions will be stored in a json file under `transcript_path`





