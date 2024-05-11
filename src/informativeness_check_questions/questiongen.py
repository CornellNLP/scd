import json
import glob
import os
import random
from collections import deque
from helpers import ordered_anonymize, anonymize_given_question


def get_summary_pairs(summary_file_names):
    """
    This function process summaries in a JSON file and returns them as a pair of summaries
    :param summary_file_names: a list of file paths to summaries
    :return:
    """
    pairs = []

    for filename in summary_file_names:
        with open(filename) as f:
            summaries = json.load(f)
            for i in range(len(summaries) // 2):
                idx = i * 2
                pairs.append((summaries[idx], summaries[idx + 1]))

    random.shuffle(pairs)
    # need a multiple of 3 for algorithm to work
    return pairs[0:len(pairs) - (len(pairs) % 3)]


def get_transcripts(pairs, data_dir="data"):
    """
    A lot of assumptions made here
    :param pairs:
    :param summaries_dir:
    :return: return a list of
    """
    transcripts = []

    for i, (summ1, summ2) in enumerate(pairs):
        id1, id2 = summ1["convo_id"], summ2["convo_id"]
        pair_dir = f"{data_dir}/pair_{id1}_{id2}"

        with open(pair_dir + "/script.json") as transcript_file:
            script1, script2 = json.load(transcript_file)
            # re anonymize everything
            script1["convo_content"] = ordered_anonymize(script1["convo_content"])
            script2["convo_content"] = ordered_anonymize(script2["convo_content"])
            transcripts.append((script1, script2))


 
    assert len(transcripts) == len(pairs)

    return transcripts


# def find_highest_numbered_speaker(summary):
#     return sorted([word for word in word_tokenize(summary) if "SPEAKER" in word])[-1]


def generate_questions(transcript_pairs, pairs):
    """
    N = # of questions
    N // 2 * 3 = the number of pairs we need to achieve this

    For each transcript, we want to have 3 answers:
    We have the actual summary
    Distractor 1 is the pair of the opposite label
    Distractor 2 is a summary of the same label from a different pair

    Questions will be in the following format:
    {
        convo_id: the convo_id of the summary
        transcript: the transcript (a string)
        choices: a dict of the question choices, seg_pos{1,2}, seg_neg{1,2}
        data: has the og id, summary, and type (target, pair, other)
        annon_scheme: list of list
    }
    :return: a list of questions in the above format

    notes: need to have 1/3 for positive, 1/3 for negative, 1/3 for filler
    first select the summary with more speakers, and prioritize both ones with really large number of speakers
    need to find the highest number speaker in the summary
    """
    num_questions = len(transcript_pairs) // 3 * 2
    questions = []

    leftover = deque([])
    curr_idx = 0
    for i in range(num_questions):
        transcript1, transcript2 = transcript_pairs[curr_idx]
        pair1, pair2 = pairs[curr_idx]
        choices_map = {}
        # choose either removed or not removed depending on how many questions there are
        if len(questions) < num_questions // 2:
            # pair1["label"], pair2["label"] = "target", "pair"
            choices_map["seg_pos"], choices_map["seg_neg1"] = pair1, pair2
            data = [pair1, pair2]
            transcript = transcript1["convo_content"]
            convo_id = transcript1["convo_id"]
            # add the last distactor
            curr_idx += 1
            choices_map["seg_neg2"] = pairs[curr_idx][0]
            # pairs[curr_idx][0]["label"] = "other"
            data.append(pairs[curr_idx][0])
            leftover.append(pairs[curr_idx][1])
        else:
            # pair2["label"], pair1["label"] = "target", "pair"
            choices_map["seg_neg1"], choices_map["seg_pos"] = pair1, pair2
            data = [pair2, pair1]
            transcript = transcript2["convo_content"]
            convo_id = transcript2["convo_id"]
            # add the last distactor
            other = leftover.popleft()
            choices_map["seg_neg2"] = other
            # other["label"] = "other"
            data.append(other)

        question = {
            "convo_id": convo_id,
            "choices": choices_map,
            "data": data,
            "transcript": transcript,
        }
        questions.append(question)
        curr_idx += 1

    # anonymize each question
    questions = [anonymize_given_question(question) for question in questions]

    return questions


def output_file(questions, out_dir="questions_genned"):
    for i, question in enumerate(questions):
        with open(f"{out_dir}/Q{i}_{question['convo_id']}.txt", "w+") as question_file:
            # write a string of hyphens
            question_file.write(("-" * 60) + "\n")
            # write the question id
            question_file.write(f"Question_ID: {question['convo_id']}\n\n")
            # write answer choices
            choices = list(question["choices"].values())
            random.shuffle(choices)
            for j, choice in enumerate(choices):
                question_file.write(f"Choice {j + 1}: {choice}\n\n")
            # write the correct answer
            question_file.write(f"[Target: Choice {choices.index(question['choices']['seg_pos']) + 1}]\n")
            # write the transcript divider
            question_file.write(("*" * 30) + "\n")
            question_file.write("Transcript:\n")
            # write the whole transcript
            question_file.write(question["transcript"] + "\n")


if __name__ == "__main__":
    # assume we have the summary files in this directory
    # summary_files = glob.glob("*.json")

    summary_files = ["data/summaries_of_conversation_dynamics/human_SCDs.json"]
    transcript_dir = "data/transcripts/test"
    
    # process these files to get pairs
    summary_pairs = get_summary_pairs(summary_files)
    random.shuffle(summary_pairs)

    # retrieve the transcripts from each pair
    script_pairs = get_transcripts(summary_pairs, data_dir=transcript_dir)

    # generate questions (in a dict format) based on each pair in a systematic way, shuffle them, write to JSON
    questions_list = generate_questions(script_pairs, summary_pairs)
    random.shuffle(questions_list)
    if not os.path.isdir("questions_genned"):
        os.mkdir("questions_genned")
    else:
        files = glob.glob('questions_genned/*')
        for f in files:
            os.remove(f)
    with open("questions_genned/questions.json", "w") as file:
        json.dump(questions_list, file, indent=4)

    # output question files based on the question list (.txt)
    output_file(questions_list)
