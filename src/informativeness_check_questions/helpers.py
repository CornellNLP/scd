import re
import random
import string
from nltk.tokenize import sent_tokenize, word_tokenize


def fix_summary_chars(input_text):
    """
    Given by Yilun
    use this right before you output the transcript
    :param input_text:
    :return:
    """

    # Define the regex pattern to match '&gt;' and the first '\n' behind it
    pattern = r'&gt;(.*?)\n'

    # Define a function to handle the replacement

    # TODO fix the quotation mark in the original json formatted transcripts.
    # quote until the end of the comment?
    def replace_quote(match):
        return '"' + match.group(1) + '"'

    # Use re.sub() to perform the replacement
    def fix_transcript_quote(transcript):
        return re.sub(pattern, replace_quote, transcript)

    # Print the result
    output_text = fix_transcript_quote(input_text)
    return output_text


def ordered_anonymize(convo_script):
    """
    Anonymizes a given list of utterances from a summary
    :param convo_script: list of utterances
    :return:
    """
    speakers = []
    for utt in convo_script:
        s = utt.split(':')[0]
        if s != '[deleted]':
            if s not in speakers:
                speakers.append(s)

    new_convo_script = []
    for sent in convo_script:
        for i, s in enumerate(speakers):
            sent = sent.replace(s + ':', f'SPEAKER{i + 1}:')
            sent = sent.replace(s + ' ', f'SPEAKER{i + 1} ')
            sent = sent.replace(' ' + s, f' SPEAKER{i + 1}')

        new_convo_script.append(fix_summary_chars(sent))
    return new_convo_script


def get_speakers(convo_script):
    """
    :param convo_script:
    :return: the number of speakers in each script
    """
    return set([word_tokenize(t)[0] for t in convo_script])


def get_speakers_summary(summary):
    """
    :param summary:
    :return: the number of speakers in each script
    """
    return set([word for word in word_tokenize(summary) if "SPEAKER" in word])


def gen_random_str():
    return ''.join(random.choices(string.ascii_uppercase, k=3))


def anonymize_given_question(question):

    total_speakers = get_speakers(question["transcript"])
    speaker_map = {speaker: f"SPEAKER_{gen_random_str()}" for speaker in total_speakers}
    target_choice_map = {}

    for i in range(len(question["transcript"])):
        for speaker in total_speakers:
            question["transcript"][i] = question["transcript"][i].replace(speaker, speaker_map[speaker])

    # guarantee that seg_pos comes first, so we sort
    for key, value in sorted(question["choices"].items(), reverse=True):
        curr_choice = value["human_written_SCD"]
        summary_tokenized = [sent for sent in sent_tokenize(curr_choice) if len(get_speakers_summary(sent)) == 2]
        # pick a random sentence that has two speakers
        target_sentence = random.sample(summary_tokenized, 1)[0]
        sentence_speakers = get_speakers_summary(target_sentence)
        speakers_avail = set(
            [value for key, value in target_choice_map.items() if key not in sentence_speakers])
        for curr_speaker in sentence_speakers:
            if key == "seg_pos":
                target_choice_map[curr_speaker] = speaker_map[curr_speaker]
            replaced = speakers_avail.pop() if target_choice_map.get(curr_speaker) is None else target_choice_map.get(curr_speaker)
            target_sentence = target_sentence.replace(curr_speaker, replaced)
            question["choices"][key] = target_sentence

    scheme = [[], []]
    for key, value in sorted(target_choice_map.items()):
        scheme[0].append(key)
        scheme[1].append(value[-3:])

    question["annon_scheme"] = scheme
    question["transcript"] = "\n\n".join(question["transcript"])
    return question
