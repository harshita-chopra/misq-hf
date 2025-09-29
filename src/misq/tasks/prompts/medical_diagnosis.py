from misq.tasks.prompts.general import *

# method
generate_prompt_rest = '''You are a doctor. Here are all diseases that the patient may suffer from:
{items_str}

{n} questions are designed to classify the possible diseases above based on the answer for these question:
{asked}
For each disease under each question, if the answer is Yes, put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many diseases in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''You are a doctor. Here are all the possible diseases that the patient may suffer from:
{items_str}

Design a question to ask your patient regarding symptoms of their illness that can only be answer by YES or NO. Then classify the possible diseases above based on this question. If the answer is 'YES', put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many diseases in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{asked} Based on this information, create most relevant {n} questions to ask (and classify the above diseases). Your response should strictly follow the exact template:
Question 1: ...?
YES: comma-seperated, list of disease names, ...
Count of YES: ...
NO: comma-seperated, list of disease names, ...
Count of NO: ...
Question 2: ...?
and so on...
'''

# conversation
target_question = "Are you experiencing {target}?"
target_question_FA = "Are you experiencing {target}?"

###
targeting_prompt_free = """Note that you should point out and ask what disease the patient suffers from now. Refer the past conversation regarding the patient's symptoms. Note that the disease would be one of the initally given possibilities only.  
Respond with 1 new question only, follow the format: 'Are you experiencing [disease name]?'. Ensure that [disease name] was not asked before. """

targeting_prompt_free_FA = """Note that you should point out and ask what disease the patient suffers from now. Refer the past conversation regarding the patient's symptoms. Note that the disease would be one of the initally given possibilities only.  
Respond with 1 new question only, follow the format: 'Are you experiencing [disease name]?'. Ensure that [disease name] was not asked before. """

###
targeting_prompt_set = """Note that you should point out and ask what disease the patient suffers from now. The patient is very likely suffering from one of diseases below:
{item_list_str}, or others from the initial set of possibilities. Refer the past conversation about patient's symptoms. 
Respond with 1 new question only, follow the format: 'Are you experiencing [disease name]?'. """

targeting_prompt_set_FA = """Note that you should point out and ask what disease the patient suffers from now. The patient is very likely suffering from one of diseases below:
{item_list_str}, or others from the initial set of possibilities. Carefully review the patient's responses and make informed decisions. 
Respond with 1 new question only, follow the format: 'Are you experiencing [disease name]?'."""

###
guesser_prologue = '''You are a doctor and your patient self-reports that: {repo}. {closed_set_declaration} 
You should ask your patient questions in English regarding their symptoms which can only be answered by 'Yes' or 'No', in order to diagnose what disease this patient suffers. Carefully review the patient's problem and the ongoing conversation. Avoid redundant questions.  
Let us begin. Ask me the first question.
'''
guesser_prologue_FA = '''You are a doctor and your patient self-reports that: {repo}. {closed_set_declaration} 
You should ask your patient questions in English regarding their symptoms, in order to diagnose what disease this patient suffers. Carefully review the patient's problem and the ongoing conversation. Avoid redundant questions.  
Let us begin. Ask me the first question.
'''

urge_prompt = "Based on the symptons above, if you find out the disease, please ask 'Are you experiencing [disease name]?'"

inform_prompt = "The patient may suffer from one of diseases below:\n{item_list_str}"

# self report / free answer
classify_prompt = '''Here is a set of all possible diseases that the patient may suffer from:
{item_list_str}

{repo}
According to the patient's report, classify each given disease into 'YES: ...' if even slightly possible, otherwise to 'NO: ...' if completely unlikely. Only classify the diseases given in above set, do not add anything else. Your answer should be like:
YES: comma-seperated list of disease names, ...
NO: comma-seperated, list of disease names, ...'''

self_repo_prompt = '''The patient self-reports that: {repo}'''

free_answer_prompt = '''The doctor and patient's conversation:
{repo}
'''

simulator_prologue = '''You are a patient suffering from the disease of {item}, and communicating with a doctor.
Here is your conversation history with another doctor:
{conv_hist}

Remember the conversation above to answer the doctor's question in English and do not reveal the disease until the doctor correctly mentions or asks about it.
If the doctor mentions your disease in their question or asks whether you experience {item}, you must answer 'You are right. I have {item}.'."
'''

# examiner
examiner_prologue = '''You are the patient suffering from {item} and I am the doctor.
I will ask you upto 6 questions and you should answer each one truthfully based on your disease, by saying Yes or No.
Note that you must never reveal the disease, until I correctly mention it. 
If I mention your disease in my question or ask about its symptoms, then you must directly respond "You are right. I am experiencing {item}." while saying Yes.
Let us begin. Here is my first question.
'''

# open set
init_open_set_prompt = '''You are a doctor and your patient self-reports that: {repo}. Please propose {size} diseases that you think your patient may suffer from.
Your response should only be a list like this: ["disease1", "disease2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} diseases that your patient may suffer from.
The list of {size} diseases should contains {item_list}
Your response should only be a list like this: ["disease1", "disease2", ...]'''