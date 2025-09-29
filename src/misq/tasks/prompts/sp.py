from misq.tasks.prompts.general import *

# ----- Generation Prompts -----

generate_prompt = '''You are a detective investigating a puzzling situation:
"{repo}"

You are considering the following {n} possible explanations:
{items_str}

So far, you've asked:
{asked}

Now propose {n} new YES/NO questions that would help you eliminate or confirm explanations. 
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
Based on the scenario described, classify each explanation into YES: if it is plausible or consistent with the facts, or NO: if it is implausible or inconsistent with the facts.

Each question should clearly split the explanations into YES and NO groups, with the counts as balanced as possible.

Format your answer like this:
Question 1: ...?
YES: {{"1": "explanation1", "2": "explanation2", ...}}
Count of YES: X
NO: {{"1": "explanation3", "2": "explanation4", ...}}
Count of NO: Y
Question 2: ...?

Output:
'''

generate_prompt_rest = '''You are continuing an investigation. The possible explanations are:
{items_str}

So far, you’ve asked:
{asked}

Suggest {n} additional YES/NO questions to further separate these explanations.
Based on the scenario described, classify each explanation into YES: if it is plausible or consistent with the facts, or NO: if it is implausible or inconsistent with the facts.

Each question should aim to divide the set as evenly as possible.

Format your answer like this:
Question 1: ...?
YES: {{"1": "explanation1", "2": "explanation2", ...}}
Count of YES: X
NO: {{"1": "explanation3", "2": "explanation4", ...}}
Count of NO: Y
Question 2: ...?

Output:
'''

classify_prompt = '''You are solving a mysterious situation:
"{repo}"

Here is a list of possible explanations:
{item_list_str}

Based on the scenario described, classify each explanation as follows:
- YES: if it is plausible or consistent with the facts
- NO: if it is implausible or inconsistent with the facts

Only use the explanations provided above. Your response should follow this format:
YES: {{"1": "explanation1", "2": "explanation2", ...}}
NO: {{"1": "explanation3", "2": "explanation4", ...}}

Output:
'''

# ----- Targeting Prompts (Final) -----

targeting_prompt_set = """Based on the conversation so far, you are ready to make your final attempt to solve the mystery.

Select one explanation from below and confirm it with the following question:
'Is the explanation: "[explanation]"?'

Choose from:
{item_list_str}

If the given explanations has been asked before, and the answer was NO, you must not repeat it. 
Based on the conversation, you must respond with the most likely explanation that has not been asked yet.

Output:
"""

targeting_prompt_set_FA = targeting_prompt_set

targeting_prompt_free = """Based on the conversation so far, you are now ready to make your final attempt to solve the mystery. 
Your task is to convey your explanation. It should be a detailed, coherent hypothesis, a scenario described in a few sentences that explains what likely happened and the reason behind it. 
Leverage the feedback. Avoid repeating ideas that have been denied. Focus on unique, plausible explanations. Think differently and creatively, making every turn count.
To confirm whether your explanation is correct or not, you must respond in the following format: 
'Is this the explanation behind the mystery?: "[your most likely explanation]"'

Ouptput:
"""

targeting_prompt_free_FA = targeting_prompt_free


target_question = "Is the explanation: '{target}'?"
target_question_FA = "Is the explanation: '{target}'?"

# ----- Guesser Prologues -----

guesser_prologue = '''You are a detective trying to understand a strange scenario:
"{repo}"

You suspect one of several possible explanations is true. Your task is to figure out which.

Initially, you ask a series of YES/NO questions to uncover the facts. After seeking enough information, you will make an attempt to resolve the mystery and give an explanation.
Be deliberate, avoid repeating ideas, and focus on deriving the explanation.

Start the investigation with your first question.
'''

guesser_prologue_FA = guesser_prologue

# ----- Self Report & Free Answer -----

self_repo_prompt = '''You are investigating the following case:
"{repo}"
'''

free_answer_prompt = '''You are solving a mystery. So far, you’ve asked:
{repo}

Use this context as you continue asking YES/NO questions.
'''

# ----- Patient/Simulator Prologue (Answerer) -----

simulator_prologue = '''You know the true explanation for the mystery:
"{item}"

Here is a structured trace of how and why it happened:
{conv_hist}

You are now being questioned by a detective.
Answer all YES/NO questions truthfully based on this context.

If the detective directly confirms the explanation by asking:
'Is this the explanation behind the mystery?: "[their explanation]"'

And if it is correct or close to correct, you must respond:
"You are right. Here’s what happened: {item}"

Otherwise, only answer YES or NO, and do not reveal any part of the explanation unless asked directly.
'''



# ----- Open-set prompts -----

init_open_set_prompt = '''You are a detective analyzing a puzzling event:
"{repo}"

Propose {size} plausible, natural and distinct explanations that could account for it.
Each explanation should be a detailed, coherent hypothesis, like a scenario described in one long sentence that explains what likely happened, why, and how.

Respond strictly in the following format, without any additional text or formatting:
{{"1": "explanation1", "2": "explanation2", ...}}

Output:
'''

renew_open_set_prompt = '''Given the current investigation:

Update your list of {size} explanations based on what you’ve learned so far. Each explanation should be a detailed, coherent hypothesis, like a scenario described in one long sentence that explains what likely happened, why, and how.

Make sure it includes:
{item_list}

Respond strictly in the following format, without any additional text or formatting:
{{"1": "explanation1", "2": "explanation2", ...}}

Output:
'''

# ----- Inform and Urge Prompts -----

inform_prompt = "Current candidate explanations:\n{item_list_str}"

urge_prompt = "If you are confident in one explanation, ask:\n'Is the explanation: \"[explanation]\"?'"
