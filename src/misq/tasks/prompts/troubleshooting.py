from misq.tasks.prompts.general import *

# method
generate_prompt_rest = '''You are a technician. Here are all issues that the client may face:
{items_str}

{n} questions are designed to classify the possible issues above based on the answer for these question:
{asked}
For each issue under each question, if the answer is 'YES', put this issue into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many issues in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''You are a technician. Here are all issues that the client may face:
{items_str}

Design a question to ask your client with specific situation that can only be answer by YES or NO. Then classify the possible issue above based on this question. If the answer is 'YES', put this issue into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many issues in YES and NO.
Notably, this question should try to fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{asked} Based on this information, create most relevant {n} questions to classify the above issues correctly. Your response should strictly follow the exact template:
Question 1: ...?
YES: comma-seperated, list of issue names, ...
Count of YES: ...
NO: comma-seperated, list of issue names, ...
Count of NO: ...
'''


# conversation
target_question = "Are you experiencing {target}?"

targeting_prompt_free = """Note that you should now point out and ask what issue the client is facing. Refer the past conversation for problem context.
Respond with 1 new question only, follow the format: 'Are you experiencing [issue name]?'. Ensure that [issue name] was not asked before. """

targeting_prompt_set = """Note that you should now point out and ask what issue the client is facing. The client is likely to be facing one of the issues below:
{item_list_str}, so you must consider these. Refer the past conversation for problem context.
Respond with 1 new question only, follow the format: 'Are you experiencing [issue name]?'. Ensure that [issue name] was not asked before. """

guesser_prologue = '''You are a technician, and your client self-reports that: {repo}. {closed_set_declaration} 
You should ask your client question with specific situation which can only be answered by Yes or No, in order to find which issue this client is facing.  Use the ongoing conversation for context to avoid redundant questions.  
Let us begin. Ask me the first question.
'''

urge_prompt = "Based on the situations above, if you find out the issue, please ask 'Are you experiencing [issue name]?'"

inform_prompt = "The client may face one of issues below:\n{item_list_str}"

# self report / free answer
classify_prompt = '''{repo}

Here are all issues that the client may face:
{item_list_str}

According to the user's problem, classify each possible issue given in the above set into 'YES: ...' if even slightly possible, otherwise to 'NO: ...' if absolutely unlikely. Only classify the issues in above set, do not add any other. Your answer should be like:
YES: comma-seperated list of issue names, ...
NO: comma-seperated, list of issue names, ...'''

self_repo_prompt = '''The client self-reports that: {repo}'''

# examiner
examiner_prologue = '''You are the client with a device that has {item} and I am the technician.
I will ask you up to 20 questions and you should answer each one truthfully based on the issue of your device, by saying Yes or No.
Note that you must never reveal the issue name, until I correctly mention or ask about it. 
If I mention the issue in my question or closely ask its possibility, directly respond "You are right. My device has {item}." while saying Yes.
Let us begin. Here is my first question.
'''

# open set
init_open_set_prompt = '''You are a technician, and your client self-reports that: {repo}. Please propose {size} issues that you think your client may face with.
Your response should only be a list like: ["issue1", "issue2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} issues that your client may face with.
The list of {size} issues should contains {item_list}
Your response should only be a list like: ["issue1", "issue2", ...]'''