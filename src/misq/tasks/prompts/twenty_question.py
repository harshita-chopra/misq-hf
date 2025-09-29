from misq.tasks.prompts.general import *

# method
generate_prompt_rest = '''Here are all the X:
{items_str}

{n} questions are designed to classify the possible X above based on the answer for these question:
{asked}
For each X under each question, if the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''Here are all the X:
{items_str}

Design a question about X that can only be answered by YES or NO. Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. When not sure, put in YES. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{asked} Based on this information, create most relevant {n} questions to classify the above X correctly. Your response should strictly follow the exact template:
Question 1: Is X ...?
YES: comma-seperated, list of things, ...
Count of YES: ...
NO: comma-seperated, list of things, ...
Count of NO: ...
Question 2: Is X ...?
'''

# conversation
target_question = "Is X a '{target}'?"

targeting_prompt_free = """Note that you should guess and ask what X exactly is from now on.
Use the past conversation for context to avoid redundant questions. 
The question must start with 'Is X ...'"""

targeting_prompt_set = """Note that you should guess and ask what X exactly is from now on. X is possibly one of the following:
{item_list_str}, or other from the initial set of possibilities.
Use the past conversation for context to avoid redundant questions. 
The question must start with 'Is X ...'"""

guesser_prologue = '''Let us play the game of 20 questions. I am impersonating the thing, X. {closed_set_declaration} 
You will ask me up to 20 questions which start with 'Is X' and can only be answered by yes or no, and I will answer each one truthfully based on being X.
Let us begin. Ask me the first question.
'''

urge_prompt = ""

inform_prompt = "The X may be one of items below:\n{item_list_str}"


# examiner
examiner_prologue = '''Let us play the game of 20 questions. You are the answerer and I am guesser. X is '{item}'. 
I will ask you up to 20 questions and you should answer each one truthfully based on being X, by saying Yes or No. 
Note that you must never reveal X, until I guess it correctly. 
If I guess X correctly in my question, directly respond "You guessed it. X is '{item}'." instead of saying Yes.
Let us begin. Here is my first question.
'''

# open set
init_open_set_prompt = '''You are playing the game of 20 questions and you are the guesser. Based on the conversation history, please propose {size} things that you think the answerer might have in mind.
Your response should only provide the list in the following format: ["thing1", "thing2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} things that the answerer of 20 question game might have in mind.
The list of {size} things should contains {item_list}
Your response should only provide the list in the following format: ["thing1", "thing2", ...]'''