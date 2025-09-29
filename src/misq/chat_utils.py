import copy
import importlib
import traceback
import ast

from misq.models import get_response_method

task_parameter_mapping = {
    "20q": "twenty_question",
    "md": "medical_diagnosis",
    "tb": "troubleshooting",
    "sp": "sp",
}


def import_prompts_by_task(task_name):
    parameter = task_parameter_mapping.get(task_name)
    module_name = f"misq.tasks.prompts.{parameter}"
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        raise ImportError(f"Failed to import module: {module_name}")


def ques_and_cls_given_items(task, items: list, n, asked_ques: list = None, parent_context: list = None, rest=False):
    response = get_response_method(task.guesser_model)
    if len(items) <= 1:
        return None

    if rest:
        asked = '\n'.join([f"Question {i + 1}: {asked_ques[i]}" for i in range(len(asked_ques))])
        message = [{"role": "user", "content": task.prompts.generate_prompt_rest.format(
            items_str=', '.join(items), n=n, asked=asked, Q1=asked_ques[0])}]
    else:
        # asked = "(The question should not be '" + "' or '".join(asked_ques) + "')" if asked_ques else ""
        #### Send parents for context
        parent_context = f"For context, following questions were already asked to build the above set of possibilities: {' ; '.join(parent_context)}. Do not repeat these." if parent_context else ""
        asked_ques = f"The question should not be {' or '.join(asked_ques)}" if asked_ques else ""
        asked = f"{parent_context} {asked_ques} \n" if (asked_ques or parent_context) else ""
        repo = task.prompts.self_repo_prompt.format(repo=task.repo) if task.repo else ""
        #### added context if it exists
        message = [{"role": "user", "content": task.prompts.generate_prompt.format(items_str=', '.join(items), n=n, asked=asked, repo=repo)}]
    rsp = "#" + response(message, model=task.guesser_model, max_tokens=5000)

    def process_ans(rspp):
        ans = []
        # Remove any markdown formatting
        rspp = rspp.replace('**', '').replace('#', '').replace('\n\n', '\n')
        
        # If task is 'sp', parse dict format
        if task.task == 'sp':
            import re, ast
            questions = rspp.split('Question')[1:]
            for q in questions:
                q_parts = q.split('\n')
                question = q_parts[0].split(': ', 1)[1].strip()
                remaining = '\n'.join(q_parts[1:])
                # YES dict
                yes_match = re.search(r'YES: (\{.*?\})', remaining, re.DOTALL)
                no_match = re.search(r'NO: (\{.*?\})', remaining, re.DOTALL)
                items_yes = []
                items_no = []
                if yes_match:
                    try:
                        yes_dict = ast.literal_eval(yes_match.group(1))
                        items_yes = list(yes_dict.values())
                    except Exception:
                        pass
                if no_match:
                    try:
                        no_dict = ast.literal_eval(no_match.group(1))
                        items_no = list(no_dict.values())
                    except Exception:
                        pass
                ans.append({
                    "question": question,
                    "items_yes": items_yes,
                    "items_no": items_no
                })
            return ans
        
        # Split by Question pattern
        questions = rsp.split('Question')[1:]  # Skip the first split as it's before Question 1

        for q in questions:
            # Extract question text
            q_parts = q.split('\n')
            question = q_parts[0].split(': ')[1].strip()
            
            # Get the remaining text
            remaining = '\n'.join(q_parts[1:])
            
            # Split for YES and NO sections, handling multiple newlines
            yes_part = remaining.split('YES:', 1)[1].split('Count of YES:', 1)[0]
            no_part = remaining.split('NO:', 1)[1].split('Count of NO:', 1)[0]
            
            # Clean and split items
            items_yes = [item.strip() for item in yes_part.strip().split(',') if item.strip()!='']
            items_no = [item.strip() for item in no_part.strip().split(',') if item.strip()!='']
            
            # Remove any empty items and convert to set to remove duplicates
            items_yes = list(set(filter(None, items_yes)))
            items_no = list(set(filter(None, items_no)))
            
            ans.append({
                "question": question,
                "items_yes": items_yes,
                "items_no": items_no
            })
        return ans

    def format_rsp(rspp):
        print("\nTrying to format prompt...")
        gpt3_response = get_response_method("llama-3-3-70b-instruct")  # task.guesser_model
        message.append({"role": "system", "content": rspp})
        message.append({"role": "user", "content": task.prompts.format_generated_prompt.format(rspp=rspp)})
        return gpt3_response(message, "llama-3-3-70b-instruct",   #  task.guesser_model
                             max_tokens=4000)

    try:
        return process_ans(rsp)
    except Exception:
        print(f"Error processing response: {rsp}\n{traceback.print_exc()}")
        try:
            rsp = format_rsp(rsp)
            return process_ans(rsp)
        except Exception as e:
            print(f"Error: {e}") # \n {traceback.print_exc()}")
            print("Trying this again...")
            return ques_and_cls_given_items(task, items, n, asked_ques, rest)


def cls_given_repo(task, items: list, repo, translate=False, self_repo=True):

    if task.fullset: # no classification
        # print(f"Full set classification for {task.task} task, returning items as is.")
        return {"items_yes": items, "items_no": items}
    
    # constrained set generation with prompting
    response = get_response_method(task.guesser_model)
    if self_repo:
        if translate:
            message = [{"role": "user", "content": f"Translate to English: {repo}"}]
            gpt3_response = get_response_method(task.guesser_model)  
            repo = gpt3_response(message, task.guesser_model,   
                                 max_tokens=2000)
        repo = task.prompts.self_repo_prompt.format(repo=repo)
    else:
        repo = task.prompts.free_answer_prompt.format(repo=repo)
    message = [{"role": "user", "content": task.prompts.classify_prompt.format(item_list_str=', '.join(items), repo=repo)}]
    rsp = response(message, model=task.guesser_model, max_tokens=len(items)*(task.expected_target_tokens+5))

    def extract_items(rsp, keyword):
        _items = []
        if getattr(task, 'task', None) == 'sp':
            import ast, re
            match = re.search(rf'{keyword}\s*(\{{.*?\}})', rsp, re.DOTALL) 
            if match:
                try:
                    d = ast.literal_eval(match.group(1))
                    _items = list(d.values())
                except Exception:
                    _items = []
            return _items
        if keyword in rsp:
            rsp_part = rsp.split(keyword, 1)[1]
            if not rsp_part or rsp_part[0] != '\n':
                _items = rsp_part.split("\n", 1)[0].split(", ")
                _items = [item for item in _items if item!='']
                _items = list(set(_items))
        return _items

    try:
        items_y = extract_items(rsp, "YES: ")
        items_n = extract_items(rsp, "NO: ")
        if len(items_y) == 0 and len(items_n) == 0:
            raise ValueError("ERROR: No items extracted from the response.")

        return {"items_yes": items_y, "items_no": items_n}

    except Exception as e:
        print(f"Error processing classification response: {rsp}\n{traceback.print_exc()}")
        return {"items_yes": list(set(items)), "items_no": []}


def initialize_open_set(task, repo=""):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    if isinstance(repo, str):
        message = [{"role": "user", "content": task.prompts.init_open_set_prompt.format(repo=repo, size=size)}]
    else:
        message = repo + [{"role": "user", "content": task.prompts.init_open_set_prompt.format(size=size, repo=task.repo)}]
    rsp = response(message, model=task.guesser_model, max_tokens=100*size)
    # print([rsp])
    try:
        if getattr(task, 'task', None) == 'sp':
            import ast
            d = ast.literal_eval(rsp)
            return list(d.values())
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(f"Error: {e}\n{traceback.print_exc()}")
        # Extra: Safely parse rsp with ast.literal_eval if eval fails
        try:
            rsp = set(rsp.strip("[]").split(", "))
            return list(rsp)
        except (SyntaxError, ValueError) as ast_error:
            print(f"Failed with ast.literal_eval as well: {ast_error}")
        return initialize_open_set(task, repo)


def renew_open_set(task, history, items):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    message = copy.deepcopy(history) + [{"role": "user", "content": task.prompts.renew_open_set_prompt.format(size=size, item_list=str(items))}]
    rsp = response(message, model=task.guesser_model, max_tokens=100*size)
    # print([rsp])
    try:
        if getattr(task, 'task', None) == 'sp':
            import ast
            d = ast.literal_eval(rsp)
            return list(d.values())
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(f"Error: {e} \n {traceback.print_exc()}")
        return renew_open_set(task, history, items)