import os
import time

# my code
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
# Create an Amazon Bedrock Runtime client.
brt = boto3.client("bedrock-runtime", region_name='us-west-2', config=Config(read_timeout=5000))

# url = "http://localhost:11434/api/chat"
# def llama3(prompt, max_tokens=250):
#     data = {
#         "model": "llama3",
#         "messages": prompt,
#         "stream": False,
#         "options": {"temperature":0.7, "num_predict":max_tokens}
#         }
#     headers = {"Content-Type": "application/json"}
#     response = requests.post(url, headers=headers, json=data)
#     return response.json()["message"]["content"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
time_gap = {"gpt-4o": 1, "gpt-3.5-turbo": 0.5}
if OPENAI_API_KEY != "":
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"OPENAI_API_KEY: ****{OPENAI_API_KEY[-4:]}")
else:
    print("Warning: OPENAI_API_KEY is not set")


def gpt_response(message: list, model="gpt-4o", temperature=0, max_tokens=500):
    # time.sleep(time_gap.get(model, 3))
    for m in message:
        if len(m['content'])>6000:
            print("\nToo long, quitting msg: ",m['content'])
            exit(1)
    try:
        res = client.chat.completions.create(model=model, messages=message, temperature=temperature, n=1,
                                             max_tokens=max_tokens)
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        print(message)
        time.sleep(time_gap.get(model, 3) * 2)
        exit(1)
        # return gpt_response(message, model, temperature, max_tokens)


def claude_response(message, model="claude-3-sonnet-20240229", temperature=0, max_tokens=500):
    msg = []
    model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    for m in message:
        role = m["role"] if m["role"] == "user" else "assistant"
        if msg and msg[-1]["role"] == role:
            msg[-1]["content"][0]["text"] += m["content"]
        else:
            msg.append({"role": role, "content": [{"text": m["content"]}]})
    try:
        # res = claude_client.messages.create(
        #     model=model,
        #     max_tokens=max_tokens,
        #     messages=msg
        # )
        # return res.content[0].text
        response = brt.converse(
            modelId=model_id,
            messages=msg,
            inferenceConfig={"maxTokens": max_tokens} )
        return response["output"]["message"]["content"][0]["text"]
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        print("messages: ",message)
        # exit(1)
        time.sleep(3)
        return claude_response(message, model, temperature, max_tokens)


def llama_response(message, model=None, temperature=0.1, max_tokens=4000):
    try:
        # my code
        # response = llama3(message, max_tokens)
        # return response

        # using boto3
        model_id = "us.meta.llama3-3-70b-instruct-v1:0" # "meta.llama3-3-70b-instruct-v1:0"
        conversation = []
        for m in message:
            m_role = m["role"] if m["role"] == "user" else "assistant"
            if conversation and conversation[-1]["role"]== m_role:
                conversation[-1]["content"][0]["text"] += m["content"]
            else:
                conversation.append({"role": m_role, "content": [{"text": m["content"]}]})
        response = brt.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, 
                            "temperature": temperature} )
        return response["output"]["message"]["content"][0]["text"]

        # chat_completion = llama_client.chat.completions.create(
        #     messages=message,
        #     model="meta-llama/Llama-2-70b-chat-hf",
        #     max_tokens=max_tokens
        # )
        # return chat_completion.choices[0].message.content
    
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        print("messages: ",message)
        exit(1)
        time.sleep(1)
        llama_response(message, model, temperature, max_tokens)


def mistral_response(message, model="mistral-large-latest", temperature=0, max_tokens=500):
    try:
        # using boto3
        model_id = "mistral.mixtral-8x7b-instruct-v0:1" 
        conversation = []
        for m in message:
            m_role = m["role"] if m["role"] == "user" else "assistant"
            if conversation and conversation[-1]["role"]== m_role:
                conversation[-1]["content"][0]["text"] += m["content"]
            else:
                conversation.append({"role": m_role, "content": [{"text": m["content"]}]})
        response = brt.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": max_tokens, 
                            "temperature": temperature} )
        return response["output"]["message"]["content"][0]["text"]

    except (ClientError, Exception) as e:
        print(f"ERROR: Bedrock: Can't invoke '{model_id}'. Reason: {e}")
        print("messages: ",message)
        exit(1)
        time.sleep(1)
        mistral_response(message, model, temperature, max_tokens)

def get_response_method(model):
    response_methods = {
        "gpt": gpt_response,
        "claude": claude_response,
        "llama": llama_response,
        "mixtral": mistral_response,
    }
    return response_methods.get(model.split("-")[0], lambda _: NotImplementedError())
