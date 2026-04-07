import random
from datetime import datetime
import CONFIG
from pathlib import Path
from openai import AzureOpenAI,OpenAI
from utilFunctions import readCSV,appendLineToCSV
import time
import ollama
import httpx

# Provides prompt functions for GPT and Llama models

#Replace with your access
#CONFIG.OPENAI_KEY,CONFIG.AZURE_ENDPOINT = "...","..."
#CONFIG.LOCAL_KEY,CONFIG.LOCAL_ENDPOINT = "...","..."

AZURE_CLIENT = AzureOpenAI(
        api_key=CONFIG.OPENAI_KEY,  
        api_version="2024-03-01-preview",
        azure_endpoint=CONFIG.AZURE_ENDPOINT
        )

LOCAL_CLIENT = OpenAI(
    base_url=CONFIG.LOCAL_ENDPOINT,
    api_key=CONFIG.LOCAL_KEY,
    http_client=httpx.Client(verify=False),
)

def logPrompt(func):
    # Decorator that logs prompts, responses, and usage for each query
    def wrapper(*args, **kwargs):
        result,logProb,response = func(*args, **kwargs)
        if type(response) == dict:
            tokenCount = None
        else:
            tokenCount = 0 if response is None else response.usage.total_tokens
        funcName = func.__name__
        prompt, temperature = args
        timeStamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not Path(CONFIG.PROMPT_LOG_FILE_PATH).exists(): appendLineToCSV(CONFIG.PROMPT_LOG_FILE_PATH,["prompt","result","temperature","funcName","timeStamp","tokenCount","logProb"])
        appendLineToCSV(CONFIG.PROMPT_LOG_FILE_PATH,[
            prompt,
            result,
            str(temperature),
            funcName,
            timeStamp,
            str(tokenCount),
            str(logProb),
        ])
        return result,logProb
    return wrapper

@logPrompt
def promptGPT4o(prompt,temperature):
    # Sends a prompt to GPT-4 via Azure OpenAI
    model = "gpt-4o"
    messages = [{"role": "user", "content": prompt}]
    for i in range(CONFIG.NUM_RETRYS):
        try:
            response = AZURE_CLIENT.chat.completions.create(
                model=model, #
                messages=messages,
                max_tokens=CONFIG.MAX_TOKEN,  # The maximum number of tokens to generate in the completion. WARNING: Rate limit depends on this, no matter how many tokens were actually needed
                n=1, # How many completions to generate for each prompt.
                temperature=temperature, # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                top_p=0.9, # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
                logprobs=True,
            )
            break
        except Exception as e:
            print(f"OpenAI not responding: {e}. Retry ...")
            time.sleep(5)
    #print(response)
    if response.choices[0].finish_reason == 'length':
        print("Warning: Output may be incomplete due to token limit.")
    answer = response.choices[0].message.content
    answer = answer.strip()
    logProb = sum([x.logprob for x in response.choices[0].logprobs.content])
    print("++++REQUEST++++, model:",model,"temp:",temperature,"token:",response.usage.total_tokens,"logProb:",logProb)
    print("++++PROMPT+++++",prompt)
    print("+++RESPONSE++++",answer)

    return answer,logProb,response

@logPrompt
def promptLlama3_1_8B(prompt,temperature):
    # Sends a prompt to an 8B-parameter Llama 3.1 model via Ollama
    response = ollama.chat(model='llama3.1:8b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ],
        options={
            "top_k": 20,
            "top_p": 0.9,
            "temperature": temperature,
        }
        )
    answer = response["message"]["content"]
    
    return answer,None, response

@logPrompt
def promptLlama3_1_70B(prompt,temperature):
    # Sends a prompt to a 70B-parameter Llama 3.1 model via Ollama
    response = ollama.chat(model='llama3.1:70b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ],
        options={
            "top_k": 20,
            "top_p": 0.9,
            "temperature": temperature,
        }
        )
    answer = response["message"]["content"]
    
    return answer,None, response

@logPrompt
def prompt_o1_unknownTemperature(prompt,temperature):
    # Sends a prompt to the "o1-preview" model, ignoring the temperature parameter
    model = "o1-preview"
    messages = [{"role": "user", "content": prompt}]
    for i in range(CONFIG.NUM_RETRYS):
        try:
            response = AZURE_CLIENT.chat.completions.create(
                model=model,
                messages=messages,
            )
            if response.choices[0].finish_reason == 'length':
                print("Warning: Output may be incomplete due to token limit.")
            answer = response.choices[0].message.content
            answer = answer.strip()
            break
        except Exception as e:
            print(f"OpenAI not responding: {e}. Retry ...")
            time.sleep(5)

    print("++++REQUEST++++, model:",model,"temp:",temperature,"TEMPERATURE IS A DUMMY PARAMETER","token:",response.usage.total_tokens)
    print("++++PROMPT+++++",prompt)
    print("+++RESPONSE++++",answer)

    return answer,None,response


@logPrompt
def promptGPT_OSS_120B(prompt,temperature):
    model = "openai/gpt-oss-120b"
    messages = [{"role": "user", "content": prompt}]
    for i in range(CONFIG.NUM_RETRYS):
        try:
            response = LOCAL_CLIENT.chat.completions.create(
                model=model, #
                messages=messages,
                #max_tokens=CONFIG.MAX_TOKEN_LOCAL,  # The maximum number of tokens to generate in the completion. WARNING: Rate limit depends on this, no matter how many tokens were actually needed
                n=1, # How many completions to generate for each prompt.
                temperature=temperature, # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
                top_p=0.9, # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
                logprobs=True,
            )
            break
        except Exception as e:
            print(f"LOCAL_API not responding: {e}. Retry ...")
            time.sleep(5)
    #print(response)
    if response.choices[0].finish_reason == 'length':
        print("Warning: Output may be incomplete due to token limit.")
    answer = response.choices[0].message.content
    answer = answer.strip()
    logProb = sum([x.logprob for x in response.choices[0].logprobs.content])
    print("++++LOCAL REQUEST++++, model:",model,"temp:",temperature,"token:",response.usage.total_tokens,"logProb:",logProb)
    print("++++LOCAL PROMPT+++++",prompt)
    print("+++LOCAL RESPONSE++++",answer)

    return answer,logProb,response


if __name__ == "__main__":
    from tqdm import tqdm
    for i in tqdm(range(100)):
        promptGPT_OSS_120B(f"Frage Nr {i}: wer bist du?",1.0)