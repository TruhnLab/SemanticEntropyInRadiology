# Goal of this Skript is split all the questions into tow files Long answer questions and short answer question
# input: 
# "questions/Questions_Semantic_Entropy_Image_Acquisition.csv",
# "questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",
# "questions/Questions_Semantic_Entropy_Imaging_Education.csv",
# "questions/Questions_Semantic_Entropy_Research.csv",
# output:
# "questions/Questions_SHORT.csv",
# "questions/Questions_LONG.csv",

O1_LIST = [
    "EVAL_Image_Acquisition_o1_unknownTemperature.csv",
    "EVAL_Guidelines_and_Indications_o1_unknownTemperature.csv",
    "EVAL_Imaging_Education_o1_unknownTemperature.csv",
    "EVAL_Research_prompt_o1_unknownTemperature.csv",
]

GPT4_LIST = [
    "EVAL_Image_Acquisition_GPT4o.csv",
    "EVAL_Guidelines_and_Indications_GPT4o.csv",
    "EVAL_Imaging_Education_GPT4o.csv",
    "EVAL_Research_GPT4o.csv",
]

LLAMA_70B_LIST = [
    "EVAL_Image_Acquisition_LLAMA3.1_70B.csv",
    "EVAL_Guidelines_and_Indications_LLAMA3.1_70B.csv",
    "EVAL_Imaging_Education_LLAMA3.1_70B.csv",
    "EVAL_Research_LLAMA3.1_70B.csv",

]

LLAMA_8B_LIST = [
    "EVAL_Image_Acquisition_LLAMA3.1_8B.csv",
    "EVAL_Guidelines_and_Indications_LLAMA3.1_8B.csv",
    "EVAL_Imaging_Education_LLAMA3.1_8B.csv",
    "EVAL_Research_LLAMA3.1_8B.csv",
]


ALL_LIST = [
    (O1_LIST,"o1_unknownTemperature"),
    (GPT4_LIST,"GPT4o"),
    (LLAMA_70B_LIST,"LLAMA3.1_70B"),
    (LLAMA_8B_LIST,"LLAMA3.1_8B")
]

SHORT_LONG_PROMPT = """You are given a question and its expected answer. Your task is to decide whether the expected answer consists of exactly one atomic fact or multiple combined facts.

Classification rules:
- Reply with 'short' if the answer can be expressed as one single fact.
- Reply with 'long' if the answer contains multiple facts, conditions, steps, or explanatory parts.

Output rules:
- Return only one word: either 'short' or 'long'.
- Do not return any explanation, punctuation, or extra text."""

from promptLLM import promptGPT_OSS_120B
from utilFunctions import readCSV,appendLineToCSV
from concurrent.futures import ThreadPoolExecutor

def isShort(questionLine):
    question = questionLine[1]
    trueAnswer = questionLine[2]
    addStr = f"\n\nQuestion: {question}\n\n True Answer: {trueAnswer}"
    prompt = SHORT_LONG_PROMPT + addStr
    out = promptGPT_OSS_120B(prompt,0.0)
    out = out[0]
    out = out.lower()
    if "short" in out and not "long" in out:
        return "short"
    if "long" in out and not "short" in out:
        return "long"
    assert False

def mapShortLong(listEl):
    pass


head = ["id","temperature","prompt","llmAnswer","question","answer","subtopic","reference","lastUpdate","logProb"]


if __name__ == "__main__":






    allQuestionList = list()
    allQuestionList.extend(readCSV("questions/Questions_Semantic_Entropy_Image_Acquisition.csv",delimiter=";")[1:])
    allQuestionList.extend(readCSV("questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",delimiter=";")[1:])
    allQuestionList.extend(readCSV("questions/Questions_Semantic_Entropy_Imaging_Education.csv",delimiter=";")[1:])
    allQuestionList.extend(readCSV("questions/Questions_Semantic_Entropy_Research.csv",delimiter=";")[1:])
    with ThreadPoolExecutor(max_workers=30) as threadPool:
        shortLongList = list(threadPool.map(isShort, allQuestionList))
        print(shortLongList)
    
    combinedList = list(zip(allQuestionList,shortLongList))

    for evalList,modelName in ALL_LIST:
        longShortDict = {
            "short":f"cache/SHORT_EVAL_{modelName}.csv",
            "long":f"cache/LONG_EVAL_{modelName}.csv"
        }
        [appendLineToCSV(x,head) for x in longShortDict.values()]
        contentList = list()
        for oneEvalList in evalList:
            contentList.extend(readCSV(f"cache/{oneEvalList}")[1:])
        #print(contentList)
        print(len(contentList))
        for line in contentList:
            for querryQuestion,querryLongShort in combinedList:
                #print(line)
                if querryQuestion[1] in line[2]:
                    appendLineToCSV(longShortDict[querryLongShort],line)
                    break
            else: #yes this is an else after a for loop
                assert False











    # for questionLine,answerLength in zip(allQuestionList,shortLongList):
    #     if answerLength == "short":
    #         appendLineToCSV("questions/Questions_SHORT.csv",questionLine,delimiter=";")
    #     elif answerLength == "long":
    #         appendLineToCSV("questions/Questions_LONG.csv",questionLine,delimiter=";")
    #     else:
    #         assert False

    print("DONE")






