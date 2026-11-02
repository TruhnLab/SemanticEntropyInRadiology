from RadDataset import RadDataset
import CONFIG as CONFIG
import promptLLM as promptLLM
import os
import csv
from tqdm import tqdm

# Generates multiple answers per question in a dataset and saves them to CSV

def execAnswerGeneration(datasetFile,outFile,promptFunc,promptPrefix):
    # Iterates through a dataset and queries the chosen LLM for each question

    radDataset = RadDataset(datasetFile)

    if not os.path.isfile(outFile): # generate output file if not existent
        with open(outFile,"a") as f: 
            f.write("id,temperature,prompt,llmAnswer,question,answer,subtopic,reference,lastUpdate,logProb\n")

    for questionAnswer in tqdm(radDataset):
        for i in range(CONFIG.PROMPT_SAMPLE_PER_QUESTION+1): # answer the the question CONFIG.PROMPT_SAMPLE_PER_QUESTION+1 times. Sample 0 is for accuracy calculation; the others for the entropy calculation 
            temperature = CONFIG.LOW_TEMPERATURE if i==0 else CONFIG.HIGH_TEMPERATURE
            prompt = promptPrefix + questionAnswer["question"]
            llmAnswer,logProb = promptFunc(prompt,temperature)
            questionEval = [
                str(i),
                str(temperature),
                prompt,
                llmAnswer,
                questionAnswer["question"],
                questionAnswer["answer"],
                questionAnswer["subtopic"],
                questionAnswer["reference"],
                questionAnswer["lastUpdate"],
                str(logProb)

            ]
            with open(outFile,"a",newline="") as f: 
                csv.writer(f).writerow(questionEval)
#Example for how it could be used     
EVAL_LIST = [
    # Collection of dataset paths, output CSVs, prompt functions, and prompt prefixes
    #Questions_Semantic_Entropy_Image_Acquisition.csv
    (
        "questions/Questions_Semantic_Entropy_Image_Acquisition.csv",
        "cache/EVAL_Image_Acquisition_LLAMA3.1_8B.csv",
        promptLLM.promptLlama3_1_8B,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Image_Acquisition.csv",
        "cache/EVAL_Image_Acquisition_LLAMA3.1_70B.csv",
        promptLLM.promptLlama3_1_70B,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Image_Acquisition.csv",
        "cache/EVAL_Image_Acquisition_GPT4o.csv",
        promptLLM.promptGPT4o,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Image_Acquisition.csv",
        "cache/EVAL_Image_Acquisition_o1_unknownTemperature.csv",
        promptLLM.prompt_o1_unknownTemperature,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    #Questions_Semantic_Entropy_Guidelines_and_Indications.csv
        (
        "questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",
        "cache/EVAL_Guidelines_and_Indications_LLAMA3.1_8B.csv",
        promptLLM.promptLlama3_1_8B,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",
        "cache/EVAL_Guidelines_and_Indications_LLAMA3.1_70B.csv",
        promptLLM.promptLlama3_1_70B,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",
        "cache/EVAL_Guidelines_and_Indications_GPT4o.csv",
        promptLLM.promptGPT4o,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    (
        "questions/Questions_Semantic_Entropy_Guidelines_and_Indications.csv",
        "cache/EVAL_Guidelines_and_Indications_o1_unknownTemperature.csv",
        promptLLM.prompt_o1_unknownTemperature,
        CONFIG.PROMPT_PREFIX_GUIDELINES,
    ),
    #Questions_Semantic_Entropy_Imaging_Education.csv
    (
        "questions/Questions_Semantic_Entropy_Imaging_Education.csv",
        "cache/EVAL_Imaging_Education_LLAMA3.1_8B.csv",
        promptLLM.promptLlama3_1_8B,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Imaging_Education.csv",
        "cache/EVAL_Imaging_Education_LLAMA3.1_70B.csv",
        promptLLM.promptLlama3_1_70B,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Imaging_Education.csv",
        "cache/EVAL_Imaging_Education_GPT4o.csv",
        promptLLM.promptGPT4o,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Imaging_Education.csv",
        "cache/EVAL_Imaging_Education_o1_unknownTemperature.csv",
        promptLLM.prompt_o1_unknownTemperature,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    #Questions_Semantic_Entropy_Research.csv
    (
        "questions/Questions_Semantic_Entropy_Research.csv",
        "cache/EVAL_Research_LLAMA3.1_8B.csv",
        promptLLM.promptLlama3_1_8B,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Research.csv",
        "cache/EVAL_Research_GPT4o.csv",
        promptLLM.promptGPT4o,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Research.csv",
        "cache/EVAL_Research_LLAMA3.1_70B.csv",
        promptLLM.promptLlama3_1_70B,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),
    (
        "questions/Questions_Semantic_Entropy_Research.csv",
        "cache/EVAL_Research_prompt_o1_unknownTemperature.csv",
        promptLLM.prompt_o1_unknownTemperature,
        CONFIG.PROMPT_PREFIX_EDU_RESEARCH,
    ),

]

for answerSet in EVAL_LIST:
    # Executes answer generation for each configuration
    print("GENERATE ANSWERS:",*answerSet)
    execAnswerGeneration(*answerSet)