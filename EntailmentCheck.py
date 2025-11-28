import promptLLM as promptLLM
import CONFIG
from utilFunctions import readCSV,appendLineToCSV
import os
import math
ENTAILMENT_PROMPT_FUNC = promptLLM.promptGPT4o
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
NEUTRAL = "neutral"
LLM_ANSWERS = [
    ENTAILMENT,
    CONTRADICTION,
    NEUTRAL,
]

PROMPT_FUNC_TO_CACHE_FILE = {
    promptLLM.promptGPT4o:CONFIG.ENTAILMENT_CACHE_FILE_GPT_4O,
    #"TODO_ADD_PROMPT_FUNC_o1":CONFIG.ENTAILMENT_CACHE_FILE_O1,
}

class EntailmentCheck():
    # This class handles entailment checks for user answers, caching results to avoid repeated prompts.

    def __init__(self,promptFunc=ENTAILMENT_PROMPT_FUNC,cacheFile=None,noNewPrompts = False):
        # Initializes the entailment check with a given prompt function, cache file, and option to disable new prompts
        if cacheFile is None:
            cacheFile = PROMPT_FUNC_TO_CACHE_FILE[promptFunc]
        self.promptFunc = promptFunc
        self.cacheFile = cacheFile
        self.noNewPrompts = noNewPrompts
        self.loadCache()
    
    def loadCache(self):
        # Loads cached prompt-answer pairs from disk
        if not os.path.isfile(self.cacheFile):
            self.cacheDict = dict()
        else:
            cacheContent = readCSV(self.cacheFile)
            self.cacheDict = {x[0]:x[1] for x in cacheContent if len(x) == 2}
            #print("IGNORED LINES IN CACHE",len([1 for x in cacheContent if len(x) != 2]))
    
    def toCache(self,prompt,answer):
        # Updates both the local cache dictionary and the CSV cache file with a new prompt-answer entry
        self.cacheDict[prompt] = answer
        appendLineToCSV(self.cacheFile,[prompt,answer])

    def getEntailmentPrompt(self,question,text1,text2):
        # Builds a system prompt asking if text1 semantically entails text2 for a given question
        text1 = text1.strip()
        text2 = text2.strip()
        question = question.strip()
        prompt = f"We are evaluating answers to the question \"{question}\"\n"
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."
        return prompt
    
    def newPromptRequest(self,prompt):
        # Issues a fresh API call to obtain entailment results when noNewPrompts is False
        assert not self.noNewPrompts
        return ENTAILMENT_PROMPT_FUNC(prompt,CONFIG.ENTAILMENT_TEMPERATURE)[0]
    
    def execPrompt(self,prompt):
        # Retrieves the entailment result from cache when possible, otherwise calls newPromptRequest
        if prompt in self.cacheDict:
            promptAnswer = self.cacheDict[prompt]
        else:
            llmAnswer = self.newPromptRequest(prompt)
            self.toCache(prompt,llmAnswer)
            promptAnswer = llmAnswer
        return promptAnswer

    def testEntailment(self,question,text1,text2):
        # Returns True if text1 semantically entails text2; otherwise False
        prompt = self.getEntailmentPrompt(question,text1,text2)
        promptAnswer = self.execPrompt(prompt)
        promptAnswer = promptAnswer.lower()
        promptAnswer = promptAnswer[:20] # only use first 20 characters. Otherwise an answer like: "Entailment, becasue ... there is no contradiction" would be misinterpreted
        if ENTAILMENT in promptAnswer and not CONTRADICTION in promptAnswer and not NEUTRAL in promptAnswer:
            return True
        else:
            return False
        

    def isEquivalent(self,question,text1,text2):
        # Checks if text1 and text2 mutually entail each other, indicating equivalent content
        if text1.strip() == text2.strip(): return True # same text 

        entailment1 = self.testEntailment(question,text1,text2)
        if not entailment1: return False

        entailment2 = self.testEntailment(question,text2,text1)#test the other direction
        if not entailment2: return False
        else: return True
    
    def clusterAnswerList(self,question,textList):
        # Groups together semantically equivalent answers into clusters
        clustering = []
        for newText in textList:
            foundCluster = False
            for cluster in clustering:
                if self.isEquivalent(question,cluster[0],newText):
                    foundCluster = True
                    cluster.append(newText)
                    break
            if not foundCluster: clustering.append([newText])#create new cluster
            clustering.sort(key=len,reverse=True)# largest classes first => higher matching probhability => less prompts needed (on average)
        return clustering
    
    def getClusterOfAnswer(self,clustering,question,answer):
        # Determines the cluster index for a given answer by checking equivalence
        for i,cluster in enumerate(clustering):
            if self.isEquivalent(question,cluster[0],answer): return i
        return -1
    


    def getAccuracyPrompt(self,question,trueAnswer,predictedAnswer):
        # Builds a prompt to assess correctness of predictedAnswer against trueAnswer
        prompt = f'We are assessing the quality of answers to the following question: {question}\n'
        prompt += f"The expected answer is: {trueAnswer}.\n"
        prompt += f"The proposed answer is: {predictedAnswer}\n"
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
        prompt += " Respond only with yes or no.\nResponse:"
        return prompt


    def isAnswerCorrect(self,question,trueAnswer,predictedAnswer):
        # Checks if GPT recognizes the predictedAnswer as correct compared to the trueAnswer
        prompt = self.getAccuracyPrompt(question,trueAnswer,predictedAnswer)
        promptAnswer = self.execPrompt(prompt)
        promptAnswer = promptAnswer.lower()
        #print(prompt)
        if "yes" in promptAnswer and not "no" in promptAnswer:
            #print("[YES]")
            return True
        else:
            #print("[NO]")
            return False











