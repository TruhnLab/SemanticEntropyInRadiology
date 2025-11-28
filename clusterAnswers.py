import sklearn.metrics
import CONFIG
from utilFunctions import readCSV,appendLineToCSV
from EntailmentCheck import EntailmentCheck
import sklearn
import promptLLM as promptLLM
from pathlib import Path
import math
from tqdm import tqdm
import random
random.seed(0,version=2)

ENTROPY_AT_LEAST_CUT = [i*0.1 for i in range(11)] #from 0 to 1 in 0.05 steps



def clusterQuestions(data): # Groups answered questions by their prompt to form clusters
    questionDict = dict() 
    for line in data:
        if line[2] not in questionDict: questionDict[line[2]] = list()
        questionDict[line[2]].append(line)

    questionList = list(questionDict.values())
    return questionList

def calcAUROC(clusterAnswerList):
    # Calculates the area under the ROC curve by comparing confidence of correct vs. incorrect answers
    compareListLess = list()
    confidenceList = [(-1*calcDiscreteSematicEntropy(x[0]),*x) for x in clusterAnswerList]# *-1 => lower entropy = higher confidence
    trueList = [x for x in confidenceList if x[4]]
    falseList = [x for x in confidenceList if not x[4]]
    # idea: generate matrix of each true and falsely answered question.
    # each combination is equally likely => average of matrix => AUROC
    for trueEl in trueList:
        for falseEl in falseList:
            compareListLess.append(1 if trueEl[0] > falseEl[0] else 0)

    aurocLess = sum(compareListLess)/len(compareListLess)
    return aurocLess

def calcRejectionAcc(clusterAnswerList,cutOffPercent):
    # Evaluates accuracy after rejecting a percentage of high-entropy answers
    entropyList = [(-1*calcDiscreteSematicEntropy(x[0]),*x) for x in clusterAnswerList]# *-1 => lower entropy = higher confidence
    entropyList.sort(key=lambda x: x[0],reverse=True)
    cutOffIdx = max(round(cutOffPercent*len(entropyList)),1)
    entropyList = entropyList[:cutOffIdx]
    trueList = [x for x in entropyList if x[4]]
    accuracy = len(trueList)/len(entropyList)
    return accuracy

def calcEntropyCutAccuracy(clusterAnswerList,cutOffEntropy,roundVal = 5):
    # Evaluates accuracy ignoring answers above a certain entropy value
    entropyList = [(calcDiscreteSematicEntropy(x[0]),*x) for x in clusterAnswerList]
    entropyList = [x for x in entropyList if round(x[0],roundVal) <= cutOffEntropy]
    totalCount = len(entropyList)
    if totalCount == 0: return None,0,0
    trueList = [x for x in entropyList if x[4]]
    trueCount = len(trueList)
    
    accuracy = trueCount/totalCount
    return accuracy, trueCount, totalCount


def calcElementwiseAURAC(clusterAnswerList):
    # Computes the AUC for the rejection accuracy curve
    valList = list()
    entropyList = [(-1*calcDiscreteSematicEntropy(x[0]),*x) for x in clusterAnswerList]# *-1 => lower entropy = higher confidence
    entropyList.sort(key=lambda x: x[0],reverse=True)
    for i in range(1,len(clusterAnswerList)+1):
        xPerc = i/len(entropyList)
        cutList = entropyList[:i]
        trueList = [x[4] for x in cutList if x[4]]
        acc = len(trueList)/len(cutList)
        valList.append((xPerc,acc))
    tup = tuple(zip(*valList))
    return sklearn.metrics.auc(*tup)

def calcAccuracy(clusterAnswerList):
    # Determines the fraction of correct answers
    return len([1 for x in clusterAnswerList if x[3]])/len(clusterAnswerList)

def calcDiscreteSematicEntropy(clustering):
    # Computes discrete semantic entropy based on clustering
    lenList = [len(x) for x in clustering]
    totalSum = sum(lenList)
    addendList = [(x/totalSum)*math.log(x/totalSum,10) for x in lenList]# Base 10, because they also do it in the original paper for some reason. Does not make a difference except for the total values.
    result = -1 * sum(addendList)
    return result

def execEvalOfQuestions(questionEvalFileList,promptFunc,noNewPrompts=False): # Main evaluation routine that loads results, clusters answers, and measures performance

    nameList = [x.split("/")[-1].replace("EVAL","").replace(".csv","").replace("_"," ") for x in questionEvalFileList]
    answerClusterFile = CONFIG.ANSWER_CLUSTER_FOLDER + "answerClustering"+"_".join(nameList) + ".csv"

    outCSV = list()

    evalQuestionsClusterList = list()
    for questionEvalFile in questionEvalFileList: # generates a list of lists containing the lines of the "answered questions" file
        head,*data = readCSV(questionEvalFile)
        evalQuestionsClusterList.extend(clusterQuestions(data))


    entailmentCheck = EntailmentCheck(promptFunc=promptFunc,noNewPrompts=noNewPrompts)
    clusterAnswerList = list()
    for evalQuestionCluster in evalQuestionsClusterList:
        #evalQuestionCluster is a list of answered questions. each answered question is a list with the following values: id,temperature,prompt,llmAnswer,question,answer,subtopic,reference,lastUpdate
        prompt = evalQuestionCluster[0][2]
        trueAnswer = evalQuestionCluster[0][5] # get answer of radiologist
        llmAnswerList = [x[3] for x in evalQuestionCluster if x[0] != "0"] # x[0] != "0" => filter out question with low sampling entropy (id == "0") (used for answering not for semantic entropy)
        predictedAnswer = [x[3] for x in evalQuestionCluster if x[0] == "0"][0] # answer with low entropy to calculate the accuracy later
        semanticClustering = entailmentCheck.clusterAnswerList(prompt,llmAnswerList) #Execute the clustering of the answers; => Give Prompt instead of question => more context / context closer to Question
        trueCluster = entailmentCheck.getClusterOfAnswer(semanticClustering,prompt,trueAnswer) #get the cluster of the true answer. / not needed for the evaluation, maybe still interesting :)
        answerCorrect = entailmentCheck.isAnswerCorrect(prompt,trueAnswer,predictedAnswer)
        clusterAnswer = (semanticClustering,trueCluster,predictedAnswer,answerCorrect) #its a bit ugly tu use a tuple
        outCSV.append((calcDiscreteSematicEntropy(semanticClustering),answerCorrect,predictedAnswer,semanticClustering,trueCluster,prompt))
        clusterAnswerList.append(clusterAnswer)
    outCSV.sort(key=lambda x: x[0])
    if not Path(answerClusterFile).exists():
        appendLineToCSV(answerClusterFile,("discreteSemanticEntropy","isAnswerCorrect","predictedAnswer","semanticClustering","trueCluster","usedPrompt"))
        for line in outCSV: appendLineToCSV(answerClusterFile,line)

    
    name = " +".join(nameList)
    print("")
    print("Evaluated questions/answers:",name)
    print("Number of questions:",len(evalQuestionsClusterList))
    print("accuracy",round(calcAccuracy(clusterAnswerList),3))
    print("AUROC:",round(calcAUROC(clusterAnswerList),3))
    print("AURAC:",round(calcElementwiseAURAC(clusterAnswerList),3))
    print("[RAC = rejection accuracy]")
    for i in range(1,11):
        print("RAC at",i*10,"%:",round(calcRejectionAcc(clusterAnswerList,i/10),3))
    print("The accuracy (ACC) for different entropy (SE) cutoff values. NUM describes the number of questions remaining after cutting off those with high entropy.")
    for i in ENTROPY_AT_LEAST_CUT:
        acc,trueCount,totalCount =  calcEntropyCutAccuracy(clusterAnswerList,i)
        acc = "n.a." if acc is None else round(acc,3)
        print("SE: ",(str(round(i,2))+"0000")[:3],"  ACC: ",(str(acc)+"0000")[:5],"  NUM: ",totalCount,sep="")
    print("")
    return clusterAnswerList



def applyBootstrapping(gpt4oClusterAnswerList: list,confP:float,numBootstraps: int = 1000000):
    # Performs bootstrap analysis to estimate confidence intervals for the accuracy gain
    accGainList = list()
    for i in tqdm(range(numBootstraps),leave=False):
        sample = random.choices(gpt4oClusterAnswerList,k=len(gpt4oClusterAnswerList))
        sampleAcc = calcAccuracy(sample)
        sampleCut05Acc,_,_ = calcEntropyCutAccuracy(sample,0.5)
        sampleACCgain = sampleCut05Acc-sampleAcc
        accGainList.append(sampleACCgain)
    accGainList.sort()
    posList = [x for x in accGainList if x > 0]
    pOneSided = 1 - len(posList)/len(accGainList)
    pTwoSided = 2 * min(pOneSided, 1 - pOneSided)
    print("[ one sided p-Val",round(pOneSided,6),"]", "[ two sided p-Val",round(pTwoSided,6),"]")
    cutNum = math.ceil(len(accGainList)*(confP/2))
    avgGain = sum(accGainList)/len(accGainList)
    confIntList = accGainList[:-cutNum][cutNum:]
    print("mean:",round(avgGain,3),", [",round(min(confIntList),3),";",round(max(confIntList),3),"] [intervall ",1 - confP,"%] num sample:",numBootstraps)

    return pTwoSided




