


Demo = False

if Demo:
    inputExample = [[[1,0,0],[1,1,0],[0,1,1],[0,0,0],[0,0,1],[1,0,0]],
                    [[1,0,0],[0,0,0],[1,1,0],[0,1,1],[0,0,0],[1,0,0],[0,1,1],[1,0,0]],
                    [[1,0,0],[0,0,0],[1,1,0],[0,1,1],[1,1,1],[0,0,0],[0,0,1],[1,1,0]]]
    inputExample = [["[1,0,0]","[1,1,0]","[0,1,1]","[0,0,0]","[0,0,1]","[1,0,0]"],
                    ["[1,0,0]","[0,0,0]","[1,1,0]","[0,1,1]","[0,0,0]","[1,0,0]","[0,1,1]","[1,0,0]"],
                    ["[1,0,0]","[0,0,0]","[1,1,0]","[0,1,1]","[1,1,1]","[0,0,0]","[0,0,1]","[1,1,0]"]]
else:
    with open(r"C:\Users\lucas\OneDrive\Área de Trabalho\output.txt", 'r') as file:
    #with open(r"C:\Users\lucas\OneDrive\Área de Trabalho\output_MissingSmallBlackWood.txt", 'r') as file:
        content = file.read()
        StartingPoint = "[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
        inputExample = content.replace("\n", "#").split(StartingPoint)[1:]
        inputExampleTemp = []
        for i in range(len(inputExample)):
            inputExample[i] = (StartingPoint+inputExample[i]+StartingPoint).split("#")
            if inputExample[i] not in inputExampleTemp:
                inputExampleTemp.append(inputExample[i])
        inputExample = inputExampleTemp


StateLabels = {}
StateLabels = {'[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x0', '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x1', '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]': 'x2', '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 'x3', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 'x4', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': 'x5', '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x6', '[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x7', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 'x8', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]': 'x9', '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x10', '[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'x11', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]': 'x12', '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': 'x13', '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]': 'x14'}
Transitions = {}
RunLabel = {} 
count = 0
runCount = 0

for run in inputExample:
    runCount+=1
    for stepNum, step in enumerate(run[:-1]):
        if step not in StateLabels.keys():
            StateLabels[step] = "x"+str(count)
            count = count+1
        if run[stepNum+1] not in StateLabels.keys():
            StateLabels[run[stepNum+1]] = "x"+str(count)
            count = count+1
            
        if StateLabels[step] not in Transitions.keys():
            Transitions[StateLabels[step]] = StateLabels[run[stepNum+1]]+","
            RunLabel[StateLabels[step]] = StateLabels[run[stepNum+1]] + ":" + str(runCount)
        else:
            if (StateLabels[run[stepNum+1]]+",") not in Transitions[StateLabels[step]]:
                Transitions[StateLabels[step]] = Transitions[StateLabels[step]] +StateLabels[run[stepNum+1]]+ ","
            RunLabel[StateLabels[step]] = RunLabel[StateLabels[step]] +","+ StateLabels[run[stepNum+1]] + ":" + str(runCount)
            
