import json


def alignTable(instanceData, width):
    tabLength = width - len(str(instanceData))
    tab = ""
    for i in range(tabLength):
        tab += " "
    return tab


def classify(distanceList, k):
    nearestK_Neighbours = distanceList[:k]
    yes = 0
    no = 0
    for neighbour in nearestK_Neighbours:
        if neighbour[1] == 0:
            no += 1
        elif neighbour[1] == 1:
            yes += 1

    if yes > no:
        return "Yes"
    return "No"


def distanceKNN(newInstance, dataEncoded):
    distance = 0
    distanceList = []
    for trainingInstance in dataEncoded:
        for key in newInstance:
            distance += abs(trainingInstance[key + '_' + newInstance[key]] - 1)
        distanceList.append([distance, trainingInstance['PlayTennis']])
        distance = 0
    distanceList.sort()
    return distanceList


def evaluate(prediction):  # instance[0] = PREDICTED, instance[1] = ACTUAL
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for instance in prediction:
        if instance[0] == "Yes" and instance[1] == "Yes":
            truePositive += 1
        elif instance[0] == "Yes" and instance[1] == "No":
            falsePositive += 1
        elif instance[0] == "No" and instance[1] == "Yes":
            falseNegative += 1
        elif instance[0] == "No" and instance[1] == "No":
            trueNegative += 1
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)

    print(f"Accuracy: {accuracy*100:.2f}\n")
    print("CONFUSION MATRIX")
    print("                    Positive          Negative")
    print("True               ", truePositive, alignTable(truePositive, 16), trueNegative)
    print("False              ", falsePositive, alignTable(falsePositive, 16), falseNegative)
    print("--------------------------------------------------------------------------------")



def test(data, dataEncoded):
    prediction3K = []
    prediction5K = []
    for trainingInstance in data:
        instance = trainingInstance.copy()
        instance.pop('PlayTennis', None)
        instance.pop('Day', None)
        distanceList = distanceKNN(instance, dataEncoded)
        result3K = classify(distanceList, 3)
        result5K = classify(distanceList, 5)
        prediction3K.append([result3K, trainingInstance['PlayTennis']])
        prediction5K.append([result5K, trainingInstance['PlayTennis']])

    print("Evaluation for K = 3\n")
    evaluate(prediction3K)

    print("\nEvaluation for K = 5\n")
    evaluate(prediction5K)


def getClearedCurrentInstance(data):
    currentInstance = {}
    attributes = []
    for attribute in data[0]:
        if attribute != "Day" and attribute != "PlayTennis":
            attributes.append(attribute)

    for instance in data:
        for attribute in attributes:
            currentInstance[attribute + '_' + instance[attribute]] = 0
    currentInstance['PlayTennis'] = 0
    return currentInstance


def oneHotEncoding(data):
    dataEncoded = []
    attributes = []

    for attribute in data[0]:
        if attribute != "Day" and attribute != "PlayTennis":
            attributes.append(attribute)

    clearedInstance = getClearedCurrentInstance(data)

    for dataInstance in data:
        currentInstance = clearedInstance.copy()
        for attribute in attributes:
            currentInstance[attribute + '_' + dataInstance[attribute]] = 1
        if dataInstance['PlayTennis'] == 'Yes':
            currentInstance['PlayTennis'] = 1
        dataEncoded.append(currentInstance)
    return dataEncoded


def printData(data):  # Print the data in tabular format
    print("Day  Outlook        Temperature    Humidity       Wind           PlayTennis")
    for instance in data:
        print(instance['Day'], alignTable(instance['Day'], 3),
              instance['Outlook'], alignTable(instance['Outlook'], 13),
              instance['Temperature'], alignTable(instance['Temperature'], 13),
              instance['Humidity'], alignTable(instance['Humidity'], 13),
              instance['Wind'], alignTable(instance['Wind'], 13),
              instance['PlayTennis'], alignTable(instance['PlayTennis'], 13))


def prepareData():
    with open('PlayTennisData.json', 'r') as json_file:
        data = json.load(json_file)
    printData(data)
    yes = 0
    no = 0
    for instance in data:
        if instance['PlayTennis'] == 'Yes':
            yes += 1
        elif instance['PlayTennis'] == 'No':
            no += 1
    print("\nPlayTennis = Yes count:", yes)
    print("PlayTennis = No  count:", no)
    print("--------------------------------------------------------------------------------")
    return data


def main():
    data = prepareData()

    dataEncoded = oneHotEncoding(data)

    test(data, dataEncoded)

    k = input("Enter the K parameter: ")
    distanceMetric = input("\nEnter 1 for Euclidean Distance\nEnter 2 for Manhattan Distance"
                           "\nChoose a distance metric: ")

    newInstance = {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"}

    if distanceMetric == '1' or distanceMetric == '2':
        distanceList = distanceKNN(newInstance, dataEncoded)
    else:
        print("Enter a valid distance metric.")
        exit()

    result = classify(distanceList, int(k))
    print("\nResult of the given instance", newInstance, "is", result)


if __name__ == "__main__":
    main()
