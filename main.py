import json
import math
import logging


def loggingFormat():  # Logging format
    logging.basicConfig(
        filename='predictions.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )


def alignTable(instanceData, width):  # Table column alignment
    tabLength = width - len(str(instanceData))
    tab = ""
    for i in range(tabLength):
        tab += " "
    return tab


def classify(distanceList, k):  # Classify the instance by the nearest k neighbours
    nearestK_Neighbours = distanceList[:k]
    yes = 0
    no = 0
    for neighbour in nearestK_Neighbours:  # Calculate the total number of "Yes" and "No" among neighbours
        if neighbour[1] == 0:
            no += 1
        elif neighbour[1] == 1:
            yes += 1

    if yes > no:  # Classify based on whether most of the k nearest neighbours are "Yes" or "No"
        return "Yes"
    return "No"


def distanceKNN(newInstance, dataEncoded, distanceMetric):  # Calculate the distance between...
    distance = 0
    distanceList = []
    for trainingInstance in dataEncoded:  # ...each training set instance and...
        for key in newInstance:  # ...given test instance
            # 0^2 = 0, 1^2 = 1 -> Therefore, no need to take the square. Only take the root if the distance metric
            # is chosen as Euclidean
            distance += math.pow(abs(trainingInstance[key + '_' + newInstance[key]] - 1), (1 / distanceMetric))
        distanceList.append([distance, trainingInstance['PlayTennis']])
        distance = 0
    distanceList.sort(key=lambda x: (x[0], x[1] != 'Yes'))
    return distanceList


def evaluate(prediction):  # instance[0] = PREDICTED, instance[1] = ACTUAL
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for instance in prediction:
        if instance[0] == "Yes" and instance[1] == "Yes":  # Calculate the True Positive classifications
            truePositive += 1
        elif instance[0] == "Yes" and instance[1] == "No":  # Calculate the False Positive classifications
            falsePositive += 1
        elif instance[0] == "No" and instance[1] == "Yes":  # Calculate the False Negative classifications
            falseNegative += 1
        elif instance[0] == "No" and instance[1] == "No":  # Calculate the True Negative classifications
            trueNegative += 1

    # Calculate the accuracy of the algorithm with the given training set and test set
    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)

    print(f"Accuracy: {accuracy * 100:.2f}\n")  # Print the evaluation
    print("CONFUSION MATRIX")
    print("                    Positive          Negative")
    print("True               ", truePositive, alignTable(truePositive, 16), trueNegative)
    print("False              ", falsePositive, alignTable(falsePositive, 16), falseNegative)
    print("--------------------------------------------------------------------------------")


def test(data, dataEncoded):  # Test the training set instances due to its small size
    prediction3K = []
    prediction5K = []
    for trainingInstance in data:
        instance = trainingInstance.copy()
        instance.pop('PlayTennis', None)  # Exclude the PlayTennis column for abstraction
        instance.pop('Day', None)  # Exclude the Day column, because unnecessary
        distanceList = distanceKNN(instance, dataEncoded, 1)
        result3K = classify(distanceList, 3)  # Test with k = 3
        result5K = classify(distanceList, 5)  # Test with k = 5
        prediction3K.append([result3K, trainingInstance['PlayTennis']])  # Save the predictions for all instances
        prediction5K.append([result5K, trainingInstance['PlayTennis']])  # Save the predictions for all instances
        # Log the results
        logging.info(f"Instance: {instance}, Predicted Outcome (k=3): {result3K}, "
                     f"Actual Outcome: {trainingInstance['PlayTennis']}")
        logging.info(f"Instance: {instance}, Predicted Outcome (k=5): {result5K}, "
                     f"Actual Outcome: {trainingInstance['PlayTennis']}")

    print("Evaluation for K = 3\n")
    evaluate(prediction3K)  # Send the predictions for evaluation of the algorithm

    print("\nEvaluation for K = 5\n")
    evaluate(prediction5K)  # Send the predictions for evaluation of the algorithm


def getClearedCurrentInstance(data):  # Get an instance where all the attribute values are 0, used for later
    currentInstance = {}
    attributes = []
    for attribute in data[0]:
        if attribute != "Day" and attribute != "PlayTennis":  # Exclude day and PlayTennis
            attributes.append(attribute)

    for instance in data:  # Create the dictionary with 0 for all attribute values, will be filled later
        for attribute in attributes:
            currentInstance[attribute + '_' + instance[attribute]] = 0
    currentInstance['PlayTennis'] = 0
    return currentInstance


def oneHotEncoding(data):  # Encode the categorical data, so we can have numerical data
    dataEncoded = []
    attributes = []

    for attribute in data[0]:  # Retrieve the attributes
        if attribute != "Day" and attribute != "PlayTennis":
            attributes.append(attribute)

    clearedInstance = getClearedCurrentInstance(data)

    for dataInstance in data:
        currentInstance = clearedInstance.copy()  # Retrieve a clear instances with 0 for all attributes values
        for attribute in attributes:  # For each unique attribute value of the instance, encode with 0 and 1
            currentInstance[attribute + '_' + dataInstance[attribute]] = 1
        if dataInstance['PlayTennis'] == 'Yes':  # PlayTennis == Yes is 1, == No is 0
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


def prepareData():  # Load the data and print the summary of the training set
    with open('PlayTennisData.json', 'r') as json_file:
        data = json.load(json_file)
    printData(data)
    yes = 0
    no = 0
    for instance in data:  # Calculate total number of Yes and No
        if instance['PlayTennis'] == 'Yes':
            yes += 1
        elif instance['PlayTennis'] == 'No':
            no += 1
    print("\nPlayTennis = Yes count:", yes)
    print("PlayTennis = No  count:", no)
    print("--------------------------------------------------------------------------------")
    return data


def main():
    loggingFormat()
    data = prepareData()
    dataEncoded = oneHotEncoding(data)
    test(data, dataEncoded)

    k = input("Enter the K parameter: ")
    distanceMetric = input("\nEnter 1 for Manhattan Distance\nEnter 2 for Euclidean Distance"
                           "\nChoose a distance metric: ")

    # If you want to classify new instance, adjust the values of the dictionary below
    newInstance = {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"}

    if distanceMetric == '1' or distanceMetric == '2':  # Check valid distance metric
        distanceList = distanceKNN(newInstance, dataEncoded, int(distanceMetric))
    else:
        print("Enter a valid distance metric.")
        exit()

    result = classify(distanceList, int(k))
    print("\nResult of the given instance", newInstance, "is", result)

    logging.info(f"Instance: {newInstance}, Predicted Outcome (k=3): {result}")  # Log the result of the new instance


if __name__ == "__main__":
    main()
