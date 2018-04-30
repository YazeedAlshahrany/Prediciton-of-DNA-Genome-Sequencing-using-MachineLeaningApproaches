import csv
import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  Imputer

#********--------------------------------------------------------**********

header=[]
col_num=0
let_num=0
def initialize():
    global header
    global col_num
    global let_num
    header.clear()
    #This is an input value by the user for the num of column
    ## enter the number of colums form the user and store it in the variable col_num
    col_num = int(input("Enter the number of Colums:-"))
    # This is an input value by the user for the num of letter
    # enter the number of letter from the user and store in the variable let_num
    let_num = int(input("Enter the number of letters in a column :-"))
    #the file path on computer
    # set the input file path to the input varibale which will give is the location of the input file
    input_file_path = "D:/Users/yazeed/Desktop/TASK1/dna.txt"
    # open the input file to reac the data
    dna_file = open(input_file_path, "r")
    # give name for the output file where the result will be stored
    output_file_path = "D:/Users/yazeed/Desktop/TASK1/input.csv"
    # open the output file to make it ready to write the output in it.
    output = open(output_file_path, "w", newline="")
    # calculate total number of rows in the file based on the number of columns and number of letters selected by user
    # and store it in the row_lenght varibale
    row_length = (col_num * let_num) + 1
    # create a variable to write into a csv file and point the file handle to the variable
    writer = csv.writer(output)
    # create a list to create a header variable for the csv file[ column name of the file]
    header = []
    # add header names into the header list based on the number of columns entered by user
    for i in range(col_num):
        header.append("Attribute" + str(i + 1))
    # add the final output in the hearder
    header.append("output")
    # write to the header
    writer.writerow(header)
    for row in dna_file:
        table = 0
        row = row.strip()
        lenght = len(row.strip())
        if len(row) % row_length == 0:
            num_row = int(len(row) / row_length)
        else:
            num_row = len(row) // row_length
        index = 0
        for i in range(0, row_length * num_row, row_length):
            temp = []
            for column in range(col_num):
                temp.append(row[index:let_num + index])
                index += let_num
            temp.append(row[row_length + i - 1])
            index += 1
            writer.writerow(temp)
    dna_file.close()
    output.close()

#************--------------------------------------****************
#***** preprocessing of file converting into readable by the machine******
def preprocess():
    global header
    global col_num
    global let_num

    reader = open("D:/Users/yazeed/Desktop/TASK1/input.csv", "r")
    output = open("D:/Users/yazeed/Desktop/TASK1/preprocess.csv", "w", newline="")
    writer = csv.writer(output)
    reader.readline()
    writer.writerow(header)
    for line in reader:
        temp = []
        for word in line.split(','):
            # print(word)
            w = ''
            for letter in word.strip():
                w = w + str(ord(letter))
            temp.append(w)
        writer.writerow(temp)
    output.close()
    reader.close()

    #************--------------------------------------****************
    #************--------------------------------------****************
def DecisionTree():

    ###################### the classification based on decison tree classifier on preprocess file******
    input_file=pandas.read_csv("D:/Users/yazeed/Desktop/TASK1/preprocess.csv")# opening the file to apply the alogrithm
    #input_file.fillna(0)
    features_Dtree=list(input_file.columns[:col_num])# feature set that is you tell the program the attributes location
    print("* features:", features_Dtree, sep="\n")
    y_Dtree=input_file['output']
    X_Dtree=input_file[features_Dtree]
    X_train_Dtree, X_test_Dtree, y_train_Dtree, y_test_Dtree = train_test_split(X_Dtree, y_Dtree, random_state=1)# add test_size=whatever
    X_train_Dtree.fillna(0)
    y_train_Dtree.fillna(0)
    model_Dtree = DecisionTreeClassifier()
    X_train_Dtree=Imputer().fit_transform(X_train_Dtree)
    model_Dtree.fit(X_train_Dtree, y_train_Dtree)
    y_predict_Dtree = model_Dtree.predict(X_test_Dtree)
    print ("Accuracy of the model is {0}".format(accuracy_score(y_test_Dtree, y_predict_Dtree)))
    c=pandas.DataFrame(confusion_matrix(y_test_Dtree,y_predict_Dtree))
    print("confusion Matrix for the decision Tree")
    print(c)


#*********************-----------------------------------*************************
#************--------------------------------------****************
###################### the classification based on SupportVectorMachine classifier on preprocess file******
def SupportVectorMachine():

    input_file = pandas.read_csv("D:/Users/yazeed/Desktop/TASK1/preprocess.csv")  # opening the file to apply the alogrithm
    # input_file.fillna(0)
    features = list(input_file.columns[:col_num])  # feature set that is you tell the program the attributes location
    print("* features:", features, sep="\n")
    y = input_file['output']
    X = input_file[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # add test_size=whatever
    X_train.fillna(0)
    y_train.fillna(0)
    model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("Accuracy of the model is {0}".format(accuracy_score(y_test, y_predict)))
    c = pandas.DataFrame(confusion_matrix(y_test, y_predict))
    print("Confusion Matrix For SVM ")
    print(c)


#***********************************----------------------------------*******************
###################### the classification based on NeuralNetwowrk classifier on preprocess file******
def NeuralNetwowrk():
    input_file = pandas.read_csv("D:/Users/yazeed/Desktop/TASK1/preprocess.csv")  # opening the file to apply the alogrithm
    #### do the classification based on the neural network
    features_mlp = list(input_file.columns[:col_num])  # feature set that is you tell the program the attributes location
    print("* features:", features_mlp, sep="\n")
    y_mlp = input_file['output']
    X_mlp = input_file[features_mlp]
    X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(X_mlp, y_mlp)  # add test_size=whatever
    X_train_mlp.fillna(0)
    y_train_mlp.fillna(0)
    model_mlp = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    model_mlp.fit(X_train_mlp, y_train_mlp)
    print(model_mlp)
    y_predict_mlp = model_mlp.predict(X_test_mlp)
    print("Accuracy of the model is {0}".format(accuracy_score(y_test_mlp, y_predict_mlp)))
    c_mlp = pandas.DataFrame(confusion_matrix(y_test_mlp, y_predict_mlp))
    print(c_mlp)

#***********************************----------------------------------*******************
#***********************************----------------------------------*******************
### main program begins here

flag=True


#DecisionTree()
while(flag):
    try:
        choice=int(input("Please enter your choice to carry out machine classificaiont\n"
                     "1. Decicion Tree\n"
                     "2. Support Vector Machine\n"
                     "3. Multilayer perceptron (MLP)\n"
                     "4. Exit\n"
                     "Enter choice as 1,2,3 or 4 only\n"))
    except ValueError:
        print("please enter correct choice")
        continue

    if choice==1:
        initialize()
        DecisionTree()
    elif choice==2:
        initialize()
        SupportVectorMachine()
    elif choice==3:
        initialize()
        NeuralNetwowrk()
    elif choice==4:
        flag=False
    else:
        print("please print choice 1,2,3 or 4 only")


