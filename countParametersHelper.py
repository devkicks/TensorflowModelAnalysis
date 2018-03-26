"""
Created on Mon Mar 12 18:00:27 2018

Helper libraries for analysing a tensorflow network

- The functions here require end_points as well as a tensorflow graph

@author: muhammad.asad
"""

import tensorflow as tf

def convertTensorNameAndShapeToText(inTensor):
    outStr = inTensor.name+','
    
    # get the tensor shape
    tShape = inTensor.shape
    
    # iterate through the shape and add them to the outStr
    for i in range(len(tShape)):
        cVal = tShape[i].value
        
        # check for None and replace with -1
        if cVal is None:
            cVal = -1
        
        #print(cVal)
        outStr += str(cVal)
        
        # check if not the last element - if not then add a ,
        if(i != len(tShape)-1):
            outStr += ','
    print(outStr)
    return outStr

def convertEndpointNameAndShapeToText(inName, inShape):
    outStr = inName+','
       
    # iterate through the shape and add them to the outStr
    for i in range(len(inShape)):
        cVal = inShape[i]
        
        # check for None and replace with -1
        if cVal is None:
            cVal = -1
        
        #print(cVal)
        outStr += str(cVal)
        
        # check if not the last element - if not then add a ,
        if(i != len(inShape)-1):
            outStr += ','
    print(outStr)
    return outStr

# Help regarding tf.GraphKeys
# https://www.tensorflow.org/api_docs/python/tf/GraphKeys
    
# save list of trainable parameters to a CSV file
def createCSVTrainParamsFromGraph(networkName):    
    networkParamsList = list()
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES ):
        tempStr = convertTensorNameAndShapeToText(variable)
        networkParamsList.append(tempStr)
    
    fileName = networkName + '_Trainable' + '.csv'
    with open(fileName, 'w') as f:
        f.write("Model: " + networkName)
        for item in networkParamsList:
            f.write("%s\n" % item)           
            

# save list of model parameters to a CSV file      
def createCSVModelParamsFromEndpoints(networkName, end_points):    
    networkParamsList = list()
    for variable in end_points:
        tempStr = convertEndpointNameAndShapeToText(variable, end_points[variable].shape)
        networkParamsList.append(tempStr)
#        print(variable)
    
    fileName = networkName + '_Model' + '.csv'
    with open(fileName, 'w') as f:
        f.write("Model: " + networkName)
        for item in networkParamsList:
            f.write("%s\n" % item)            



