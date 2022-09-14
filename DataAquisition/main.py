import pandas as pd
import numpy as np
import json
import csv

def export_json(filename, jsonObject):
    JsonFile = open(filename, "w")
    JsonFile.write(jsonObject)
    JsonFile.close()
    

def main():
    # Getting the data from csv into a dataframe (tab separated values, with "latin-1" encoding)
    d1 = pd.read_csv('mnt/data/dagstuhl-15512-argquality-corpus-annotated.csv', encoding="latin-1", delimiter="\t")
    
    # Keeping only the headers which are useful for analysis and rearranging and renaming them
    listofHeaders = {"#id", "issue", "stance", "argumentative", "overall quality", "effectiveness", "argument"}
    d2 = d1[listofHeaders]
    d2 = d2[['#id', 'issue', 'stance', 'argumentative', 'overall quality', 'effectiveness', 'argument']]
    d2.rename(columns = {'#id':'id', 'stance':'stance_on_topic', 'overall quality':'argument_quality_scores', 'effectiveness':'effectiveness_scores', 'argument':'text'}, inplace = True)

    # cleaning the columns to keep just numeric values
    d2['argument_quality_scores'] = d2['argument_quality_scores'].str[:1]
    d2['effectiveness_scores'] = d2['effectiveness_scores'].str[:1]
    d2 = d2.dropna(subset=['argument_quality_scores'])
    d2 = d2.dropna(subset=['effectiveness_scores'])
    d2['argument_quality_scores'] = d2['argument_quality_scores'].astype(float)
    d2['effectiveness_scores'] = d2['effectiveness_scores'].astype(float)
    
    # grouping the data by the data based on Argument ID (#id) and then resetting the index
    d3 = d2.groupby('id').agg(lambda x: x.tolist())
    d3 = d3.reset_index()

    # Removing non-argumentative rows, where majority annonators put 'n' in argumentative )
    for index, row in d3.iterrows():
        if row['argumentative'].count('n') > 1:
            d3.drop(index, inplace=True)
    
    # After grouping by ID, removing redundant data from "issue", "argument" and "stance" 
    for index, row in d3.iterrows():
        row['issue'] = row['issue'][0]
        row['text']= row['text'][0]
        row['stance_on_topic'] = row['stance_on_topic'][0]
    
    # splitting data into 70:30:10 into train, test and validate sets
    train = d3.sample(frac=0.7,random_state=200) #random state is a seed value
    test = d3.drop(train.index)
    validate = test.sample(frac=0.33,random_state=200)
    test = test.drop(validate.index)

    # Creating JSON objects from the splits
    train_json = train.to_json(orient="records")
    test_json = test.to_json(orient="records")
    validate_json = validate.to_json(orient="records")

    # Exporting them to JSON Files
    export_json("train.json", train_json)
    export_json("test.json", test_json)
    export_json("validate.json", validate_json)
    

    pass


if __name__ == '__main__':
    main()
