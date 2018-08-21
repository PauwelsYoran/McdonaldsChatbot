import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

#this block sets up and loads the sentence model
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

    # load our saved model
    model.load('./model.tflearn')

#this block sets and loads the amount model
data = pickle.load( open( "training_data_count", "rb" ) )
words2 = data['words']
classes2 = data['classes']
train_x2 = data['train_x']
train_y2 = data['train_y']

tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x2[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y2[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model_count = tflearn.DNN(net, tensorboard_dir='tflearn_logs')


import json
with open('count.json') as json_data:
    count = json.load(json_data)
    model_count.load('./model_count.tflearn')




def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))




#classify count

ERROR_THRESHOLD_COUNT = 0.6

#classifies the sentence for an amount for example -> 1  Big mac
def classify_count(sentence):
    # generate probabilities from the model
    results = model_count.predict([bow(sentence, words2)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD_COUNT]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes2[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

#returns the answer linked with the classiefier
def response_count(sentence, userID='123', show_details=False):
    results = classify_count(sentence)
    res = None
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in count['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:

                    return  int(i['count'])

            results.pop(0)





#sets the context
context = {"123":"none"}




ERROR_THRESHOLD = 0.01

#classifies a sentence for the menu item
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

#the total order
order ={}
#if a menu is chosen it keeps the size to link it to a drink
drinkSizeMenu =""
#saves how much of a nitemn is needed
amount = None

def response(sentence, userID='123', show_details=False):
    global order
    global drinkSizeMenu
    global context
    global amount
    results = classify(sentence)
    res = None

    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:

                    # check if this intent is contextual and applies to this user's conversation)

                    #sets the amount of the item if their is one
                    if response_count(sentence) != None and context[userID] =="none":
                        amount = response_count(sentence)

                    #checks if a context is related to an item
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):

                        if show_details: print('tag:', i['tag'])
                        # a random response from the intent
                        res = (random.choice(i['responses']))


                        if 'price' in i:

                            #checks if an item is already orderd so the amount can be raised or the item added
                            if drinkSizeMenu +i['name'] not in order:
                                # the drinkSizeMenu value's are used specify if the drink is a large or a medium
                               order[drinkSizeMenu +i['name']] = {'price':i['price'],'count':amount}
                            else:

                                order[drinkSizeMenu +i['name']]['count'] = int(order[drinkSizeMenu +i['name']]['count']) + amount



                            #checks for the menu filter  to set the driksize in a menu
                            if 'filter' in i and i ['filter'] == 'menu':

                                drinkSizeMenu = str(i['name']).split(" ")[0] + " "
                            else:
                                drinkSizeMenu = ""


                        #looks if the order is over
                        if i['tag'] == "done":
                            total = 0

                            #calculate the totoal
                            for key,value in order.items():

                                total = total + (float(value['price'])*int(value['count']))
                            res = res +" " +str(total)

                            #prints the ticket
                            for key, value in order.items():
                                print(str(value['count'])+ " " + str(key) + " " + str((float(value['price'])*int(value['count']))))
                            order = {}

                        #changes the context if nececarry
                        if 'context_set' in i:
                            print(i['context_set'])
                            if show_details: print('context:', i['context_set'])
                            context[userID] = i['context_set']

                    if res != None:
                        return print(res)

            results.pop(0)

while True:

    inp = input("question: ")
    response(inp)


#; menu aanvullen;
