import os
from flask import Flask, request, jsonify
from functools import reduce
import pickle
app = Flask(__name__)


@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 3 attribute values from the query string
    # use the request.args dictionary
    fans = request.args.get("fans", "")
    cp = request.args.get("cp", "")
    friends = request.args.get("friends", "")
    # get a prediction for this unseen instance via naive bayes
    # return the prediction as a JSON Response

    prediction = predict_yelp_well([float(fans), float(cp), float(friends)])
    # if anything goes wrong, predict_yelp_well() is going to return None
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else:
        # Failed
        return "Error making prediction", 400

def predict_yelp_well(instance):
    infile = open("nb.p", "rb")
    priors, posteriors = pickle.load(infile)
    try:
        probs = compute_probs(instance, priors, posteriors)
        return predict_from(probs)
    except:
        return None

def multiply(a, b):
    """
    Function to multiply two values
    params:
            a - one operand
            b - the second operand
    return:
            the multiplication value of a*b
    """
    return a*b

def compute_probs(test, priors, posteriors):
    """
    Function to compute the probabilities of a test set
    params:
            test - the test set
            priors - the priors dictionary
            posteriors - the posteriors dictionary
    return:
            a return dictionary with the probalities for the test instances
    """
    return_dictionary = {}

    # Loop through the priors
    for k, v in priors.items():
        prior = v
        dictionary = posteriors[k] # Get the posteriors dictionary
        probs = []
        probs.append(prior) # Append the prior probability
        # Loop through the test
        for i in range(len(test)):
            if test[i] in dictionary[i]:  
                # Append the probability value
                probs.append(dictionary[i][test[i]])
            else:
                # Not in the dictionary, append a probability of 0
                probs.append(0)

        # Reduce the list by multiplying all values
        probability = reduce(multiply, probs)
        return_dictionary[k] = probability # Set the dictionary
    
    return return_dictionary

def predict_from(probs_dictionary):
    """
    Function to make a prediction from a dictionary
    params:
            probs_dictionary: the dictionary holding the probabilities
    return:
            the prediction label
    """
    # Init
    max = 0 
    prediction = ""

    # Loop through probabilities and check maxes
    for k, v, in probs_dictionary.items():
        if v >= max:
            prediction = k
            max = v
    
    # Return the prediction with the highest probability
    return prediction

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access 
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)
