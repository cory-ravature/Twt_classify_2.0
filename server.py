# Import required dependencies
from flask import Flask,render_template,url_for,request
import pickle

from sklearn.feature_extraction.text import CountVectorizer
import utils
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
from support import fmt_input_tweet, classify_new_tweet, determine_airline, determine_feedback



# Load model, and vectororizer
vect = pickle.load(open('vectorizer.plk','rb'))
#model = pickle.load(open('Tweet_Classifier.plk','rb'))
model = utils.load_model("<Tweet_Classifier.plk>")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form['message']
    targetAirline = determine_airline(message)
    tweet_feedback = determine_feedback(message)
    answer = classify_new_tweet(message,model,vect)
		
    return render_template('results.html',prediction = answer, targ_airline = targetAirline, feedback = tweet_feedback)

app.run(debug=True)



