import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Preprocessing function: new_twt is user input, model/cv are created/available above
def classify_new_tweet(new_twt, model,cv):  
    clf = model
    vect = cv

    fmt_twt = fmt_input_tweet(new_twt)
    fmt_twt_dtm = vect.transform([fmt_twt])[0]
    pred = clf.predict(fmt_twt_dtm.toarray())

    def mood(x):
        return {
            0: 'negative',
            1: 'positive',
            2: 'neutral'
        }[x]

    return mood(pred[0])
#
def fmt_input_tweet(txt):
    
    # Remove @tweets, numbers, hyperlinks that do not start with letters
    txt = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([0-9])"," ",txt)
    #print(txt)
    
    # tokenize into words
    tokens = [word for word in nltk.word_tokenize(txt)]
    #print(tokens)

    # only keep tokens that start with a letter (using regular expressions)
    clean_tokens = [token for token in tokens if re.search(r'^[a-zA-Z]+', token)]
    #print('clean_tokens:\n',clean_tokens)

    # stem the tokens
    stemmer = SnowballStemmer('english')
    stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]
    #print('stemmed_tokens:\n',stemmed_tokens)

    #Lemmatizing
    lemmatizer = nltk.WordNetLemmatizer()
    lem_tokens = [lemmatizer.lemmatize(t) for t in stemmed_tokens]
    #print('lemmatizer : \n',lem_tokens)
    
    #Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')

    # stem the stopwords
    stemmed_stops = [stemmer.stem(t) for t in stopwords]

    # remove stopwords from stemmed/lemmatized tokens
    lem_tokens_no_stop = [stemmer.stem(t) for t in lem_tokens if t not in stemmed_stops]

    # remove words whose length is <3
    clean_lem_tok = [e for e in lem_tokens_no_stop if len(e) >= 3]
    #print('clean_lem_tok: ',clean_lem_tok)
    
    # Detokenize new tweet for vector processing
    new_formatted_tweet=" ".join(clean_lem_tok)
    #print('new_formatted_tweet: ',new_formatted_tweet)
    
    return new_formatted_tweet
# Preprocessing Functions end

def determine_airline(txt):
    listAirlines = ['virginamerica','americanair','united','southwestair','jetblue','usairways']
    lowered_input = txt.lower()
    target_airline = ''

    for i in range(len(listAirlines)):
        if(listAirlines[i] in lowered_input):
            target_airline = listAirlines[i]
    if (target_airline == ''):
        target_airline = 'NA'
    return target_airline

def determine_feedback(txt):
    
    lowered_input = txt.lower()
    listFeedback =	{
  "wait": "waiting too long",
  "hour": "waiting too long",
  "delay": "waiting too long",
  "cancel": "flight was canceled",
  "customer service": "customer service",
  "expensive": "pricing",
  "cost": "pricing"
    }
    feedback = ''

    for i in listFeedback:
        if(i in lowered_input):
            feedback = listFeedback[i]
    if (feedback == ''):
        feedback = 'NA'
    return feedback
