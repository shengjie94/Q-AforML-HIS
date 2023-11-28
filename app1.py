from flask import Flask, render_template, request, jsonify, make_response
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import speech_recognition as sr
import uuid
import time
import threading
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import json
import os
import pandas as pd

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
pronouns = ["it", "they", "he", "she", "this", "these", "those", "him", "her", "them"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("C:/Users/shengjie zhao/Desktop/UI1/model/GPT2_model4")
    print("Tokenizer loaded.")

    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained("C:/Users/shengjie zhao/Desktop/UI1/model/GPT2_model4").to(device)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading the model: {e}")

app = Flask(__name__)

sessions = {}

def extract_keywords(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in tokens]
    filtered_tokens = [i for i in lemmatized_tokens if i not in stop_words]
    return ' '.join(filtered_tokens)
class QA_Model:
    def __init__(self, data_path=None, tokenizer=None, model=None, device=None):
        if data_path:
            self.dataset = pd.read_json(data_path, encoding='utf-8')
            self.dataset_keywords = [extract_keywords(row['Question']) for _, row in self.dataset.iterrows()]
            self.vectorizer = TfidfVectorizer()
            self.dataset_vectors = self.vectorizer.fit_transform(self.dataset_keywords)
        else:
            self.dataset = None
        
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
    def is_new_context(self, current_input, previous_input):
        contains_pronoun = any(pronoun in current_input.lower() for pronoun in pronouns)
        current_keywords = extract_keywords(current_input)
        previous_keywords = extract_keywords(previous_input)
        vectorized_current = self.vectorizer.transform([current_keywords])
        vectorized_previous = self.vectorizer.transform([previous_keywords])
        similarity = cosine_similarity(vectorized_current, vectorized_previous)[0][0]
        
        # print(f"Current Input: {current_input}")
        # print(f"Previous Input: {previous_input}")
        # print(f"Contains pronoun: {contains_pronoun}")
        # print(f"Similarity: {similarity}")
        
        if contains_pronoun and similarity > 0.5:
            return False 
        elif not contains_pronoun and similarity < 0.2:
            return True  
        else:
            return similarity < 0.5
    def find_max_similar(self, user_input, session_id=None):
        if session_id and sessions.get(session_id):
            history = sessions[session_id]['history']
            if self.is_new_context(user_input, history[-1]):
                combined_input = user_input
            else:
                combined_input = " | ".join(history) + " " + user_input   
        else:
            combined_input = user_input
        
        print(f"Processed input: {combined_input}")

        combined_keywords = extract_keywords(combined_input)
        all_keywords = self.dataset_keywords + [combined_keywords]
        vectorizer = TfidfVectorizer()
        all_vectors = vectorizer.fit_transform(all_keywords)

        vectorized_input = all_vectors[-1]
        dataset_vectors = all_vectors[:-1]
        similarities = cosine_similarity(vectorized_input, dataset_vectors)  
        max_similarity = max(similarities[0])
        similar_index = similarities[0].argmax()
        similarity_threshold = 0.5
        print(similarities)
        print(max_similarity)
        is_code_response = "python" in combined_input.lower() or "code" in combined_input.lower()
        if max_similarity < similarity_threshold:
            return ("Sorry, we cannot provide a definitive answer to this question. However, if you could provide<br>more details or further information about the issue, we'll do our best to assist you.", False)
        else:
            response = self.dataset.iloc[similar_index]['Answer']
            return (response, is_code_response)
    def generate_response(self, user_input):
        try:
            if user_input:
                user_input = user_input[0].upper() + user_input[1:]
            print("Generating response...")
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
            attention_mask = torch.tensor([1] * len(input_ids[0]), dtype=torch.long).unsqueeze(0).to(device)
            model.eval()
            
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=230,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            first_period_index = response.find('. ')
            if first_period_index != -1:
                response = response[first_period_index+2:]

            formatted_response = response.replace(". ", ".\n")

            print(f"Generated response: {formatted_response}")
            return formatted_response
        except Exception as e:
            print(f"Error generating the response: {e}")
            return "I'm sorry, I can't process that right now."
        
    def answer_question(self, user_input, session_id=None):
        qa_response, is_code_response = None, False
        gpt2_response = None
        
        if self.dataset is not None:
            qa_response, is_code_response = self.find_max_similar(user_input, session_id)
        
        gpt2_response = self.generate_response(user_input)
        
        return {
            'qa_response': qa_response,
            'gpt2_response': gpt2_response,
            'is_code_response': is_code_response
        }


bot = QA_Model('C:/Users/shengjie zhao/Desktop/UI1/json/t.json')

@app.route('/get_voice', methods=['POST'])
def get_voice():
    session_id = request.cookies.get('session_id')
    print('session_id',session_id)
    if not session_id:
        session_id = str(uuid.uuid4())

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("please speak...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            response, is_code = bot.find_max_similar(text)
            response = response.replace("\n", "<br>")
        except sr.UnknownValueError:
            text = "Sorry, can't understand the words"
            response = text
        except sr.RequestError as e:
            text = f"Sorry, can't request Google Web Speech API；{e}"
            response = text

    if session_id not in sessions:
        sessions[session_id] = {'history': []}
    sessions[session_id]['history'].append(text)
    sessions[session_id]['last_interaction'] = time.time()  # Update the timestamp for session cleanup

    if text == "Sorry, can't understand the words" or ("Sorry, can't request Google Web Speech API" in text):
        flask_response = jsonify({'botresponse': response})
    else:
        response_content = {'userinput': text, 'botresponse': response, 'is_code': is_code}
        flask_response = make_response(jsonify(response_content))
        flask_response.set_cookie('session_id', session_id)  # Send the session_id back to the client

    return flask_response
def clear_old_sessions():
    current_time = time.time()
    timeout = 3600  # 1 hour
    for session_id, session_data in list(sessions.items()):
        if current_time - session_data['last_interaction'] > timeout:
            del sessions[session_id]
    # Call the function again after your desired timeout
    threading.Timer(timeout, clear_old_sessions).start()    
    

@app.route('/text_to_voice', methods=['POST'])
def text_to_voice_route():
    text = request.form.get('text')
    tts = gTTS(text)
    tts.save("static/output.mp3")
    return jsonify({'status': 'done'})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('userinput')
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())

        responses = bot.answer_question(user_input, session_id)
        qa_response = responses['qa_response'].replace("\n", "<br>") if responses['qa_response'] else "No matching entry found in QA database."
        gpt2_response = responses['gpt2_response'].replace("\n", "<br>")

        response_content = {
            'qa_response': qa_response,
            'gpt2_response': gpt2_response,
            'is_code': responses['is_code_response']
        }

        flask_response = make_response(jsonify(response_content))
        flask_response.set_cookie('session_id', session_id)
        return flask_response

    return render_template('index.html')


if __name__ == "__main__":
    clear_old_sessions()
    app.run(debug=True)


