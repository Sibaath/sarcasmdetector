import google.generativeai as genai
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json



genai.configure(api_key='AIzaSyDsI62mSwNz-HvXQGX-QNcak_dEtornDV4')

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)


chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "i am developing an sarcasm detector that uses takes an input text and analyze it for sarcasm if the sarcasm is present in the input sentence then it should give the json response as 0 or 1 for the presence or absence of sarcasm....the response should be a json...this should be accurate and precise",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```python\nimport nltk\nfrom nltk.sentiment import SentimentIntensityAnalyzer\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\n\nnltk.download('vader_lexicon')\nnltk.download('stopwords')\nnltk.download('punkt')\n\ndef detect_sarcasm(text):\n    \"\"\"\n    Detects sarcasm in a given text using sentiment analysis and linguistic features.\n\n    Args:\n        text (str): The input text to analyze.\n\n    Returns:\n        dict: A JSON response containing the sarcasm detection result (0 or 1).\n    \"\"\"\n\n    # Tokenize the text\n    tokens = word_tokenize(text)\n\n    # Remove stop words\n    stop_words = set(stopwords.words('english'))\n    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]\n\n    # Calculate sentiment score using VADER\n    sia = SentimentIntensityAnalyzer()\n    sentiment = sia.polarity_scores(text)\n\n    # Define features for sarcasm detection\n    features = {\n        'negative_sentiment': sentiment['neg'],\n        'positive_sentiment': sentiment['pos'],\n        'compound_sentiment': sentiment['compound'],\n        'exclamation_count': text.count('!'),\n        'question_count': text.count('?'),\n        'uppercase_count': sum(1 for char in text if char.isupper()),\n        'token_count': len(filtered_tokens),\n    }\n\n    # Use a simple rule-based approach based on sentiment and features\n    if features['compound_sentiment'] < 0 and (\n        features['exclamation_count'] > 0 or\n        features['question_count'] > 0 or\n        features['uppercase_count'] > 0\n    ):\n        sarcasm_detected = 1\n    else:\n        sarcasm_detected = 0\n\n    # Return the result as a JSON response\n    return {'sarcasm': sarcasm_detected}\n\n# Example usage\ntext = \"That's just fantastic! I love when my car breaks down on the way to work.\"\nresponse = detect_sarcasm(text)\nprint(response)  # Output: {'sarcasm': 1}\n```\n\n**Explanation:**\n\n1. **Import Libraries:** Import necessary libraries from NLTK for sentiment analysis, stop word removal, and tokenization.\n2. **Preprocessing:**\n   - Tokenize the text into individual words.\n   - Remove stop words (common words like \"a,\" \"the,\" \"is\") to focus on more meaningful words.\n3. **Sentiment Analysis:**\n   - Utilize the VADER sentiment analyzer to get sentiment scores for positive, negative, and compound sentiment.\n4. **Feature Engineering:**\n   - Extract relevant features from the text, such as:\n     - Negative, positive, and compound sentiment scores.\n     - Number of exclamation marks, question marks, and uppercase letters.\n     - Token count (number of meaningful words after stop word removal).\n5. **Sarcasm Detection Rules:**\n   - Combine features and sentiment analysis to create rules for detecting sarcasm. For example:\n     - If the compound sentiment is negative and there are exclamation marks or question marks, it suggests sarcasm.\n6. **JSON Response:**\n   - Format the result as a JSON object with a single key \"sarcasm\" and a value of 0 (no sarcasm) or 1 (sarcasm detected).\n\n**Important Notes:**\n\n- **Accuracy and Precision:** This is a simple rule-based approach and may not be perfectly accurate. More sophisticated techniques like machine learning models might be needed for higher accuracy.\n- **Contextual Understanding:** Sarcasm detection is a complex task that often requires understanding the context of the text. This example doesn't fully address context.\n- **Data:** For a more robust sarcasm detector, you'll need a large dataset of sarcastic and non-sarcastic text to train a machine learning model.\n\n**Improvements:**\n\n- **Machine Learning Models:** Consider training models like Naive Bayes, Logistic Regression, or Support Vector Machines on labeled sarcasm datasets.\n- **Contextual Features:** Include features like word embeddings, part-of-speech tags, and entity recognition to capture contextual information.\n- **Deep Learning:** Explore deep learning models like recurrent neural networks (RNNs) or transformers for more sophisticated analysis.\n- **Ensemble Methods:** Combine multiple detection methods to enhance accuracy.\n",
      ],
    },
  ]
)

@csrf_exempt
def chatbotResponse(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body.decode('utf-8'))
            input_text = body.get('message')

            if not input_text:
                return JsonResponse({'error': 'input_text is required'}, status=400)

            response = chat_session.send_message(input_text)
            cleaned_response_str = response.text.replace('```json\n', '').replace('\n``` \n', '')
    
    # Parse the cleaned string as a JSON object
            try:
                response_data = json.loads(cleaned_response_str)
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON format"}, status=400)

            return JsonResponse(response_data)
            

            
        
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def convert_to_json(response_str):
    lines = response_str.strip().split("\n")
    data = []
    for line in lines:
        parts = line.split(",")
        if len(parts) == 3:
            record = {
                "item": parts[0].strip(),    
                "category": parts[1].strip(),    
                "cost": float(parts[2].strip()) 
            }
            data.append(record)
    return data








  

  
  