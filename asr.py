import speech_recognition as sr
from textblob import TextBlob

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def analyze_audio_sentiment(audio_path):
    transcribed_text = transcribe_audio_to_text(audio_path)

    if transcribed_text is not None:
        sentiment_score = analyze_sentiment(transcribed_text)
        if sentiment_score < 0:
            sentiment_type = "Negative"
        elif sentiment_score > 0:
            sentiment_type = "Positive"
        else:
            sentiment_type = "Neutral"
        print(f"The sentiment analysis for the audio file is {sentiment_type}")
        return sentiment_score, sentiment_type
    else:
        return None

audio_path = "diff_audio_output.wav"
transcribed_text = transcribe_audio_to_text(audio_path)
if transcribed_text is not None:
    print(f"Transcribed Text: {transcribed_text}")
result = analyze_audio_sentiment(audio_path)
if result is not None:
    sentiment_score, sentiment_type = result
    print(f"Sentiment Score: {sentiment_score}")
    print(f"Sentiment Type: {sentiment_type}")
