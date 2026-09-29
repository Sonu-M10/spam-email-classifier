import joblib

# Load model and vectorizer
model = joblib.load("results/ai_detection_model.pkl")
vectorizer = joblib.load("results/ai_detection_vectorizer.pkl")

# Unseen test emails
emails = [
    {
        "name": "Normal meeting email",
        "text": """Hi John,

Just wanted to remind you that our meeting is scheduled for tomorrow at 10 AM. Please let me know if you need to reschedule.

Thanks,
Sarah"""
    },
    
    {
        "name": "Normal school email",
        "text": """Hi Professor,

I wanted to ask if I could stop by during office hours this week to discuss my assignment. I have a few questions about the feedback.

Thank you,
Moresh"""
    },
    
    {
        "name": "AI phishing 1",
        "text": """Dear Customer,

We are writing to inform you that your account requires immediate verification. To maintain uninterrupted access to your account, please review and confirm your information using the secure verification process.

Failure to complete this process may result in temporary restrictions on your account.

Thank you for your prompt attention to this matter.

Customer Support Team"""
    },
    
    {
        "name": "AI phishing 2",
        "text": """Dear User,

Our security system has detected unusual activity associated with your account. To protect your information, please verify your identity as soon as possible.

If verification is not completed, access to your account may be temporarily suspended.

Thank you,
Security Support Team"""
    },
    
    {
        "name": "AI phishing 3",
        "text": """Dear Customer,

Your recent payment could not be processed successfully. Please review your payment information and confirm your account details to prevent any interruption to your service.

We appreciate your immediate attention to this matter.

Billing Department"""
    }
]

print("=" * 60)
print("UNSEEN EMAIL TEST")
print("=" * 60)

for email in emails:

    email_tfidf = vectorizer.transform([email["text"]])

    prediction = model.predict(email_tfidf)[0]
    probabilities = model.predict_proba(email_tfidf)[0]

    if prediction == 0:
        result = "Human-written"
        confidence = probabilities[0]
    else:
        result = "AI-generated"
        confidence = probabilities[1]

    print("\n" + "-" * 60)
    print(email["name"])
    print("Prediction:", result)
    print(f"Confidence: {confidence:.2%}")