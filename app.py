import os
import pickle
import numpy as np
from flask import Flask, render_template, request, send_file  # Import 'send_file' for serving files
import pymongo
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load the trained machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")  #  MongoDB connection URI
db = mongo_client["liver"]
collection = db["liverpre"]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=[ 'POST','GET'])
def index():
       if request.method == 'POST':
        return render_template('index.html', prediction_text="")
       else:
        return render_template('index.html') 
        
        
    
    
@app.route('/instruction', methods=['POST','GET'])
def instruction():
        return render_template('instruction.html')

    
@app.route('/about', methods=['GET', 'POST'])
def about():
     return render_template('about.html')


@app.route('/feedback', methods=['POST','GET'])
def feedback():
        return render_template('feedback.html')
@app.route('/response', methods=['POST','GET'])
def response():
        return render_template('response.html')
def generate_pdf(pdf_path, user_data):
    try:
        with open(pdf_path, 'wb') as pdf_file:
               c = canvas.Canvas(pdf_file)

            # Add content to the PDF using user_data
        c.drawString(100, 750, f"User Name: {user_data['name']}")
        c.drawString(100, 730, f"User Email: {user_data['email']}")
        print(f"PDF generated successfully at {pdf_path}")

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        Age = float(request.form['Age'])
        Gender = int(request.form['Gender'])
        AlcoholIntake = float(request.form['AlcoholIntake'])
        BMI = float(request.form['BMI'])
        DrugUse = int(request.form['DrugUse'])
        SmokingStatus = float(request.form['SmokingStatus'])
        StressLevels = float(request.form['StressLevels'])
        
        # Preprocess the user input (scaling, etc.) - same preprocessing as your training data
        input_data = np.array([Age, Gender, AlcoholIntake, BMI, DrugUse, SmokingStatus, StressLevels]).reshape(1, -1)

        # Make a prediction using the loaded SVM model
        prediction = model.predict(input_data)

        # Display the prediction (you can modify this part to display it as needed)
        if prediction[0] == 1:
            prediction_text = "Liver Disease Detected"
        else:
            prediction_text = "No Liver Disease Detected"

        # Save the user input and prediction to MongoDB
        user_data = {
            "Age": Age,
            "Gender": Gender,
            "AlcoholIntake": AlcoholIntake,
            "BMI": BMI,
            "DrugUse": DrugUse,
            "SmokingStatus": SmokingStatus,
            "StressLevels": StressLevels,
            "Prediction": prediction_text
        }

        # Insert the user data into the MongoDB collection
        collection.insert_one(user_data)
       

        try:
            return render_template('result.html', prediction_text=prediction_text)
        except Exception as e:
            print(f"Error: {str(e)}")
            return "An error occurred while making predictions."
        
@app.route('/download_pdf')
def download_pdf():
    pdf_path = 'path_to_static_pdf_file.pdf'

    # Check if the file exists before sending it
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name='filename.pdf')
    else:
        return "PDF file not found."

        
 

   


if __name__ == '__main__':
    app.run(debug=True)
