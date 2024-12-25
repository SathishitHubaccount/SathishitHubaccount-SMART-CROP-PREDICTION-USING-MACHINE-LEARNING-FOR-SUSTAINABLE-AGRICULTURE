import joblib
import pandas as pd
from django.shortcuts import render


def home(request):
    return render(request, 'home.html')

# Load model and scaler
model = joblib.load('naive_bayes_model (1).pkl')
scaler = joblib.load('scaler (1).pkl')

def predict_crop(request):
    if request.method == 'POST':
        try:
            data = {
                'N': float(request.POST['N']),
                'P': float(request.POST['P']),
                'K': float(request.POST['K']),
                'temperature': float(request.POST['temperature']),
                'humidity': float(request.POST['humidity']),
                'ph': float(request.POST['ph']),
                'rainfall': float(request.POST['rainfall']),
            }
        except (ValueError, KeyError):
            # Reload form with an error message if inputs are invalid
            error_message = "All fields are required and must be valid numbers."
            return render(request, 'predictor/form.html', {'error_message': error_message})
        
        # Process the input data
        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)
        prediction = model.predict(scaled_data)
        
        # Mapping prediction back to crop name
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
            6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon',
            10: 'watermelon', 11: 'grapes', 12: 'mango', 13: 'banana',
            14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 
            17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 
            20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
        }
        crop_name = crop_dict[prediction[0]]
        return render(request, 'predictor/result.html', {'prediction': crop_name})
    return render(request, 'predictor/form.html')
