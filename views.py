import json
from django.shortcuts import render
from django.http import JsonResponse
import joblib
from django.views.decorators.csrf import csrf_exempt

# Load the machine learning model
model = joblib.load('knn_model.pkl')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))

            # Validate and extract feature values from the request data
            features = [float(data[key]) for key in data]

            # Make prediction using the loaded model
            prediction = model.predict([features])[0]

            # Return prediction as JSON response
            return JsonResponse({'prediction': prediction})
        except Exception as e:
            # Handle any exceptions gracefully
            return JsonResponse({'error': str(e)}, status=400)

    # If request method is not POST, render a form
    return render(request, 'prediction/form.html')
