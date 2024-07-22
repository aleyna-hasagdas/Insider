FROM tensorflow/tensorflow:latest

LABEL authors="aleynahasagdas"

# Install required packages
RUN pip install --ignore-installed Flask joblib scikit-learn nltk imbalanced-learn

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]
