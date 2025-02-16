FROM python:3.11-alpine
# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pytesseract sentence-transformers

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "automation:app", "--host", "0.0.0.0", "--port", "8000"]

