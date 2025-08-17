1. You must set your OpenAI API key before running the app.

Option A: Hardcode in app.py

Open app.py, and at line 30, replace the placeholder with your own OpenAI key:

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


Option B: Set via Command Line (Windows)

Alternatively, you can set your API key as an environment variable:

setx OPENAI_API_KEY "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


2. Run the Project with Docker

To run the app and view the frontend interface:

Step 1: Build the Docker image
docker build -t mini_blinkist .

Step 2: Run the container

If you hardcoded your API key, just run:

docker run -p 5000:5000 mini_blinkist


If you're using the environment variable method, use:

docker run -p 5000:5000 -e OPENAI_API_KEY=%OPENAI_API_KEY% mini_blinkist


On Unix/macOS, replace %OPENAI_API_KEY% with $OPENAI_API_KEY.

3. Open the Web App

Once the container is running, open your browser and navigate to:

http://localhost:5000


You'll be able to upload a PDF and chat with its contents.

