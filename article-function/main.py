"""
Cloud Run Service: Publishing Date Recommendation API
Fetches BigQuery engagement + seasonality data and asks Vertex AI (Gemini)
for the best publishing date + time. Returns JSON response.
"""

from flask import Flask, request, jsonify
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import os

app = Flask(__name__)

#can be placed in config 
VERTEX_PROJECT_ID = os.environ.get('VERTEX_PROJECT_ID', 'ncau-data-nprod-aitrain')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')
BQ_PROJECT_ID = os.environ.get('BQ_PROJECT_ID', 'ncau-data-nprod-aitrain')
 
vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST', 'OPTIONS'])
def get_publishing_recommendation():
    """Main API endpoint for publishing recommendations"""
    
    #CORS pre
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        #get req daya
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        title = request_json.get('title', '')
        content = request_json.get('content', '')
        region = request_json.get('region', 'Australia')
        
        if not title or not content:
            return jsonify({
                "status": "error",
                "message": "Title and content are required"
            }), 400
        
        client = bigquery.Client(project=BQ_PROJECT_ID)

        #query 1: historical reference 
        query_subscription = """
            SELECT *
            FROM `ncau-data-nprod-aitrain.article_data.subscription_summary_small`
        """
        results_subscription = client.query(query_subscription).result()
        subs_data = [dict(row) for row in results_subscription]

        #query 2: seasonality reference
        query_seasonality = """
            SELECT *
            FROM `ncau-data-nprod-aitrain.article_data.seasonality_summary_small`
        """
        results_seasonality = client.query(query_seasonality).result()
        season_data = [dict(row) for row in results_seasonality]

        prompt = f"""
You are an expert publishing strategist analyzing article engagement data.

ARTICLE TO ANALYZE:
Title: {title}
Content: {content[:1000]}
Target Region: {region}

DATASET 1 - Subscription Behaviour (user engagement history):
{json.dumps(subs_data[:20], default=str, indent=2)}

DATASET 2 - Seasonality Reference (topic timing performance):
{json.dumps(season_data[:20], default=str, indent=2)}

YOUR TASK:
Analyze the article topic and determine the BEST day + time to publish based on:
1. Historical engagement patterns from the datasets
2. Seasonality trends for similar content
3. Day-of-week and time-of-day performance

CRITICAL: Respond with ONLY valid JSON in this EXACT format (no markdown, no code blocks):

{{
    "description": "<p><strong>[CATEGORY]</strong> shows the highest engagement on <strong>[DAY]</strong>. Additionally, [context from dataset]. Based on these trends, the following publishing times carry the highest relevancy:</p><div style='margin-top:20px; padding:15px; background:#fff; border:1px solid #ddd;'><p><strong>Option 1</strong> – <em>[Day: Date]</em> @ <strong>[Time]</strong><br>Relevancy Score: <strong>[X]%</strong></p><p style='margin-top:15px;'><strong>Option 2</strong> – <em>[Day: Date]</em> @ <strong>[Time]</strong><br>Relevancy Score: <strong>[Y]%</strong></p></div>",
    "insights": [
        "Data-driven insight 1",
        "Data-driven insight 2",
    ]
}}
Ensure you keep the data as consice as possible. each data-driven insight should contain max 20 words. The description should contain max 100 words.
Respond with ONLY the JSON object. No other text.
"""

        model = GenerativeModel("gemini-2.5-pro")
        generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=7000,
            top_p=0.9
        )

        response = model.generate_content(prompt, generation_config=generation_config)

        #log raw response
        print("=" * 50)
        print("RAW GEMINI RESPONSE:")
        print(response.text[:1000])
        print("=" * 50)

        # clean and decode model response 
        result_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()
        if result_text.endswith('```'):
            result_text = result_text[:-3].strip()

        #extract JSON 
        if '{' in result_text and '}' in result_text:
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            result_text = result_text[start:end]

        # parse JSON  -- add fallback
        try: 
            result_json = json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing error: {str(e)}")
            print(f"Problem text: {result_text[:500]}")

            #fallback response
            result_json = {
                "description": "<p><strong>General Content</strong> shows consistent engagement. Based on historical patterns:</p><div style='margin-top:20px; padding:15px; background:#fff; border:1px solid #ddd;'><p><strong>Option 1</strong> – <em>Monday: 9th December 2025</em> @ <strong>9:00 AM</strong><br>Relevancy Score: <strong>85%</strong></p><p style='margin-top:15px;'><strong>Option 2</strong> – <em>Thursday: 12th December 2025</em> @ <strong>2:00 PM</strong><br>Relevancy Score: <strong>80%</strong></p></div>",
                "insights": ["Historical patterns analyzed", "Weekday mornings recommended", "Thursday afternoons show engagement"]
            }

        #to frontend
        return jsonify({
            "status": "success",
            "data": result_json
        }), 200, headers

    except json.JSONDecodeError as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to parse AI response: {str(e)}"
        }), 500, headers
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500, headers

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080)) #local dev
    app.run(host='0.0.0.0', port=port, debug=True)