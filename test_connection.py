"""
Standalone diagnostic -- bypasses the google-generativeai library entirely
and makes a raw HTTPS request to Gemini's API with an explicit timeout,
so it CANNOT hang forever like before. This isolates whether the problem
is in the Python library specifically, or something deeper (antivirus
SSL inspection, proxy, etc.) that affects even a basic request.

Usage: python test_connection.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or api_key == "your_key_here":
    print("ERROR: GEMINI_API_KEY not set correctly in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
payload = {
    "contents": [{"parts": [{"text": "Say hello in one word."}]}]
}

print("Sending a test request to Gemini (15 second timeout, will NOT hang)...")

try:
    response = requests.post(url, json=payload, timeout=15)
    print(f"\nStatus code: {response.status_code}")
    print(f"Response body:\n{response.text[:500]}")
except requests.exceptions.Timeout:
    print("\nTIMED OUT after 15 seconds -- the request never got a response.")
    print("This points to something actively interfering with HTTPS traffic")
    print("to this specific domain (antivirus SSL inspection, a proxy, or")
    print("a VPN) rather than a total network block.")
except requests.exceptions.ConnectionError as e:
    print(f"\nCONNECTION ERROR: {e}")
    print("This suggests the connection itself is being actively blocked or reset.")
except Exception as e:
    print(f"\nUNEXPECTED ERROR ({type(e).__name__}): {e}")
