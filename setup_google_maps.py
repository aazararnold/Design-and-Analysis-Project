"""
Google Maps API Setup Helper
This script helps you set up and test your Google Maps API key
"""

import requests
import json

def test_google_maps_api(api_key):
    """Test if Google Maps API key is working"""
    print("Testing Google Maps API key...")
    
    # Test Geocoding API
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    geocode_params = {
        'address': 'Times Square, New York, NY',
        'key': api_key
    }
    
    try:
        response = requests.get(geocode_url, params=geocode_params)
        data = response.json()
        
        if data['status'] == 'OK':
            print("✅ Geocoding API: Working")
            location = data['results'][0]['geometry']['location']
            print(f"   Times Square coordinates: {location['lat']}, {location['lng']}")
        else:
            print(f"❌ Geocoding API: {data['status']}")
            return False
    except Exception as e:
        print(f"❌ Geocoding API Error: {e}")
        return False
    
    # Test Distance Matrix API
    distance_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    distance_params = {
        'origins': 'Times Square, New York, NY',
        'destinations': 'Central Park, New York, NY',
        'key': api_key
    }
    
    try:
        response = requests.get(distance_url, params=distance_params)
        data = response.json()
        
        if data['status'] == 'OK':
            print("✅ Distance Matrix API: Working")
            element = data['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                distance = element['distance']['text']
                duration = element['duration']['text']
                print(f"   Times Square to Central Park: {distance}, {duration}")
            else:
                print(f"❌ Distance calculation failed: {element['status']}")
                return False
        else:
            print(f"❌ Distance Matrix API: {data['status']}")
            return False
    except Exception as e:
        print(f"❌ Distance Matrix API Error: {e}")
        return False
    
    print("\n🎉 All APIs working correctly!")
    return True

def get_api_setup_instructions():
    """Print API setup instructions"""
    instructions = """
🔧 Google Maps API Setup Instructions:
=====================================

1. Go to Google Cloud Console:
   https://console.cloud.google.com/

2. Create a new project or select existing one

3. Enable the following APIs:
   • Geocoding API
   • Distance Matrix API
   • Maps JavaScript API (optional, for web integration)

4. Create credentials:
   • Go to "Credentials" in the left menu
   • Click "Create Credentials" → "API Key"
   • Copy your API key

5. Secure your API key (recommended):
   • Click on your API key to edit it
   • Under "API restrictions", select "Restrict key"
   • Choose the APIs you enabled above
   • Under "Application restrictions", you can restrict by IP

6. Set up billing:
   • Google Maps APIs require a billing account
   • You get $200 free credit per month
   • Typical usage for TSP testing is well within free limits

7. Test your API key using this script:
   python setup_google_maps.py YOUR_API_KEY_HERE

💡 Cost Information:
• Geocoding: $5 per 1000 requests (first 40,000 free/month)
• Distance Matrix: $5 per 1000 elements (first 40,000 free/month)
• For TSP with 10 locations: ~100 API calls total
• Monthly free tier covers extensive testing!

🔒 Security Tips:
• Never commit API keys to version control
• Use environment variables in production
• Restrict API key to specific APIs and IPs
• Monitor usage in Google Cloud Console
"""
    print(instructions)

def main():
    """Main function for API setup"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python setup_google_maps.py YOUR_API_KEY")
        print("\nIf you don't have an API key yet:")
        get_api_setup_instructions()
        return
    
    api_key = sys.argv[1]
    
    if test_google_maps_api(api_key):
        print(f"\n✅ Your API key is ready to use!")
        print(f"You can now run the main application with this key.")
    else:
        print(f"\n❌ API key test failed.")
        print(f"Please check your API key and ensure the required APIs are enabled.")
        get_api_setup_instructions()

if __name__ == "__main__":
    main()
