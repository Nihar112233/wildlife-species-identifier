"""
Script to download authentic sample wildlife images from Wikimedia API with proper headers.
"""

import os
import requests

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_images")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Direct raw image URLs or Wikipedia API titles
WIKI_TITLES = {
    "lion.jpg": "File:Lion_waiting_in_Namibia.jpg",
    "african_elephant.jpg": "File:African_Bush_Elephant.jpg",
    "plains_zebra.jpg": "File:Plains_Zebra_Equus_quagga.jpg",
    "bengal_tiger.jpg": "File:Tiger.50.jpg",
    "brown_bear.jpg": "File:2010-kodiak-bear-1.jpg",
    "giant_panda.jpg": "File:Grosser_Panda.JPG",
    "cheetah.jpg": "File:Cheetah_(Acinonyx_jubatus_jubatus)_female_2.jpg",
    "red_fox.jpg": "File:Fox_-_British_Wildlife_Centre_(17429406401).jpg"
}


def fetch_image_urls():
    headers = {"User-Agent": "WildlifeClassifierEduProject/1.0 (edu_research@college.edu)"}
    session = requests.Session()
    
    for filename, file_title in WIKI_TITLES.items():
        out_path = os.path.join(SAMPLE_DIR, filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"Already exists: {filename}")
            continue
            
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 800,
            "format": "json"
        }
        
        try:
            r = session.get(api_url, params=params, headers=headers, timeout=10)
            data = r.json()
            pages = data["query"]["pages"]
            for _, p_data in pages.items():
                if "imageinfo" in p_data:
                    thumb_url = p_data["imageinfo"][0].get("thumburl") or p_data["imageinfo"][0]["url"]
                    img_data = session.get(thumb_url, headers=headers, timeout=15)
                    with open(out_path, "wb") as f:
                        f.write(img_data.content)
                    print(f"Successfully downloaded {filename} ({len(img_data.content)//1024} KB)")
        except Exception as e:
            print(f"Error fetching {filename}: {e}")


if __name__ == "__main__":
    fetch_image_urls()
