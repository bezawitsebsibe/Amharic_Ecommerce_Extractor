from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import pandas as pd
import os
from data_preprocessing import clean_text, tokenize_amharic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

if not api_id or not api_hash:
    raise ValueError("API_ID or API_HASH is missing in your .env file")

api_id = int(api_id)

session_name = 'ethio_ecom'

# Channel usernames to scrape
channels = [
    '@onlinemarkethio',
    '@Beautyheavenn1',
    '@BFRME',
    '@wecanmakeadeal',
    '@Bethyyonlineshop'
]

# Ensure directories exist
os.makedirs('data/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

output_path = 'data/telegram_data.csv'
preview_path = 'outputs/tokenized_preview.csv'

# Initialize client
client = TelegramClient(session_name, api_id, api_hash)

def fetch_messages():
    all_data = []

    with client:
        for channel in channels:
            print(f"📡 Scraping from {channel}...")
            try:
                for message in client.iter_messages(channel, limit=100):
                    if message.message:
                        msg = message.message
                        image_url = None

                        if message.media and hasattr(message.media, 'photo'):
                            image_path = f"data/images/{channel[1:]}_{message.id}.jpg"
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            client.download_media(message.media, image_path)
                            print(f"🖼️ Downloaded image: {image_path}")
                            image_url = image_path

                        clean_msg = clean_text(msg)
                        tokens = tokenize_amharic(clean_msg)

                        data = {
                            'channel': channel,
                            'text': msg,
                            'timestamp': message.date.isoformat(),
                            'views': message.views or 0,
                            'sender_id': message.sender_id,
                            'image': image_url,
                            'clean_text': clean_msg,
                            'tokens': " ".join(tokens)
                        }

                        all_data.append(data)
            except Exception as e:
                print(f"⚠️ Failed to scrape {channel}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_path, index=False)
        df.head(10).to_csv(preview_path, index=False)
        print(f"✅ Saved {len(df)} messages to {output_path}")
        print(f"✅ Saved preview to {preview_path}")
    else:
        print("⚠️ No data was scraped. Please check your channels or connectivity.")

if __name__ == '__main__':
    fetch_messages()
