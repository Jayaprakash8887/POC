import base64
import os

import openai


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Image summary"""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    result = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
    )
    message = result.choices[0].message.model_dump()
    response = message["content"]
    print("RESPONSE:: ", response)

    return response


# Store base64 encoded images
img_base64_list = []

# Store image summaries
image_summaries = []

# Prompt
prompt = """You are an assistant tasked with rephrasing query and image for retrieval. \
This rephrased query will be embedded and used to retrieve the text information. \
Give a concise rephrased query combining the query and the image contents that is well optimized for retrieval.

query: What activity can I do with this? """

path = "/home/jayaprakash/D/BOT/DJP/sample_images_rag"

# Apply to images
for img_file in sorted(os.listdir(path)):
    print("\nimg_file:: ", img_file)
    img_path = os.path.join(path, img_file)
    base64_image = encode_image(img_path)
    img_base64_list.append(base64_image)
    image_summaries.append(image_summarize(base64_image, prompt))

# print("\n\nimage summaries:: ", image_summaries)
