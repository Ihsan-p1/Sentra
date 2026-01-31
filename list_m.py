import google.generativeai as genai
try:
    genai.configure(api_key="AIzaSyAa_m-XbKJ6YLDWV0TiKi3aEiHNyzM1UXU")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(e)
