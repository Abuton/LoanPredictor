from pyngrok import ngrok

public_url = ngrok.connect('8501')
print(public_url)