from pyngrok import ngrok
# !ngrok authtoken [Enter your authtoken here]
# !nohup streamlit run app.py &
ngrok.set_auth_token("2KrjEtr6VnMuVI3PwrjUCKcTNN8_7QdqYaaAVupi4DE65FXzD")

url = ngrok.connect(port = 8080)
print(url) #generates our URL

print(ngrok.get_tunnels())