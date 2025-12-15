# Deploy RAGOps
make secret
```
kubectl create secret generic ragapp-secret \
  --from-literal=GEMINI_API_KEY=your_gemini_api_key_here

helm upgrade --install ./deployments/ragapp 
```
test api service
```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "shopping-rag",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful shopping assistant."
      },
      {
        "role": "user",
        "content": "Tôi muốn mua laptop cho lập trình Python."
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```


# Deploy Open WebUI 
```
helm repo add open-webui https://helm.openwebui.com/
helm repo update
# download to modify 
helm search repo open-webui
helm pull open-webui/open-webui --untar
helm upgrade --install open-webui open-webui/open-webui
```

Add host
```
nano /etc/hosts
http://chat.example.com/
```

