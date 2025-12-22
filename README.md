# Deploy RAGOps
make secret
```
kubectl create secret generic ragapp-secret \
  --from-literal=GEMINI_API_KEY=your_gemini_api_key_here

#config redis

PASSWORD=$(kubectl get secret redis -n redis -o jsonpath='{.data.redis-password}' | base64 -d)

# Tạo secret mới trong namespace serving
kubectl create secret generic redis-password-external \
  --namespace serving \
  --from-literal=redis-password=$PASSWORD

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
        "content": "hoa M207 có giá bao nhiêu."
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


# Monitoring
generate app mail password
```
k create secret generic gmail-auth --from-literal=password='mail-app' -n monitoring
```

# Logging
```
k get secret -n logging

k get secret elasticsearch-master-credentials -n logging -o jsonpath='{.data.password}' | base64 -d

```

# Tracing
tạo GSA
```
gcloud iam service-accounts create tempo-sa

gsutil iam ch serviceAccount:tempo-sa@ragops-481207.iam.gserviceaccount.com:objectAdmin gs://tempo-traces-bucket-grafana
```

# Tao skaffold 
```
skaffold dev --verbosity=info
```